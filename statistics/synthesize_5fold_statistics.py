#!/usr/bin/env python3
"""
Synthesize 5-Fold Evaluation Results & Compute Paired Wilcoxon Signed-Rank Tests
================================================================================
Usage:
  python synthesize_5fold_statistics.py --results-dir results/
Outputs:
  results/final_5fold_synthesis.json
  Prints complete summary table with:
    1. Overall 5-fold CV & Held-Out Test Mean ± SD (N=306)
    2. Cohort-Stratified Subgroup Performance (DUKE, ISPY2, ISPY1, NACT)
    3. Paired Wilcoxon Signed-Rank Tests with Holm-Bonferroni correction
"""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats

MODES = ["standard", "pointwise_deep", "boundary_matched", "boundary_full", "segresnet", "no_scse", "kg_scse"]
COHORTS = ["DUKE", "ISPY2", "ISPY1", "NACT"]


def holm_bonferroni(p_values):
    """Applies Holm-Bonferroni step-down procedure for FWER control."""
    p_arr = np.array(p_values)
    n = len(p_arr)
    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]

    adj_p = np.empty(n)
    for i, p in enumerate(sorted_p):
        adj_p[i] = min(1.0, p * (n - i))
    # Enforce monotonicity
    for i in range(1, n):
        adj_p[i] = max(adj_p[i], adj_p[i - 1])

    out = np.empty(n)
    out[sorted_indices] = adj_p
    return list(out)


def main():
    parser = argparse.ArgumentParser(description="5-fold CV and Test Synthesis")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Results directory")
    parser.add_argument("--out-json", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    results_dir = args.results_dir
    out_json = args.out_json or (results_dir / "final_5fold_synthesis.json")

    print("==========================================================================")
    print("  5-FOLD CROSS-VALIDATION & HELD-OUT TEST SYNTHESIS (N=306 Test Patients)")
    print("==========================================================================")

    data = {}
    for mode in MODES:
        fold_results = []
        for f in range(5):
            eval_file = results_dir / f"eval_test_{mode}_fold_{f}.json"
            if eval_file.exists():
                with open(eval_file) as fp:
                    fold_results.append(json.load(fp))
        if fold_results:
            data[mode] = fold_results

    if not data:
        print("[warn] No completed 5-fold evaluation JSONs found.")
        return

    summary_table = {}
    cohort_table = {c: {} for c in COHORTS}
    test_per_case_by_mode = {}

    for mode, f_list in data.items():
        n_completed = len(f_list)
        pids = [c["pid"] for c in f_list[0]["per_case"]]

        case_dices = []
        for i in range(len(pids)):
            d = np.mean([f["per_case"][i]["dsc"] * (100.0 if f["per_case"][i]["dsc"] <= 1.0 else 1.0) for f in f_list])
            case_dices.append(d)

        test_per_case_by_mode[mode] = case_dices

        summary_table[mode] = {
            "completed_folds": n_completed,
            "test_dice_mean": float(np.mean(case_dices)),
            "test_dice_std": float(np.std(case_dices)),
            "test_dice_median": float(np.median(case_dices)),
        }

        # Subgroup analysis by cohort
        for c in COHORTS:
            c_indices = [i for i, pid in enumerate(pids) if pid.startswith(c)]
            if c_indices:
                c_dices = [case_dices[i] for i in c_indices]
                cohort_table[c][mode] = {
                    "n": len(c_indices),
                    "mean": float(np.mean(c_dices)),
                    "std": float(np.std(c_dices)),
                    "median": float(np.median(c_dices))
                }

    # Paired Wilcoxon Tests against Standard scSE
    wilcoxon_results = {}
    if "standard" in test_per_case_by_mode:
        std_cases = test_per_case_by_mode["standard"]
        raw_p = []
        tested_modes = []
        for mode in MODES:
            if mode != "standard" and mode in test_per_case_by_mode:
                cand_cases = test_per_case_by_mode[mode]
                diff = np.array(cand_cases) - np.array(std_cases)
                if len(diff[diff != 0]) > 0:
                    w = stats.wilcoxon(diff, zero_method="wilcox")
                    raw_p.append(float(w.pvalue))
                    tested_modes.append(mode)
                    wilcoxon_results[mode] = {
                        "stat": float(w.statistic),
                        "raw_p": float(w.pvalue)
                    }
        if raw_p:
            adj_p = holm_bonferroni(raw_p)
            for m, ap in zip(tested_modes, adj_p):
                wilcoxon_results[m]["holm_p"] = float(ap)

    print(f"\n1. OVERALL TEST COHORT PERFORMANCE (N=306 Patients):")
    print(f"{'Architecture Variant':<22} {'Folds':<8} {'Test Dice (Mean ± SD)':<24} {'Median':<10} {'Wilcoxon p vs Standard'}")
    print("-" * 88)
    for mode in MODES:
        if mode in summary_table:
            st = summary_table[mode]
            wp_str = "Reference" if mode == "standard" else (
                f"p={wilcoxon_results[mode]['holm_p']:.4e} (Holm)" if mode in wilcoxon_results else "Pending"
            )
            print(f"{mode:<22} {st['completed_folds']}/5     {st['test_dice_mean']:.2f} ± {st['test_dice_std']:.2f}%      {st['test_dice_median']:.2f}%     {wp_str}")

    print(f"\n2. STRATIFIED PERFORMANCE BY COHORT (Mean ± SD Dice, %):")
    print(f"{'Cohort':<10} {'N':<6} {'Standard':<18} {'Pointwise-Deep':<18} {'Boundary-Matched':<18} {'Boundary-Full':<18}")
    print("-" * 88)
    for c in COHORTS:
        if cohort_table[c]:
            n_c = list(cohort_table[c].values())[0]["n"]
            st_str = f"{cohort_table[c]['standard']['mean']:.2f} ± {cohort_table[c]['standard']['std']:.2f}" if "standard" in cohort_table[c] else "N/A"
            pw_str = f"{cohort_table[c]['pointwise_deep']['mean']:.2f} ± {cohort_table[c]['pointwise_deep']['std']:.2f}" if "pointwise_deep" in cohort_table[c] else "N/A"
            bm_str = f"{cohort_table[c]['boundary_matched']['mean']:.2f} ± {cohort_table[c]['boundary_matched']['std']:.2f}" if "boundary_matched" in cohort_table[c] else "N/A"
            bf_str = f"{cohort_table[c]['boundary_full']['mean']:.2f} ± {cohort_table[c]['boundary_full']['std']:.2f}" if "boundary_full" in cohort_table[c] else "N/A"
            print(f"{c:<10} {n_c:<6} {st_str:<18} {pw_str:<18} {bm_str:<18} {bf_str:<18}")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fp:
        json.dump({
            "overall_summary": summary_table,
            "cohort_stratified": cohort_table,
            "wilcoxon_vs_standard": wilcoxon_results
        }, fp, indent=2)
    print(f"\n[done] Saved synthesis to {out_json}")


if __name__ == "__main__":
    main()
