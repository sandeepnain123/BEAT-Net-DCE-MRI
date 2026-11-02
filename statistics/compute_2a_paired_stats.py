#!/usr/bin/env python3
"""
Compute paired statistics between Section 3 experiments (e.g., 3.1_raw3ch vs 3.0_ref)
using Wilcoxon signed-rank test and 10,000 bootstrap resamples.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def compute_paired(candidate_csv, ref_csv, label):
    df_c = pd.read_csv(candidate_csv).sort_values("pid")
    df_r = pd.read_csv(ref_csv).sort_values("pid")
    assert len(df_c) == len(df_r), f"Length mismatch: {len(df_c)} vs {len(df_r)}"
    assert (df_c["pid"].values == df_r["pid"].values).all(), "PIDs mismatch!"

    results = {}
    metrics = ["dsc", "iou", "precision", "recall", "hd95", "assd"]

    # In per_case CSVs, check if dsc is 0-1 or 0-100
    c_mult = 100.0 if df_c["dsc"].mean() <= 1.0 else 1.0
    r_mult = 100.0 if df_r["dsc"].mean() <= 1.0 else 1.0

    print(f"\n==========================================================================")
    print(f"  PAIRED ANALYSIS: {label}")
    print(f"==========================================================================")

    for m in metrics:
        if m not in df_c.columns or m not in df_r.columns:
            continue
        cand_vals = df_c[m].values.astype(float) * (c_mult if m in ["dsc", "iou", "precision", "recall"] else 1.0)
        ref_vals = df_r[m].values.astype(float) * (r_mult if m in ["dsc", "iou", "precision", "recall"] else 1.0)

        # Handle NaNs if any
        valid = (~np.isnan(cand_vals)) & (~np.isnan(ref_vals))
        cand_vals = cand_vals[valid]
        ref_vals = ref_vals[valid]

        diff = cand_vals - ref_vals
        mean_diff = float(np.mean(diff))
        median_diff = float(np.median(diff))

        # Non-zero differences for Wilcoxon
        nonzero_diff = diff[diff != 0]
        n_nonzero = len(nonzero_diff)

        if n_nonzero > 0:
            w_res = stats.wilcoxon(diff, zero_method="wilcox")
            w_stat = float(w_res.statistic)
            p_val = float(w_res.pvalue)

            mean_w = n_nonzero * (n_nonzero + 1) / 4.0
            var_w = n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24.0
            z_stat = float((w_stat - mean_w) / np.sqrt(var_w))
            effect_r = float(abs(z_stat) / np.sqrt(n_nonzero))
        else:
            w_stat, p_val, z_stat, effect_r = 0.0, 1.0, 0.0, 0.0

        # Nonparametric bootstrap 95% CI (10,000 resamples)
        rng = np.random.default_rng(42)
        n_cases = len(diff)
        boot_idx = rng.integers(0, n_cases, size=(10000, n_cases))
        boot_diffs = diff[boot_idx]
        boot_means = np.mean(boot_diffs, axis=1)

        ci_two_sided = [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))]
        # One-sided 95% lower bound = 5th percentile
        ci_lower_onesided = float(np.percentile(boot_means, 5.0))
        non_inferior_established = bool(ci_lower_onesided > -1.0) if m == "dsc" else None

        results[m] = {
            "cand_mean": float(np.mean(cand_vals)),
            "ref_mean": float(np.mean(ref_vals)),
            "mean_diff": mean_diff,
            "median_diff": median_diff,
            "n_total": len(diff),
            "n_nonzero": n_nonzero,
            "wilcoxon_w": w_stat,
            "wilcoxon_z": z_stat,
            "p_value": p_val,
            "effect_size_r": effect_r,
            "bootstrap_ci95_two_sided": ci_two_sided,
            "bootstrap_ci95_onesided_lower": ci_lower_onesided,
            "non_inferior_established": non_inferior_established
        }

        print(f"  {m:10s} | Cand: {np.mean(cand_vals):.2f} | Ref: {np.mean(ref_vals):.2f} | Diff: {mean_diff:+.3f} | 95% CI: [{ci_two_sided[0]:.3f}, {ci_two_sided[1]:.3f}] | W: {w_stat:.1f} | z: {z_stat:.3f} | p: {p_val:.4f} | r: {effect_r:.3f} | n_nz: {n_nonzero}")
        if m == "dsc":
            print(f"             -> Non-inferiority: Lower 95% bound = {ci_lower_onesided:.3f} (> -1.0? {non_inferior_established})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Paired statistical comparison")
    parser.add_argument("--base-dir", type=Path, default=Path("results/ablation_section3"), help="Base directory for section 3 results")
    parser.add_argument("--candidate-csv", type=Path, default=None, help="Direct candidate CSV path")
    parser.add_argument("--ref-csv", type=Path, default=None, help="Direct reference CSV path")
    parser.add_argument("--out-json", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    if args.candidate_csv and args.ref_csv:
        res = compute_paired(args.candidate_csv, args.ref_csv, f"{args.candidate_csv.stem} vs {args.ref_csv.stem}")
        if args.out_json:
            with open(args.out_json, "w") as fp:
                json.dump(res, fp, indent=2)
            print(f"[done] Saved to {args.out_json}")
    else:
        base = args.base_dir
        res_s0 = compute_paired(base / "3.1_raw3ch_seed0/test_per_case_3.1_raw3ch_seed0.csv", base / "3.0_ref_seed0/test_per_case_3.0_ref_seed0.csv", "3.1_raw3ch (Seed 0) vs 3.0_ref (Seed 0)")
        res_s1 = compute_paired(base / "3.1_raw3ch_seed1/test_per_case_3.1_raw3ch_seed1.csv", base / "3.0_ref_seed1/test_per_case_3.0_ref_seed1.csv", "3.1_raw3ch (Seed 1) vs 3.0_ref (Seed 1)")

        out_path = args.out_json or (base / "paired_stats_sec3_raw3_vs_ref.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fp:
            json.dump({"seed_0": res_s0, "seed_1": res_s1}, fp, indent=2)

        print(f"\n[done] Saved paired statistics to {out_path}")


if __name__ == "__main__":
    main()
