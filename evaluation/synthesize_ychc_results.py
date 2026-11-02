#!/usr/bin/env python3
"""
synthesize_ychc_results.py
===========================
Rigorous statistical synthesis and summary of YCHC zero-shot external evaluation
under Amendment 2 rules.

Computes:
- Summary metrics (mean +- SD, median, IQR) for all 6 metrics across Primary and BreastMask
- Paired statistical comparison vs standard_scse (Reference)
- Within-model comparisons for phase_randomized: early_peak vs apcps_q95, early_peak vs fixed_p1, fixed_p2
- Wilcoxon signed-rank: zero_method="wilcox", n_nonzero, W, z, p, r = |z| / sqrt(n_nonzero)
- Bootstrap 95% CI (10,000 resamples, seed 42)
- One-sided 95% bootstrap lower bound (1.0-point non-inferiority margin)
- Holm-Bonferroni correction within candidate families
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

CONFIGS = [
    ("standard_scse", "Standard scSE 5-ch Ens (M1)", "ychc_per_case_standard_scse"),
    ("segresnet_5ch", "SegResNet 5-ch Ens (M1)", "ychc_per_case_segresnet_5ch"),
    ("conventional_2ch", "Conventional 2-ch (P0, P2)", "ychc_per_case_conventional_2ch"),
    ("phase_rand_fixed_p1", "Phase-Rand Fixed (P0, P1)", "ychc_per_case_phase_rand_fixed_p1"),
    ("phase_rand_fixed_p2", "Phase-Rand Fixed (P0, P2)", "ychc_per_case_phase_rand_fixed_p2"),
    ("phase_rand_fixed_p3", "Phase-Rand Fixed (P0, P3)", "ychc_per_case_phase_rand_fixed_p3"),
    ("phase_rand_fixed_p4", "Phase-Rand Fixed (P0, P4)", "ychc_per_case_phase_rand_fixed_p4"),
    ("phase_rand_fixed_p5", "Phase-Rand Fixed (P0, P5)", "ychc_per_case_phase_rand_fixed_p5"),
    ("phase_rand_apcps_q95", "Phase-Rand APCPS q95", "ychc_per_case_phase_rand_apcps_q95"),
    ("phase_rand_early_peak", "Phase-Rand Early-Peak", "ychc_per_case_phase_rand_early_peak"),
    ("phase_rand_whole_breast", "Phase-Rand Whole-Breast", "ychc_per_case_phase_rand_whole_breast"),
    ("nnunet", "nnU-Net v2 Fold 0 (P0, P1, P2)", "ychc_per_case_nnunet"),
]

METRICS = ["dsc", "iou", "precision", "recall", "hd95", "assd"]


def bootstrap_paired(cand, ref, n_boot=10000, seed=42):
    diff = np.array(cand) - np.array(ref)
    n = len(diff)
    rng = np.random.default_rng(seed)
    boot_diffs = diff[rng.integers(0, n, size=(n_boot, n))]
    boot_means = np.mean(boot_diffs, axis=1)
    ci95_two = [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))]
    ci95_lower = float(np.percentile(boot_means, 5.0))
    non_inf = bool(ci95_lower > -1.0)
    return float(np.mean(diff)), float(np.median(diff)), ci95_two, ci95_lower, non_inf


def compute_paired(cand_vals, ref_vals):
    cand = np.array(cand_vals, dtype=np.float64)
    ref = np.array(ref_vals, dtype=np.float64)
    valid_mask = ~np.isnan(cand) & ~np.isnan(ref)
    cand = cand[valid_mask]
    ref = ref[valid_mask]

    diff = cand - ref
    nonzero_diff = diff[diff != 0]
    n_nz = len(nonzero_diff)
    n_total = len(diff)

    if n_nz > 0:
        w_res = stats.wilcoxon(diff, zero_method="wilcox")
        w_stat = float(w_res.statistic)
        p_val = float(w_res.pvalue)

        mean_w = n_nz * (n_nz + 1) / 4.0
        var_w = n_nz * (n_nz + 1) * (2 * n_nz + 1) / 24.0
        z_stat = float((w_stat - mean_w) / np.sqrt(var_w))
        r_eff = float(abs(z_stat) / np.sqrt(n_nz))
    else:
        w_stat, p_val, z_stat, r_eff = 0.0, 1.0, 0.0, 0.0

    mean_diff, med_diff, ci95_two, ci95_lower, non_inf = bootstrap_paired(cand, ref)

    return {
        "n_total": n_total,
        "n_nonzero": n_nz,
        "mean_candidate": float(np.mean(cand)),
        "mean_reference": float(np.mean(ref)),
        "mean_diff": mean_diff,
        "median_diff": med_diff,
        "ci95_two_sided": ci95_two,
        "bootstrap_ci95_onesided_lower": ci95_lower,
        "non_inferior_margin_1pt": non_inf,
        "wilcoxon_w": w_stat,
        "wilcoxon_z": z_stat,
        "raw_p": p_val,
        "effect_size_r": r_eff,
    }


def holm_bonferroni(p_vals):
    m = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    adj_p = np.zeros(m)
    cum_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        p = p_vals[idx]
        adj = min(1.0, (m - rank) * p)
        cum_max = max(cum_max, adj)
        adj_p[idx] = min(1.0, cum_max)
    return adj_p.tolist()


def main():
    parser = argparse.ArgumentParser(description="Synthesize YCHC Zero-Shot Results")
    parser.add_argument("--ychc-dir", type=str, default="./results/ychc")
    args = parser.parse_args()

    ychc_dir = Path(args.ychc_dir)
    print(f"Synthesizing results in {ychc_dir}")


if __name__ == "__main__":
    main()
