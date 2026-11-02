#!/usr/bin/env python3
"""
Refresh fold_ensembles_2a.json and stats_vs_standard.json
Includes no_scse (folds 3, 4) and kg_scse (folds 3, 4) alongside 5-fold models.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import torch
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance

TARGET_SPACING = (0.8, 0.8, 1.0)


def compute_all_metrics(p_bin, g_bin, spacing=TARGET_SPACING):
    p = torch.from_numpy(p_bin).bool()
    g = torch.from_numpy(g_bin).bool()
    p_sum = p.sum().item()
    g_sum = g.sum().item()
    inter = (p & g).sum().item()

    if p_sum == 0 and g_sum == 0:
        return 100.0, 100.0, 100.0, 100.0, 0.0, 0.0
    if p_sum == 0 or g_sum == 0:
        diag = float(np.sqrt(sum((s * d) ** 2 for s, d in zip(spacing, p_bin.shape))))
        return 0.0, 0.0, (100.0 if g_sum == 0 else 0.0), (100.0 if p_sum == 0 else 0.0), diag, diag

    union = p_sum + g_sum - inter
    dsc = (2.0 * inter) / (p_sum + g_sum) * 100.0
    iou = (inter / union) * 100.0 if union > 0 else 0.0
    prec = (inter / p_sum) * 100.0 if p_sum > 0 else 0.0
    rec = (inter / g_sum) * 100.0 if g_sum > 0 else 0.0

    idx = torch.where(p | g)
    pad = 4
    hmin, hmax = max(0, idx[0].min().item() - pad), min(p_bin.shape[0], idx[0].max().item() + pad + 1)
    wmin, wmax = max(0, idx[1].min().item() - pad), min(p_bin.shape[1], idx[1].max().item() + pad + 1)
    dmin, dmax = max(0, idx[2].min().item() - pad), min(p_bin.shape[2], idx[2].max().item() + pad + 1)

    p_crop = p[hmin:hmax, wmin:wmax, dmin:dmax].unsqueeze(0).unsqueeze(0).float()
    g_crop = g[hmin:hmax, wmin:wmax, dmin:dmax].unsqueeze(0).unsqueeze(0).float()

    try:
        hd95 = compute_hausdorff_distance(p_crop, g_crop, include_background=True, percentile=95.0, spacing=spacing).item()
    except Exception:
        hd95 = float("nan")
    try:
        assd = compute_average_surface_distance(p_crop, g_crop, include_background=True, symmetric=True, spacing=spacing).item()
    except Exception:
        assd = float("nan")

    return dsc, iou, prec, rec, hd95, assd


def bootstrap_ci_diff(x, y, n_boot=10000, ci=95.0, seed=42):
    diff = np.array(x) - np.array(y)
    rng = np.random.default_rng(seed)
    boot_means = [np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(n_boot)]
    alpha = (100.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha))
    upper = float(np.percentile(boot_means, 100.0 - alpha))
    return float(np.mean(diff)), lower, upper


def main():
    parser = argparse.ArgumentParser(description="Update 2A & 2B results")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--mmap-dir", type=str, default="./data/preprocessed_mmap")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    mmap_dir = Path(args.mmap_dir)
    print(f"Results dir: {results_dir} | Mmap dir: {mmap_dir}")


if __name__ == "__main__":
    main()
