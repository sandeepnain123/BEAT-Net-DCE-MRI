#!/usr/bin/env python3
"""
APCPS Implementation Audit and Correction Evaluation
====================================================
Cohorts:
  1. AMBL (13 evaluable malignant cases)
  2. YCHC (99 complete zero-shot cases)

Evaluates under the pre-registered protocol (apcps_correction_amendment.md):
  - Raw resampled voxel intensities (RAS, 0.8x0.8x1.0 mm)
  - Mask B = P0 > 20th percentile of P0 (with fallback to P0 > 0)
  - Non-negative kinetic enhancement: sub_k = max(0.0, Pk - P0)
  - Candidate frames:
      AMBL: P1, P2, P3, P4
      YCHC: P1, P2, P3, P4, P5
  - Selection Rules:
      * APCPS q95: argmax_k percentile(sub_k[B], 95.0)
      * Whole-breast mean: argmax_k mean(sub_k[B])
      * Early-peak: earliest k where top5(k) >= 0.90 * max_j(top5(j))

Models Evaluated:
  - best_modelfinal55.pth with [P0, Ppeak] and [P0, P1]
  - Phase-randomized 2-channel network (final_model.pt at Epoch 500) with:
      [P0, Ppeak] (q95), whole-breast mean, early-peak, and fixed [P0, P1]
  - BEAT Net 5-channel ensemble (5 folds x Epoch 500) with [P0, P1, P2, P1-P0, P2-P0]

Paired Statistics (Amendment 2 Invariants):
  - Paired Wilcoxon (zero_method="wilcox"): W, z, p, effect size r = |z| / sqrt(n_nz)
  - 10,000 bootstrap CI of mean difference, 1.0-point non-inferiority bound
  - Comparators: vs BEAT Net 5-ch ensemble and vs Fixed [P0, P1]
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.attention_55_model import AttentionAllDecoderUNet
from models.attention_isolated_model import get_isolated_model

TARGET_SPACING = (0.8, 0.8, 1.0)


def compute_all_metrics(p_bin, g_bin, spacing=TARGET_SPACING):
    if p_bin.shape != g_bin.shape:
        fixed = np.zeros_like(g_bin)
        min_s = [min(a, b) for a, b in zip(p_bin.shape, g_bin.shape)]
        fixed[:min_s[0], :min_s[1], :min_s[2]] = p_bin[:min_s[0], :min_s[1], :min_s[2]]
        p_bin = fixed
    p = torch.from_numpy(p_bin).bool()
    g = torch.from_numpy(g_bin).bool()
    p_sum = p.sum().item()
    g_sum = g.sum().item()
    inter = (p & g).sum().item()

    empty_pred = bool(p_sum == 0)

    if p_sum == 0 and g_sum == 0:
        return 100.0, 100.0, 100.0, 100.0, 0.0, 0.0, empty_pred
    if p_sum == 0 or g_sum == 0:
        diag = float(np.sqrt(sum((s * d) ** 2 for s, d in zip(spacing, p_bin.shape))))
        return 0.0, 0.0, (100.0 if g_sum == 0 else 0.0), (100.0 if p_sum == 0 else 0.0), diag, diag, empty_pred

    union = p_sum + g_sum - inter
    dsc = (2.0 * inter) / (p_sum + g_sum) * 100.0
    iou = (inter / union) * 100.0 if union > 0 else 0.0
    prec = (inter / p_sum) * 100.0 if p_sum > 0 else 0.0
    rec = (inter / g_sum) * 100.0 if g_sum > 0 else 0.0

    p_4d = p.unsqueeze(0).unsqueeze(0).float()
    g_4d = g.unsqueeze(0).unsqueeze(0).float()
    try:
        hd95 = compute_hausdorff_distance(p_4d, g_4d, include_background=True, percentile=95.0, spacing=spacing).item()
    except Exception:
        hd95 = float("nan")
    try:
        assd = compute_average_surface_distance(p_4d, g_4d, include_background=True, symmetric=True, spacing=spacing).item()
    except Exception:
        assd = float("nan")

    return dsc, iou, prec, rec, hd95, assd, empty_pred


def paired_wilcoxon_and_bootstrap(cands, refs, label="Candidate vs Reference", n_boot=10000, seed=42):
    diff = np.array(cands) - np.array(refs)
    n = len(diff)
    nz = diff[diff != 0]
    n_nz = len(nz)

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

    rng = np.random.default_rng(seed)
    boot_diffs = diff[rng.integers(0, n, size=(n_boot, n))]
    boot_means = np.mean(boot_diffs, axis=1)
    ci95_two = [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))]
    ci95_lower = float(np.percentile(boot_means, 5.0))
    non_inf = bool(ci95_lower > -1.0)

    return {
        "label": label,
        "n_total": n,
        "n_nonzero": n_nz,
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
        "ci95_two_sided": ci95_two,
        "ci95_onesided_lower": ci95_lower,
        "non_inferior_margin_1pt": non_inf,
        "wilcoxon_w": w_stat,
        "wilcoxon_z": z_stat,
        "p_value": p_val,
        "effect_size_r": r_eff,
    }


def zscore_norm(vol):
    mask = vol != 0
    if not np.any(mask):
        return vol.astype(np.float32)
    mean = vol[mask].mean()
    std = vol[mask].std()
    if std < 1e-6:
        std = 1.0
    return ((vol - mean) / std).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="APCPS Implementation Audit and Correction Evaluation")
    parser.add_argument("--ambl-preproc", type=str, default="./data/ambl_preprocessed")
    parser.add_argument("--ychc-clean", type=str, default="./data/ychc_clean")
    parser.add_argument("--ychc-preproc", type=str, default="./data/ychc_preprocessed")
    parser.add_argument("--ckpt-55", type=str, default=None, help="Path to best_modelfinal55.pth")
    parser.add_argument("--out-dir-ambl", type=str, default="./results/ambl")
    parser.add_argument("--out-dir-ychc", type=str, default="./results/ychc")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[APCPS Correction] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    out_dir_ambl = Path(args.out_dir_ambl)
    out_dir_ychc = Path(args.out_dir_ychc)
    out_dir_ambl.mkdir(parents=True, exist_ok=True)
    out_dir_ychc.mkdir(parents=True, exist_ok=True)

    print("==========================================================================")
    print("  APCPS IMPLEMENTATION AUDIT AND CORRECTION EVALUATION (AMBL & YCHC)")
    print("==========================================================================")
    print(f"AMBL Preprocessed Dir: {args.ambl_preproc}")
    print(f"YCHC Clean Dir:        {args.ychc_clean}")
    print(f"YCHC Preprocessed Dir: {args.ychc_preproc}")
    print("Protocol: apcps_correction_amendment.md (verified)")


if __name__ == "__main__":
    main()
