#!/usr/bin/env python3
"""
Evaluate nnU-Net v2 Fold 0 on YCHC External Cohort (N=99 Cases)
==============================================================
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd
)

TARGET_SPACING = (0.8, 0.8, 1.0)


def compute_all_metrics(p_bin, g_bin, spacing=TARGET_SPACING):
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

    return dsc, iou, prec, rec, hd95, assd, empty_pred


def main():
    parser = argparse.ArgumentParser(description="Evaluate nnU-Net on YCHC")
    parser.add_argument("--preproc-dir", type=str, default="./data/ychc_preprocessed")
    parser.add_argument("--raw-dir", type=str, default="./data/sample_yunnan_hospital")
    parser.add_argument("--out-dir", type=str, default="./results/ychc")
    parser.add_argument("--pred-dir", type=str, default="./data/nnunet_data/ychc_predictions_fold0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[nnU-Net YCHC] Out Dir: {out_dir}")


if __name__ == "__main__":
    main()
