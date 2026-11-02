#!/usr/bin/env python3
"""
TCIA AMBL External Zero-Shot Evaluation (Phase B)
=================================================
Pre-Registered Protocol (ambl_preregistration.md):
  - Cohort: 13 Evaluable Malignant Cases with complete multi-phase series
  - Input: Dynamic Contrast-Enhanced MRI
  - Checkpoints: Evaluated strictly at Epoch 500 (final_model.pt)

Configurations:
  (a) best_modelfinal55.pth (2-ch) with [P0, P1]
  (b) BEAT Net 5-ch ensemble (5 x final_model.pt at Epoch 500) with [P0, P1, P2, P1-P0, P2-P0]
  (c) Phase-randomised 2-ch model (final_model.pt at Epoch 500) with [P0, Pk] (k=1..4), APCPS, early-peak
  (d) Conventional 2-ch fold 4 (Epoch 500) with [P0, P2]
  (e) nnU-Net v2 fold 0 with [P0, P1, P2]
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance
import nibabel as nib

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.attention_55_model import AttentionAllDecoderUNet
from models.attention_isolated_model import get_isolated_model

TARGET_SPACING = (0.8, 0.8, 1.0)
CASES = [
    "AMBL-003", "AMBL-005", "AMBL-007", "AMBL-008", "AMBL-011",
    "AMBL-016", "AMBL-018", "AMBL-022", "AMBL-028", "AMBL-031",
    "AMBL-032", "AMBL-038", "AMBL-047"
]


def zscore_norm(vol):
    mask = vol != 0
    if not np.any(mask):
        return vol.astype(np.float32)
    mean = vol[mask].mean()
    std = vol[mask].std()
    if std < 1e-6:
        std = 1.0
    return ((vol - mean) / std).astype(np.float32)


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


def main():
    parser = argparse.ArgumentParser(description="TCIA AMBL Zero-Shot Evaluation")
    parser.add_argument("--preproc-dir", type=str, default="./data/ambl_preprocessed")
    parser.add_argument("--clean-dir", type=str, default="./data/ambl_clean")
    parser.add_argument("--out-dir", type=str, default="./results/ambl")
    parser.add_argument("--ckpt-55", type=str, default=None)
    parser.add_argument("--beatnet-dir", type=str, default="./results/5fold_5chan")
    parser.add_argument("--conv2ch-ckpt", type=str, default="./results/5fold_2chan/fold_4/final_model.pt")
    parser.add_argument("--phase-rand-ckpt", type=str, default="./results/phase_randomized_2chan_seed0/final_model.pt")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[AMBL Evaluation] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"[AMBL Evaluation] Preprocessed Dir: {args.preproc_dir}")


if __name__ == "__main__":
    main()
