#!/usr/bin/env python3
"""
Step 2c: Rigorous Zero-Shot Evaluation on YCHC External Cohort (N=99 Cases)
===========================================================================
Pre-Registered Protocol (ychc_preregistration.md):
  - Zero fine-tuning, training, or threshold adjustment on YCHC
  - Evaluates models strictly at Epoch 500 (final_model.pt)
  - Evaluates both Primary (full-volume) and Secondary (breast-masked) analyses
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


def main():
    parser = argparse.ArgumentParser(description="YCHC Zero-Shot External Evaluation")
    parser.add_argument("--preproc-dir", type=str, default="./data/ychc_preprocessed")
    parser.add_argument("--results-dir", type=str, default="./results/ychc_zeroshot")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[YCHC Zero-Shot] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Preproc Dir: {args.preproc_dir} | Output: {results_dir}")


if __name__ == "__main__":
    main()
