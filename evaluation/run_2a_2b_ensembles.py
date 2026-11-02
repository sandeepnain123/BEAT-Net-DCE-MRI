#!/usr/bin/env python3
"""
Section 2A & 2B: Fold Ensembles and Paired Statistical Analysis on 306 Test Cases
================================================================================
Models evaluated:
  1. standard_scse (5-fold ensemble at Epoch 500)
  2. pointwise_deep (5-fold ensemble)
  3. matched_3x3x3 (5-fold ensemble)
  4. full_3x3x3 (5-fold ensemble)
  5. segresnet_5ch (5-fold ensemble)

Computations:
  2A:
    - Single-fold metrics: mean +- SD across folds, and mean of single-fold models per case
    - 5-fold ensemble: average sigmoid probabilities of the fold models, threshold at 0.5
    - All 6 metrics per case: Dice (%), IoU (%), Precision (%), Recall (%), HD95 (mm), ASSD (mm)
  2B:
    - Paired Wilcoxon signed-rank test vs standard_scse (per metric, for single-fold mean and ensemble)
    - Effect size r = Z / sqrt(N)
    - Bootstrap 95% CI of the mean difference (10,000 resamples)
    - Holm correction within each family
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
from torch.utils.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.attention_isolated_model import get_isolated_model
from data.mama_mia_5fold import PATCH_SIZE, TARGET_SPACING


class TestDataset(Dataset):
    def __init__(self, test_dirs):
        self.test_dirs = test_dirs

    def __len__(self):
        return len(self.test_dirs)

    def __getitem__(self, idx):
        pdir = self.test_dirs[idx]
        img = np.load(pdir / "image.npy", mmap_mode="r")
        lbl = np.load(pdir / "label.npy", mmap_mode="r")
        pid = pdir.name.replace("_test", "")
        img_t = torch.from_numpy(img.copy()).to(torch.float32)
        lbl_t = torch.from_numpy(lbl.copy()).to(torch.float32)
        return {"image": img_t, "label": lbl_t, "pid": pid}


def compute_all_metrics(p_bin, g_bin, spacing=TARGET_SPACING):
    p = torch.from_numpy(p_bin).bool()
    g = torch.from_numpy(g_bin).bool()
    p_sum = p.sum().item()
    g_sum = g.sum().item()
    inter = (p & g).sum().item()

    if p_sum == 0 and g_sum == 0:
        return 100.0, 100.0, 100.0, 100.0, 0.0, 0.0
    if p_sum == 0 or g_sum == 0:
        diag = float(np.sqrt(sum((s * d) ** 2 for s, d in zip(spacing, p.shape))))
        dsc = 0.0
        iou = 0.0
        prec = 100.0 if g_sum == 0 else 0.0
        rec = 100.0 if p_sum == 0 else 0.0
        return dsc, iou, prec, rec, diag, diag

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


class EnsemblePredictor:
    def __init__(self, models):
        self.models = models

    def __call__(self, x):
        probs = [torch.sigmoid(m(x)) for m in self.models]
        return torch.stack(probs, dim=0).mean(dim=0)


def load_model_fold(mode, fold, ckpt_path, device):
    m = get_isolated_model(mode=mode, in_channels=5, out_channels=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def bootstrap_ci_diff(x, y, n_boot=10000, ci=95.0, seed=42):
    diff = np.array(x) - np.array(y)
    rng = np.random.default_rng(seed)
    boot_means = [np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(n_boot)]
    alpha = (100.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha))
    upper = float(np.percentile(boot_means, 100.0 - alpha))
    return float(np.mean(diff)), lower, upper


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
    parser = argparse.ArgumentParser(description="Fold Ensembles and Paired Statistical Analysis")
    parser.add_argument("--mmap-dir", type=str, default="./data/preprocessed_mmap")
    parser.add_argument("--results-dir", type=str, default="./results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    mmap_dir = Path(args.mmap_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================================================")
    print("  SECTION 2A & 2B: FOLD ENSEMBLES & PAIRED STATISTICS (306 TEST CASES)")
    print("==========================================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    test_dirs = sorted([d for d in mmap_dir.iterdir() if d.is_dir() and d.name.endswith("_test")])
    test_ds = TestDataset(test_dirs)
    print(f"[data] Found {len(test_ds)} test volumes.")


if __name__ == "__main__":
    main()
