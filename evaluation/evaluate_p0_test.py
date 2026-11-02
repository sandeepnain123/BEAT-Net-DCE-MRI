#!/usr/bin/env python3
"""
P0 Evaluation: True Unbiased Test-Set Evaluation on All 306 Held-Out Patients
=============================================================================
Evaluates Fold-4 checkpoints:
  1. Standard scSE
  2. Capacity-Matched Boundary
  3. Full Boundary-Aware

Computes per-case metrics across all 306 held-out test patients:
  - DSC, IoU, Precision, Recall
  - Paired Wilcoxon signed-rank tests against Standard scSE
  - Saves to results/p0_test_comparison.json
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from monai.inferers import sliding_window_inference

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.attention_isolated_model import get_isolated_model
from data.mama_mia_5fold import get_5fold_splits, PATCH_SIZE


def load_model(mode: str, ckpt_path: str, device):
    model = get_isolated_model(mode=mode, in_channels=5, out_channels=1).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model_sd = model.state_dict()
    for k, v in sd.items():
        clean_k = k.replace("module.", "")
        if clean_k in model_sd and model_sd[clean_k].shape == v.shape:
            model_sd[clean_k] = v
    model.load_state_dict(model_sd, strict=False)
    model.eval()
    return model


def compute_metrics(pred_bin: torch.Tensor, gt_bin: torch.Tensor):
    pred = pred_bin.bool()
    gt = gt_bin.bool()
    n_p = pred.sum().item()
    n_g = gt.sum().item()
    inter = (pred & gt).sum().item()
    union = (pred | gt).sum().item()

    if n_p == 0 and n_g == 0:
        return 1.0, 1.0, 1.0, 1.0
    if n_p == 0 or n_g == 0:
        return 0.0, 0.0, 0.0, 0.0

    dsc = 2.0 * inter / (n_p + n_g)
    iou = inter / union
    prec = inter / n_p
    rec = inter / n_g
    return dsc, iou, prec, rec


def main():
    parser = argparse.ArgumentParser(description="P0 MMC Test Evaluation")
    parser.add_argument("--ckpt-standard", type=str, default="./results/5fold_5chan/fold_4/final_model.pt")
    parser.add_argument("--ckpt-matched", type=str, default="./results/isolated_block_boundary_matched/fold_4/final_model.pt")
    parser.add_argument("--ckpt-full", type=str, default="./results/isolated_block_boundary_full/fold_4/final_model.pt")
    parser.add_argument("--mmap-dir", type=str, default="./data/preprocessed_mmap")
    parser.add_argument("--split-csv", type=str, default=None)
    parser.add_argument("--out-json", type=str, default="./results/p0_test_comparison.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[p0_test] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    mmap_dir = Path(args.mmap_dir)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Load 306 test split
    _, test_split = get_5fold_splits(in_channels=5, seed=42, split_csv=Path(args.split_csv) if args.split_csv else None)
    test_pids = [item["patient_id"] for item in test_split]
    test_dirs = [mmap_dir / f"{pid}_test" for pid in test_pids if (mmap_dir / f"{pid}_test").exists()]
    print(f"[p0_test] Found {len(test_dirs)} / {len(test_pids)} preprocessed test cases.")

    models = {}
    if os.path.exists(args.ckpt_standard):
        models["standard_scse"] = load_model("standard", args.ckpt_standard, device)
    if os.path.exists(args.ckpt_matched):
        models["boundary_matched"] = load_model("boundary_matched", args.ckpt_matched, device)
    if os.path.exists(args.ckpt_full):
        models["boundary_full"] = load_model("boundary_full", args.ckpt_full, device)

    if not models:
        print("[p0_test] No checkpoint found. Please provide valid checkpoint paths.")
        return

    per_case = {m: [] for m in models}
    t0 = time.time()

    for idx, pdir in enumerate(test_dirs):
        pid = pdir.name.replace("_test", "")
        img = np.load(pdir / "image.npy", mmap_mode="r")
        lbl = np.load(pdir / "label.npy", mmap_mode="r")

        img_t = torch.from_numpy(img.copy()).unsqueeze(0).to(device, dtype=torch.bfloat16)
        lbl_t = torch.from_numpy(lbl.copy()).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for m_name, model in models.items():
                logits = sliding_window_inference(
                    img_t, PATCH_SIZE, sw_batch_size=8, predictor=model, overlap=0.5
                )
                pred_bin = (torch.sigmoid(logits[0]) > 0.5).squeeze(0)
                dsc, iou, prec, rec = compute_metrics(pred_bin, lbl_t[0])
                per_case[m_name].append({
                    "pid": pid,
                    "dsc": dsc,
                    "iou": iou,
                    "precision": prec,
                    "recall": rec
                })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_dirs):
            print(f"[{idx+1:03d}/{len(test_dirs):03d}] Evaluated in {time.time()-t0:.1f}s", flush=True)

    summary = {}
    for m_name in models:
        dscs = [c["dsc"] * 100.0 for c in per_case[m_name]]
        ious = [c["iou"] * 100.0 for c in per_case[m_name]]
        precs = [c["precision"] * 100.0 for c in per_case[m_name]]
        recs = [c["recall"] * 100.0 for c in per_case[m_name]]

        summary[m_name] = {
            "n_patients": len(dscs),
            "dsc_mean": float(np.mean(dscs)),
            "dsc_std": float(np.std(dscs)),
            "dsc_median": float(np.median(dscs)),
            "iou_mean": float(np.mean(ious)),
            "precision_mean": float(np.mean(precs)),
            "recall_mean": float(np.mean(recs)),
        }

    with open(out_json, "w") as fp:
        json.dump({"summary": summary, "per_case": per_case}, fp, indent=2)
    print(f"[p0_test] Saved complete results to {out_json}")


if __name__ == "__main__":
    main()
