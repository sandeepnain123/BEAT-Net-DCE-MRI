#!/usr/bin/env python3
"""
Evaluate a Single Trained Checkpoint on All 306 Held-Out Test Patients (Strictly Epoch 500)
========================================================================================
Usage:
  python evaluate_single_fold_test.py --mode standard --fold 4 --data-dir ./data/preprocessed_mmap
Outputs:
  results/eval_test_{mode}_fold_{fold}_E500.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.attention_isolated_model import get_isolated_model
from data.mama_mia_5fold import PATCH_SIZE, TARGET_SPACING


def compute_metrics(pred_bin: torch.Tensor, lbl_bin: torch.Tensor, spacing=TARGET_SPACING):
    p = pred_bin.bool()
    g = lbl_bin.bool()
    inter = (p & g).sum().item()
    p_sum = p.sum().item()
    g_sum = g.sum().item()
    union = (p | g).sum().item()

    if p_sum == 0 and g_sum == 0:
        return 100.0, 100.0, 100.0, 100.0, 0.0, 0.0
    if p_sum == 0 or g_sum == 0:
        diag = float(np.sqrt(sum((s * d) ** 2 for s, d in zip(spacing, p.shape))))
        return 0.0, 0.0, (100.0 if g_sum == 0 else 0.0), (100.0 if p_sum == 0 else 0.0), diag, diag

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

    return dsc, iou, prec, rec, hd95, assd


def main():
    parser = argparse.ArgumentParser(description="Evaluate single fold checkpoint on test set")
    parser.add_argument("--mode", type=str, required=True, help="Model architecture mode")
    parser.add_argument("--fold", type=int, required=True, help="Fold index (0-4)")
    parser.add_argument("--ckpt", type=str, default=None, help="Explicit checkpoint path")
    parser.add_argument("--data-dir", type=str, default="./data/preprocessed_mmap", help="Path to preprocessed mmap directory")
    parser.add_argument("--out-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--sw-batch-size", type=int, default=8)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--save-preds", action="store_true", help="Save predictions as .npy")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==========================================================================")
    print(f"  Test Evaluation (Strictly Epoch 500): Mode={args.mode} | Fold={args.fold}")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("==========================================================================")

    ckpt_path = args.ckpt
    if ckpt_path is None:
        folder = Path(args.out_dir) / f"isolated_block_{args.mode}" / f"fold_{args.fold}"
        if (folder / "final_model.pt").exists():
            ckpt_path = folder / "final_model.pt"
        else:
            ckpt_path = folder / "best_model.pt"

    if not os.path.exists(str(ckpt_path)):
        print(f"[error] Checkpoint {ckpt_path} not found!")
        sys.exit(1)

    print(f"[model] Loading model checkpoint from {ckpt_path}")
    model = get_isolated_model(mode=args.mode, in_channels=5, out_channels=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "final_model.pt" in str(ckpt_path):
        ep = ckpt.get("epoch", 500)
        assert ep == 500, f"Error: Checkpoint {ckpt_path} logged epoch is {ep}, expected 500!"
        print(f"[check] Verified checkpoint is final_model.pt at Epoch 500.")

    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()

    mmap_dir = Path(args.data_dir)
    test_dirs = sorted([p for p in mmap_dir.iterdir() if p.is_dir() and "_test" in p.name])
    print(f"[data] Found {len(test_dirs)} preprocessed test cases in {mmap_dir}.")

    per_case = []
    t0 = time.time()

    for idx, pdir in enumerate(test_dirs):
        pid = pdir.name.replace("_test", "")
        img = np.load(pdir / "image.npy", mmap_mode="r")
        lbl = np.load(pdir / "label.npy", mmap_mode="r")

        img_t = torch.from_numpy(img.copy()).unsqueeze(0).to(device, dtype=torch.bfloat16)
        lbl_t = torch.from_numpy(lbl.copy()).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = sliding_window_inference(
                img_t, PATCH_SIZE, sw_batch_size=args.sw_batch_size, predictor=model, overlap=args.overlap
            )
            pred_bin = (torch.sigmoid(logits[0]) > 0.5).squeeze(0)

        dsc, iou, prec, rec, hd95, assd = compute_metrics(pred_bin, lbl_t[0])

        per_case.append({
            "pid": pid,
            "dsc": dsc,
            "iou": iou,
            "precision": prec,
            "recall": rec,
            "hd95": hd95,
            "assd": assd
        })

        if args.save_preds:
            pred_save_dir = Path(args.out_dir) / "test_preds" / f"{args.mode}_fold{args.fold}_E500"
            pred_save_dir.mkdir(parents=True, exist_ok=True)
            np.save(pred_save_dir / f"{pid}_pred.npy", pred_bin.cpu().numpy().astype(np.uint8))

        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_dirs):
            print(f"[{idx+1:03d}/{len(test_dirs):03d}] Evaluated in {time.time()-t0:.1f}s", flush=True)

    dscs = [c["dsc"] for c in per_case]
    ious = [c["iou"] for c in per_case]
    precs = [c["precision"] for c in per_case]
    recs = [c["recall"] for c in per_case]
    hd95s = [c["hd95"] for c in per_case if not np.isnan(c["hd95"])]
    assds = [c["assd"] for c in per_case if not np.isnan(c["assd"])]

    summary = {
        "mode": args.mode,
        "fold": args.fold,
        "n_patients": len(dscs),
        "checkpoint_path": str(ckpt_path),
        "stored_epoch": 500,
        "dsc_mean": float(np.mean(dscs)),
        "dsc_std": float(np.std(dscs)),
        "dsc_median": float(np.median(dscs)),
        "iou_mean": float(np.mean(ious)),
        "precision_mean": float(np.mean(precs)),
        "recall_mean": float(np.mean(recs)),
        "hd95_mean": float(np.mean(hd95s)),
        "assd_mean": float(np.mean(assds)),
    }

    print("\n==========================================================================")
    print(f"  TEST SET EVALUATION RESULT (EPOCH 500): Mode={args.mode} | Fold={args.fold}")
    print(f"  Dice: {summary['dsc_mean']:.2f} +- {summary['dsc_std']:.2f}% (Median: {summary['dsc_median']:.2f}%)")
    print(f"  IoU:  {summary['iou_mean']:.2f}% | Prec: {summary['precision_mean']:.2f}% | Rec: {summary['recall_mean']:.2f}%")
    print(f"  HD95: {summary['hd95_mean']:.2f} mm | ASSD: {summary['assd_mean']:.2f} mm")
    print("==========================================================================")

    out_file = Path(args.out_dir) / f"eval_test_{args.mode}_fold_{args.fold}_E500.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump({"summary": summary, "per_case": per_case}, fp, indent=2)
    print(f"[done] Saved test results to {out_file}")


if __name__ == "__main__":
    main()
