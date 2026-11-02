#!/usr/bin/env python3
"""
Full nnU-Net v2 Evaluation on Held-Out Test Set (N=306 Patients)
================================================================
1. Runs nnUNetv2_predict on imagesTs using the trained Fold 0 checkpoint.
2. Compares predictions to ground-truth expert segmentations.
3. Computes DSC, IoU, Precision, Recall, HD95, and ASSD.
4. Breaks down metrics overall and by cohort (DUKE, ISPY1, ISPY2, NACT).
5. Saves results to JSON.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance


def run_prediction(images_ts: Path, pred_dir: Path, nnunet_base: Path, predict_bin: str = "nnUNetv2_predict"):
    pred_dir.mkdir(parents=True, exist_ok=True)
    existing_preds = list(pred_dir.glob("*.nii.gz"))
    if len(existing_preds) >= 306:
        print(f"[predict] Found {len(existing_preds)} existing predictions in {pred_dir}, skipping inference.")
        return

    print(f"[predict] Commencing nnUNetv2_predict on {images_ts}...")
    t0 = time.time()
    env = os.environ.copy()
    env["nnUNet_raw"] = str(nnunet_base / "nnUNet_raw")
    env["nnUNet_preprocessed"] = str(nnunet_base / "nnUNet_preprocessed")
    env["nnUNet_results"] = str(nnunet_base / "nnUNet_results")

    cmd = [
        predict_bin,
        "-i", str(images_ts),
        "-o", str(pred_dir),
        "-d", "001",
        "-c", "3d_fullres",
        "-f", "0",
        "--disable_tta"
    ]
    subprocess.run(cmd, env=env, check=True)
    print(f"[predict] Inference complete in {time.time() - t0:.1f}s.")


def compute_metrics(pred_bin, lbl_bin, spacing):
    p = pred_bin.astype(bool)
    g = lbl_bin.astype(bool)
    inter = np.logical_and(p, g).sum()
    p_sum = p.sum()
    g_sum = g.sum()
    union = np.logical_or(p, g).sum()

    if p_sum == 0 and g_sum == 0:
        return 1.0, 1.0, 1.0, 1.0, 0.0, 0.0
    if p_sum == 0 or g_sum == 0:
        diag = float(np.sqrt(sum((s * d) ** 2 for s, d in zip(spacing, pred_bin.shape))))
        return 0.0, 0.0, (1.0 if g_sum == 0 else 0.0), (1.0 if p_sum == 0 else 0.0), diag, diag

    dsc = (2.0 * inter) / (p_sum + g_sum)
    iou = inter / union if union > 0 else 0.0
    prec = inter / p_sum if p_sum > 0 else 0.0
    rec = inter / g_sum if g_sum > 0 else 0.0

    idx = np.where(p | g)
    pad = 4
    hmin, hmax = max(0, idx[0].min() - pad), min(pred_bin.shape[0], idx[0].max() + pad + 1)
    wmin, wmax = max(0, idx[1].min() - pad), min(pred_bin.shape[1], idx[1].max() + pad + 1)
    dmin, dmax = max(0, idx[2].min() - pad), min(pred_bin.shape[2], idx[2].max() + pad + 1)

    p_crop = torch.from_numpy(p[hmin:hmax, wmin:wmax, dmin:dmax]).unsqueeze(0).unsqueeze(0).float()
    g_crop = torch.from_numpy(g[hmin:hmax, wmin:wmax, dmin:dmax]).unsqueeze(0).unsqueeze(0).float()

    try:
        hd95 = compute_hausdorff_distance(p_crop, g_crop, include_background=True, percentile=95.0, spacing=spacing).item()
    except Exception:
        hd95 = float("nan")
    try:
        assd = compute_average_surface_distance(p_crop, g_crop, include_background=True, symmetric=True, spacing=spacing).item()
    except Exception:
        assd = float("nan")

    return dsc, iou, prec, rec, hd95, assd


def main():
    parser = argparse.ArgumentParser(description="nnU-Net v2 Test Set Evaluation")
    parser.add_argument("--nnunet-base", type=str, default=os.environ.get("NNUNET_BASE", "./data/nnunet_data"))
    parser.add_argument("--seg-expert", type=str, default="./data/MAMA-MIA/segmentations/expert")
    parser.add_argument("--pred-dir", type=str, default=None)
    parser.add_argument("--out-json", type=str, default="./results/nnunet_test.json")
    parser.add_argument("--skip-prediction", action="store_true")
    args = parser.parse_args()

    nnunet_base = Path(args.nnunet_base)
    raw_dir = nnunet_base / "nnUNet_raw" / "Dataset001_MAMAMIA"
    images_ts = raw_dir / "imagesTs"
    case_map_file = raw_dir / "case_map.json"
    seg_expert = Path(args.seg_expert)
    pred_dir = Path(args.pred_dir) if args.pred_dir else (nnunet_base / "test_predictions")
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_prediction:
        predict_bin = shutil.which("nnUNetv2_predict") or "nnUNetv2_predict"
        run_prediction(images_ts, pred_dir, nnunet_base, predict_bin=predict_bin)

    if not case_map_file.exists():
        print(f"[error] case_map.json not found in {raw_dir}")
        return

    with open(case_map_file) as f:
        case_map = json.load(f)
    nnunet_to_patient = {v: k for k, v in case_map.items()}

    pred_files = sorted(list(pred_dir.glob("*.nii.gz")))
    print(f"[eval] Evaluating {len(pred_files)} predictions in {pred_dir}...")

    per_patient = []
    t0 = time.time()
    for idx, ppath in enumerate(pred_files):
        cid = ppath.name.replace(".nii.gz", "")
        pid = nnunet_to_patient.get(cid, cid)
        cohort = pid.split("_")[0]

        gt_path = seg_expert / f"{pid}.nii"
        if not gt_path.exists():
            gt_path = seg_expert / f"{pid}.nii.gz"

        pred_nii = nib.load(str(ppath))
        pred_arr = (pred_nii.get_fdata() > 0.5).astype(np.uint8)
        spacing = pred_nii.header.get_zooms()[:3]

        gt_nii = nib.load(str(gt_path))
        gt_arr = (gt_nii.get_fdata() > 0.5).astype(np.uint8)

        dsc, iou, prec, rec, hd95, assd = compute_metrics(pred_arr, gt_arr, spacing=spacing)

        per_patient.append({
            "pid": pid,
            "nnunet_case": cid,
            "cohort": cohort,
            "dsc": dsc,
            "iou": iou,
            "precision": prec,
            "recall": rec,
            "hd95": hd95,
            "assd": assd
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(pred_files):
            print(f"[{idx+1:03d}/{len(pred_files):03d}] Evaluated in {time.time()-t0:.1f}s", flush=True)

    dscs = [c["dsc"] * 100.0 for c in per_patient]
    ious = [c["iou"] * 100.0 for c in per_patient]
    precs = [c["precision"] * 100.0 for c in per_patient]
    recs = [c["recall"] * 100.0 for c in per_patient]
    hd95s = [c["hd95"] for c in per_patient if not np.isnan(c["hd95"])]
    assds = [c["assd"] for c in per_patient if not np.isnan(c["assd"])]

    by_cohort = {}
    for cohort in ["DUKE", "ISPY2", "ISPY1", "NACT"]:
        c_cases = [c for c in per_patient if c["cohort"] == cohort]
        if c_cases:
            c_dsc = [c["dsc"] * 100.0 for c in c_cases]
            by_cohort[cohort] = {
                "n": len(c_cases),
                "dsc_mean": float(np.mean(c_dsc)),
                "dsc_std": float(np.std(c_dsc)),
                "dsc_median": float(np.median(c_dsc))
            }

    summary = {
        "model": "nnUNet_v2_fullres",
        "fold": 0,
        "n_patients": len(per_patient),
        "overall": {
            "dsc_mean": float(np.mean(dscs)),
            "dsc_std": float(np.std(dscs)),
            "dsc_median": float(np.median(dscs)),
            "iou_mean": float(np.mean(ious)),
            "precision_mean": float(np.mean(precs)),
            "recall_mean": float(np.mean(recs)),
            "hd95_mean": float(np.mean(hd95s)),
            "assd_mean": float(np.mean(assds))
        },
        "by_cohort": by_cohort,
        "per_patient": per_patient
    }

    with open(out_json, "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"[done] Wrote results to {out_json}")


if __name__ == "__main__":
    main()
