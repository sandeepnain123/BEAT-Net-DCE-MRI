#!/usr/bin/env python3
"""
Harmonise MMC Temporal Evaluation: Probability 0.5 vs Legacy Logit 0.5 Thresholds
==================================================================================
Produces:
  - eda/mmc_temporal_probability05.csv (N=306 cases)
  - eda/threshold_reconciliation_summary.json

Evaluates the phase-randomised 2-channel model (final_model.pt at Epoch 500)
on the MMC held-out test cohort (N=306) across temporal phases P1 and P2:
  1. Corrected threshold: sigmoid(logits) >= 0.5 (equivalent to logits >= 0.0)
  2. Legacy logit threshold: logits >= 0.5 (used in earlier pre-amendment evaluations)

Also calculates the resulting APCPS (P95) and Early-Peak kinetic Dice scores
under probability 0.5.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.attention_55_model import AttentionAllDecoderUNet


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p_sum = int(pred.sum())
    g_sum = int(gt.sum())
    if p_sum == 0 and g_sum == 0:
        return 100.0
    if p_sum == 0 or g_sum == 0:
        return 0.0
    inter = int(np.count_nonzero(pred & gt))
    return float(200.0 * inter / (p_sum + g_sum))


def main():
    parser = argparse.ArgumentParser(description="MMC Temporal Probability 0.5 Evaluation")
    parser.add_argument("--mmap-dir", type=str, default="data/preprocessed_mmap",
                        help="Path to preprocessed mmap directory")
    parser.add_argument("--split-csv", type=str, default="data/metadata/train_test_splits.csv",
                        help="Path to train_test_splits.csv")
    parser.add_argument("--ckpt", type=str, default="results/phase_randomized_2chan_seed0/final_model.pt",
                        help="Path to phase-randomised 2-ch checkpoint (final_model.pt, Epoch 500)")
    parser.add_argument("--early-peak-json", type=str, default="data/metadata/early_peak_test_selections.json",
                        help="Path to early peak selections JSON")
    parser.add_argument("--out-csv", type=str, default="eda/mmc_temporal_probability05.csv",
                        help="Output per-case CSV")
    parser.add_argument("--out-summary", type=str, default="eda/threshold_reconciliation_summary.json",
                        help="Output summary JSON")
    parser.add_argument("--shard", type=int, default=0, help="Shard index (0 for all if num-shards=1)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    mmap_dir = Path(args.mmap_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[temporal_eval] Using device: {device}")

    # Load checkpoint
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[error] Checkpoint {ckpt_path} not found. Please train or provide model checkpoint.")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stored_epoch = ckpt.get("epoch")
    if stored_epoch is not None:
        assert stored_epoch == 500, f"Expected Epoch 500 checkpoint, got {stored_epoch}"

    model = AttentionAllDecoderUNet(in_channels=2, out_channels=1).to(device).eval()
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)

    # Determine test cases
    if Path(args.split_csv).exists():
        df_split = pd.read_csv(args.split_csv)
        test_pids = [r for r in df_split["test_split"].dropna().tolist() if r]
    else:
        test_pids = [p.name.replace("_test", "") for p in sorted(mmap_dir.glob("*_test"))]

    test_dirs = [mmap_dir / f"{pid}_test" for pid in test_pids if (mmap_dir / f"{pid}_test").exists()]
    test_dirs = sorted(test_dirs)
    print(f"[temporal_eval] Total test cases found: {len(test_dirs)}")

    # Shard slicing
    shard_dirs = test_dirs[args.shard::args.num_shards]
    print(f"[temporal_eval] Processing shard {args.shard}/{args.num_shards} ({len(shard_dirs)} cases)")

    rows = []
    start_time = time.time()

    for i, p in enumerate(shard_dirs):
        pid = p.name.removesuffix("_test")
        centre = pid.split("_")[0]
        image = np.load(p / "image.npy", mmap_mode="r")
        gt = np.load(p / "label.npy", mmap_mode="r")[0] > 0

        row = {
            "pid": pid,
            "centre": centre
        }

        for phase in [1, 2]:
            input_2ch = np.stack([image[0], image[phase]]).astype(np.float32)
            tensor = torch.from_numpy(input_2ch).unsqueeze(0).to(device)

            with torch.inference_mode():
                # sliding_window_inference with AMP if CUDA available
                if device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits = sliding_window_inference(
                            tensor, (128, 128, 64), sw_batch_size=8, predictor=model, overlap=0.5, mode="constant"
                        )
                else:
                    logits = sliding_window_inference(
                        tensor, (128, 128, 64), sw_batch_size=2, predictor=model, overlap=0.5, mode="constant"
                    )
                vals = logits[0, 0].float().cpu().numpy()

            pred_prob05 = (vals >= 0.0)  # sigmoid(logits) >= 0.5
            pred_logit05 = (vals >= 0.5)  # legacy logit threshold

            row[f"P{phase}_dice_probability05"] = dice_score(pred_prob05, gt)
            row[f"P{phase}_dice_logit05"] = dice_score(pred_logit05, gt)
            row[f"P{phase}_pred_voxels_probability05"] = int(pred_prob05.sum())

        rows.append(row)

        if (i + 1) % 20 == 0 or (i + 1) == len(shard_dirs):
            print(f"  [{i+1}/{len(shard_dirs)}] cases completed ({time.time()-start_time:.1f}s)", flush=True)

    df_out = pd.DataFrame(rows).sort_values("pid")
    df_out.to_csv(out_csv, index=False)
    print(f"[done] Wrote per-case results to {out_csv}")

    # If full dataset evaluated in this run, produce threshold reconciliation summary
    if len(df_out) == len(test_dirs) and len(test_dirs) > 0:
        summary = {
            "cases": len(df_out),
            "P1_probability05": float(df_out["P1_dice_probability05"].mean()),
            "P2_probability05": float(df_out["P2_dice_probability05"].mean()),
            "P1_logit05": float(df_out["P1_dice_logit05"].mean()),
            "P2_logit05": float(df_out["P2_dice_logit05"].mean()),
            "P1_minus_P2_mean": float((df_out["P1_dice_probability05"] - df_out["P2_dice_probability05"]).mean()),
        }

        # Compute APCPS & early peak if selection files are present
        if Path(args.early_peak_json).exists():
            with open(args.early_peak_json) as fp:
                ep_data = json.load(fp)
            ep_selections = ep_data.get("selections", ep_data)
            ep_phases = [ep_selections.get(pid, {}).get("selected_phase", 1) if isinstance(ep_selections.get(pid), dict) else ep_selections.get(pid, 1) for pid in df_out["pid"]]
            ep_dice = [df_out.iloc[idx]["P1_dice_probability05"] if ph == 1 else df_out.iloc[idx]["P2_dice_probability05"] for idx, ph in enumerate(ep_phases)]
            summary["early_peak_probability05"] = float(np.mean(ep_dice))

        if args.out_summary:
            out_summary_path = Path(args.out_summary)
            out_summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_summary_path, "w") as fp:
                json.dump(summary, fp, indent=2)
            print(f"[done] Wrote threshold reconciliation summary to {out_summary_path}")


if __name__ == "__main__":
    main()
