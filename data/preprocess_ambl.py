#!/usr/bin/env python3
"""
Preprocess and Cache TCIA AMBL External Cohort (13 Evaluable Malignant Cases)
=============================================================================
Converts raw NIfTI dynamic volumes and expert tumor segmentations:
  - Canonical RAS orientation
  - Voxel resampling to 0.8 x 0.8 x 1.0 mm (trilinear for images, nearest for GT)
  - Caches p0.npy - p4.npy and gt.npy
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd
)

TARGET_SPACING = (0.8, 0.8, 1.0)
EVALUABLE_MALIGNANT_CASES = [
    "AMBL-003", "AMBL-005", "AMBL-007", "AMBL-008", "AMBL-011",
    "AMBL-016", "AMBL-018", "AMBL-022", "AMBL-028", "AMBL-031",
    "AMBL-032", "AMBL-038", "AMBL-047"
]


def preprocess_ambl(clean_dir: Path, out_dir: Path, cases=EVALUABLE_MALIGNANT_CASES):
    print("=" * 80)
    print("STEP: Preprocessing and Caching AMBL Volumes (RAS, 0.8x0.8x1.0 mm)")
    print(f"Source: {clean_dir} | Output: {out_dir}")
    print("=" * 80)

    out_dir.mkdir(parents=True, exist_ok=True)

    tf = Compose([
        LoadImaged(keys=["p0", "p1", "p2", "p3", "p4", "gt"]),
        EnsureChannelFirstd(keys=["p0", "p1", "p2", "p3", "p4", "gt"]),
        Orientationd(keys=["p0", "p1", "p2", "p3", "p4", "gt"], axcodes="RAS"),
        Spacingd(keys=["p0", "p1", "p2", "p3", "p4"], pixdim=TARGET_SPACING, mode="bilinear"),
        Spacingd(keys=["gt"], pixdim=TARGET_SPACING, mode="nearest"),
    ])

    for cid in cases:
        c_dir = out_dir / cid
        c_dir.mkdir(parents=True, exist_ok=True)
        gt_npy = c_dir / "gt.npy"
        p4_npy = c_dir / "p4.npy"
        if gt_npy.exists() and p4_npy.exists():
            print(f"[{cid}] Already preprocessed. Skipping.")
            continue

        print(f"[{cid}] Preprocessing from {clean_dir / cid}...")
        dd = {
            "p0": str(clean_dir / cid / "P0.nii"),
            "p1": str(clean_dir / cid / "P1.nii"),
            "p2": str(clean_dir / cid / "P2.nii"),
            "p3": str(clean_dir / cid / "P3.nii"),
            "p4": str(clean_dir / cid / "P4.nii"),
            "gt": str(clean_dir / cid / "GT.nii"),
        }
        res = tf(dd)
        gt_arr = (res["gt"][0].numpy() > 0.5).astype(np.uint8)
        for k in range(5):
            arr = res[f"p{k}"][0].numpy().astype(np.float32)
            if arr.shape != gt_arr.shape:
                fixed = np.zeros_like(gt_arr, dtype=np.float32)
                min_s = [min(a, b) for a, b in zip(arr.shape, gt_arr.shape)]
                fixed[:min_s[0], :min_s[1], :min_s[2]] = arr[:min_s[0], :min_s[1], :min_s[2]]
                arr = fixed
            np.save(c_dir / f"p{k}.npy", arr)
        np.save(gt_npy, gt_arr)
        print(f"[{cid}] Saved preprocessed volumes. Shape: {gt_arr.shape}, GT Voxels: {gt_arr.sum()}")
    print("All AMBL cases preprocessed successfully.\n")


def main():
    parser = argparse.ArgumentParser(description="Preprocess TCIA AMBL Dataset")
    parser.add_argument("--clean-dir", type=str, default=os.environ.get("AMBL_CLEAN", "./data/ambl_clean"))
    parser.add_argument("--out-dir", type=str, default="./data/ambl_preprocessed")
    args = parser.parse_args()

    preprocess_ambl(Path(args.clean_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
