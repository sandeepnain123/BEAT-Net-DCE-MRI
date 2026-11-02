#!/usr/bin/env python3
"""
Parallel High-Throughput Preprocessor for YCHC (Yunnan) External Cohort (N=99 Cases)
==================================================================================
Identical preprocessing to MMC:
  - Orientationd(axcodes="RAS")
  - Spacingd(pixdim=(0.8, 0.8, 1.0), mode=["bilinear"]*6 + ["nearest"]*2)
  - NormalizeIntensityd(nonzero=True) for images P0-P5
  - Saves phases_6c.npy (float16), gt.npy (uint8), breast_mask.npy (uint8)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd
)

TARGET_SPACING = (0.8, 0.8, 1.0)
keys_img = ["p0", "p1", "p2", "p3", "p4", "p5"]
keys_all = keys_img + ["gt", "mask"]

preproc_tf = Compose([
    LoadImaged(keys=keys_all),
    EnsureChannelFirstd(keys=keys_all),
    Orientationd(keys=keys_all, axcodes="RAS"),
    Spacingd(keys=keys_all, pixdim=TARGET_SPACING, mode=tuple(["bilinear"] * 6 + ["nearest"] * 2)),
    NormalizeIntensityd(keys=keys_img, nonzero=True)
])


def process_case_args(args_tuple):
    case_id, src_dir, out_dir = args_tuple
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)

    case_dir = src_dir / str(case_id)
    out_case_dir = out_dir / str(case_id)

    img_save = out_case_dir / "phases_6c.npy"
    gt_save = out_case_dir / "gt.npy"
    mask_save = out_case_dir / "breast_mask.npy"
    meta_save = out_case_dir / "meta.json"

    if img_save.exists() and gt_save.exists() and mask_save.exists() and meta_save.exists():
        return case_id, True, 0.0

    t0 = time.time()
    item = {
        "p0": str(case_dir / "P0.nii"),
        "p1": str(case_dir / "P1.nii"),
        "p2": str(case_dir / "P2.nii"),
        "p3": str(case_dir / "P3.nii"),
        "p4": str(case_dir / "P4.nii"),
        "p5": str(case_dir / "P5.nii"),
        "gt": str(case_dir / "GT.nii"),
        "mask": str(case_dir / "Breast_mask.nii")
    }

    try:
        res = preproc_tf(item)

        # Stack 6 phases: [6, H, W, D] in float16
        phases = [res[k].squeeze(0).numpy() for k in keys_img]
        phases_6c = np.stack(phases, axis=0).astype(np.float16)

        gt = (res["gt"].squeeze(0).numpy() > 0.5).astype(np.uint8)
        mask = (res["mask"].squeeze(0).numpy() > 0.5).astype(np.uint8)

        out_case_dir.mkdir(parents=True, exist_ok=True)
        np.save(img_save, phases_6c)
        np.save(gt_save, gt)
        np.save(mask_save, mask)

        hdr_p0 = nib.load(item["p0"]).header
        meta = {
            "case_id": case_id,
            "orig_shape": list(nib.load(item["p0"]).shape),
            "orig_spacing": [float(x) for x in hdr_p0.get_zooms()[:3]],
            "resampled_shape": list(phases_6c.shape[1:]),
            "resampled_spacing": list(TARGET_SPACING),
            "gt_voxels": int(gt.sum()),
            "mask_voxels": int(mask.sum()),
            "elapsed_s": time.time() - t0
        }
        with open(meta_save, "w") as fp:
            json.dump(meta, fp, indent=2)

        return case_id, False, time.time() - t0
    except Exception as e:
        print(f"[ERROR] Case {case_id}: {str(e)}")
        return case_id, False, -1.0


def main():
    parser = argparse.ArgumentParser(description="Preprocess YCHC Cohort (N=99 Cases)")
    parser.add_argument("--src-dir", type=str, default=os.environ.get("YCHC_CLEAN", "./data/ychc_clean"))
    parser.add_argument("--out-dir", type=str, default="./data/ychc_preprocessed")
    parser.add_argument("--workers", type=int, default=min(16, cpu_count()))
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [c for c in range(1, 100) if (src_dir / str(c) / "P0.nii").exists()]
    print(f"Found {len(cases)} cases in {src_dir}. Starting preprocessing with {args.workers} workers...")

    tasks = [(c, str(src_dir), str(out_dir)) for c in cases]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (cid, cached, el) in enumerate(pool.imap_unordered(process_case_args, tasks)):
            status = "CACHED" if cached else f"DONE ({el:.1f}s)"
            print(f"[{i+1:02d}/{len(cases):02d}] Case {cid:03d}: {status} (Total: {time.time()-t0:.1f}s)")

    print(f"Preprocessing completed in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
