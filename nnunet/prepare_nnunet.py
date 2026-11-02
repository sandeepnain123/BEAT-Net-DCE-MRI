#!/usr/bin/env python3
"""
Convert MAMA-MIA to nnU-Net v2 format and write a custom split
matching the 960/240/306 train/val/test split.

Usage:
  python3 prepare_nnunet.py --dataset-root /path/to/MAMA-MIA --nnunet-raw /path/to/nnUNet_raw
"""
import argparse
import csv
import gzip
import json
import os
import shutil
from pathlib import Path

DATASET_ID = "001"
DATASET_NAME = f"Dataset{DATASET_ID}_MAMAMIA"


def gz(src: Path, dst: Path):
    """Copy a .nii file as .nii.gz"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def main():
    p = argparse.ArgumentParser(description="Prepare nnU-Net v2 Dataset from MAMA-MIA")
    p.add_argument("--dataset-root", default=os.environ.get("MAMAMIA_ROOT", "data/MAMA-MIA"),
                   help="Root directory of MAMA-MIA dataset containing images/ and segmentations/")
    p.add_argument("--split-csv", default=None,
                   help="Path to train_test_splits.csv (default: data/metadata/train_test_splits.csv or inside dataset-root)")
    p.add_argument("--nnunet-raw", default=os.environ.get("nnUNet_raw", "nnunet_data/nnUNet_raw"),
                   help="Target nnUNet_raw directory")
    p.add_argument("--nnunet-preprocessed", default=os.environ.get("nnUNet_preprocessed", None),
                   help="Target nnUNet_preprocessed directory (default: parent(nnunet-raw)/nnUNet_preprocessed)")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dataset_root = Path(args.dataset_root)
    images_dir = dataset_root / "images"
    seg_expert = dataset_root / "segmentations" / "expert"

    if args.split_csv:
        split_csv = Path(args.split_csv)
    elif (Path("data/metadata/train_test_splits.csv")).exists():
        split_csv = Path("data/metadata/train_test_splits.csv")
    else:
        split_csv = dataset_root / "train_test_splits.csv"

    raw_dir = Path(args.nnunet_raw) / DATASET_NAME
    imgs_tr = raw_dir / "imagesTr"
    lbls_tr = raw_dir / "labelsTr"
    imgs_ts = raw_dir / "imagesTs"
    for d in [imgs_tr, lbls_tr, imgs_ts]:
        d.mkdir(parents=True, exist_ok=True)

    # Load split
    with open(split_csv) as f:
        rows = list(csv.DictReader(f))
    train_ids = [r["train_split"] for r in rows if r["train_split"]]
    test_ids = [r["test_split"] for r in rows if r["test_split"]]

    import random
    random.seed(args.seed)
    random.shuffle(train_ids)
    n_val = int(len(train_ids) * args.val_fraction)
    val_ids = train_ids[:n_val]
    train_ids = train_ids[n_val:]
    print(f"Split: {len(train_ids)} train  {len(val_ids)} val  {len(test_ids)} test")

    # Map patient_id -> nnunet case id (zero-padded index)
    all_train_val = train_ids + val_ids
    case_map = {pid: f"MAMAMIA_{i+1:04d}" for i, pid in enumerate(all_train_val)}

    # Copy training + val images
    total = len(all_train_val)
    for i, pid in enumerate(all_train_val):
        case = case_map[pid]
        img_dir = images_dir / pid
        for ch in range(3):
            src = img_dir / f"{pid}_{ch:04d}.nii"
            if src.exists():
                gz(src, imgs_tr / f"{case}_{ch:04d}.nii.gz")
        seg_src = seg_expert / f"{pid}.nii"
        if seg_src.exists():
            gz(seg_src, lbls_tr / f"{case}.nii.gz")
        if (i + 1) % 100 == 0:
            print(f"  train/val: {i+1}/{total}")

    # Copy test images (no labels needed for nnU-Net inference)
    test_map = {pid: f"MAMAMIA_{len(all_train_val)+i+1:04d}" for i, pid in enumerate(test_ids)}
    for i, pid in enumerate(test_ids):
        case = test_map[pid]
        img_dir = images_dir / pid
        for ch in range(3):
            src = img_dir / f"{pid}_{ch:04d}.nii"
            if src.exists():
                gz(src, imgs_ts / f"{case}_{ch:04d}.nii.gz")
        if (i + 1) % 50 == 0:
            print(f"  test: {i+1}/{len(test_ids)}")

    # dataset.json
    dataset_json = {
        "channel_names": {
            "0": "pre_contrast",
            "1": "early_post_contrast",
            "2": "late_post_contrast"
        },
        "labels": {"background": 0, "tumour": 1},
        "numTraining": len(all_train_val),
        "file_ending": ".nii.gz",
        "name": "MAMA-MIA",
        "description": "3-channel DCE-MRI breast tumour segmentation",
        "reference": "Garrucho et al. 2024",
        "licence": "CC BY 4.0",
        "release": "1.0",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    (raw_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print("Wrote dataset.json")

    # Custom splits_final.json - matches our 960/240/306 split exactly
    train_cases = [case_map[pid] for pid in train_ids]
    val_cases = [case_map[pid] for pid in val_ids]
    splits = [{"train": train_cases, "val": val_cases}]

    preprocessed_base = Path(args.nnunet_preprocessed) if args.nnunet_preprocessed else Path(args.nnunet_raw).parent / "nnUNet_preprocessed"
    splits_dir = preprocessed_base / DATASET_NAME
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "splits_final.json").write_text(json.dumps(splits, indent=2))
    print(f"Wrote splits_final.json ({len(train_cases)} train, {len(val_cases)} val)")

    nnunet_base = Path(args.nnunet_raw).parent
    print(f"""
Done. Set these environment variables before running nnU-Net:
  export nnUNet_raw={args.nnunet_raw}
  export nnUNet_preprocessed={preprocessed_base}
  export nnUNet_results={nnunet_base}/nnUNet_results

Then run:
  nnUNetv2_plan_and_preprocess -d {DATASET_ID} -c 3d_fullres --verify_dataset_integrity
""")

    maps = {"train": {v: k for k, v in case_map.items()},
            "test": {v: k for k, v in test_map.items()}}
    (raw_dir / "case_map.json").write_text(json.dumps(maps, indent=2))
    print("Saved case_map.json")


if __name__ == "__main__":
    main()
