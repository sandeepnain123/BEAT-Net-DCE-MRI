"""
MAMA-MIA 5-Fold Cross Validation Data Pipeline
================================================
Deterministic 5-Fold splits for 1,200 training patients.
Supports:
  - 2-channel input: [image_0, image_2] (Pre-contrast, Post-contrast 2)
  - 5-channel input: [image_0, image_1, image_2, sub1, sub2] (Pre, Post1, Post2, Subtraction 1, Subtraction 2)
"""

import os
import csv
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
import torch
from sklearn.model_selection import KFold

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    NormalizeIntensityd, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, RandGaussianNoised,
    SpatialPadd, ConcatItemsd, DeleteItemsd, EnsureTyped
)

# Configurable paths with environment variable and local metadata fallback
DEFAULT_DATASET_ROOT = Path(os.environ.get("MAMAMIA_ROOT", "./data/MAMA-MIA"))
_METADATA_SPLIT = Path(__file__).resolve().parent / "metadata" / "train_test_splits.csv"
DEFAULT_SPLIT_CSV = _METADATA_SPLIT if _METADATA_SPLIT.exists() else (DEFAULT_DATASET_ROOT / "train_test_splits.csv")

TARGET_SPACING = (0.8, 0.8, 1.0)
PATCH_SIZE = (128, 128, 64)


def load_train_test_patients(split_csv: Optional[Path] = None) -> Tuple[List[str], List[str]]:
    csv_path = Path(split_csv) if split_csv else DEFAULT_SPLIT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found at: {csv_path}")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    train_ids = [r["train_split"] for r in rows if r.get("train_split")]
    test_ids = [r["test_split"] for r in rows if r.get("test_split")]
    return train_ids, test_ids


def build_patient_dict(patient_id: str, dataset_root: Optional[Path] = None, in_channels: int = 2) -> Dict[str, str]:
    root = Path(dataset_root) if dataset_root else DEFAULT_DATASET_ROOT
    img_dir = root / "images" / patient_id
    seg_expert = root / "segmentations" / "expert"
    return {
        "patient_id": patient_id,
        "image_0": str(img_dir / f"{patient_id}_0000.nii"),
        "image_1": str(img_dir / f"{patient_id}_0001.nii"),
        "image_2": str(img_dir / f"{patient_id}_0002.nii"),
        "label": str(seg_expert / f"{patient_id}.nii"),
    }


def get_5fold_splits(
    in_channels: int = 2,
    seed: int = 42,
    dataset_root: Optional[Path] = None,
    split_csv: Optional[Path] = None
) -> Tuple[List[Tuple[List[dict], List[dict]]], List[dict]]:
    train_ids, test_ids = load_train_test_patients(split_csv=split_csv)

    valid_train = []
    for pid in train_ids:
        d = build_patient_dict(pid, dataset_root=dataset_root, in_channels=in_channels)
        if all(os.path.exists(d[k]) for k in ["image_0", "image_1", "image_2", "label"]):
            valid_train.append(d)

    valid_test = []
    for pid in test_ids:
        d = build_patient_dict(pid, dataset_root=dataset_root, in_channels=in_channels)
        if all(os.path.exists(d[k]) for k in ["image_0", "image_1", "image_2", "label"]):
            valid_test.append(d)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in kf.split(valid_train):
        train_fold = [valid_train[i] for i in train_idx]
        val_fold = [valid_train[i] for i in val_idx]
        folds.append((train_fold, val_fold))

    return folds, valid_test


def get_transforms(in_channels: int = 2, is_train: bool = True):
    keys = ["image_0", "image_1", "image_2"]

    transforms = [
        LoadImaged(keys=keys + ["label"]),
        EnsureChannelFirstd(keys=keys + ["label"]),
        Orientationd(keys=keys + ["label"], axcodes="RAS"),
        Spacingd(keys=keys + ["label"], pixdim=TARGET_SPACING, mode=tuple(["bilinear"] * 3 + ["nearest"])),
        NormalizeIntensityd(keys=keys, nonzero=True),
    ]

    if in_channels == 2:
        transforms.extend([
            ConcatItemsd(keys=["image_0", "image_2"], name="image"),
            DeleteItemsd(keys=keys),
        ])
    elif in_channels == 5:
        transforms.extend([
            ConcatItemsd(keys=["image_0", "image_1", "image_2"], name="image_raw"),
            DeleteItemsd(keys=keys),
        ])

    img_key = "image" if in_channels == 2 else "image_raw"
    if is_train:
        transforms.append(CropForegroundd(keys=[img_key, "label"], source_key=img_key))

    transforms.append(SpatialPadd(keys=[img_key, "label"], spatial_size=PATCH_SIZE))

    if is_train:
        transforms.extend([
            RandCropByPosNegLabeld(
                keys=[img_key, "label"],
                label_key="label",
                spatial_size=PATCH_SIZE,
                pos=1.0, neg=1.0, num_samples=2,
            ),
            RandFlipd(keys=[img_key, "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=[img_key, "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=[img_key, "label"], prob=0.5, spatial_axis=2),
            RandGaussianNoised(keys=[img_key], prob=0.2, mean=0.0, std=0.1),
        ])

    transforms.append(EnsureTyped(keys=[img_key, "label"]))
    return Compose(transforms)


def process_5chan_batch(batch: dict) -> dict:
    """Adds kinetic subtraction channels dynamically: [I0, I1, I2, I1-I0, I2-I0]."""
    if "image_raw" in batch:
        img = batch["image_raw"]  # shape: [B, 3, D, H, W]
        sub1 = img[:, 1:2] - img[:, 0:1]  # Post1 - Pre
        sub2 = img[:, 2:3] - img[:, 0:1]  # Post2 - Pre
        batch["image"] = torch.cat([img, sub1, sub2], dim=1)  # [B, 5, D, H, W]
    return batch
