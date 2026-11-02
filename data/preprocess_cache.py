"""
High-Throughput Parallel Preprocessor for MAMA-MIA 5-Fold Dynamic MRI
====================================================================
Runs offline or one-time on compute node:
  1. Loads raw 3-phase NIfTI (Pre, Post1, Post2) + expert segmentation
  2. Reorients to RAS, resamples to (0.8, 0.8, 1.0) mm spacing
  3. Nonzero intensity normalization
  4. Generates 5-channel dynamic volume: [Pre, Post1, Post2, Post1-Pre, Post2-Pre]
  5. Caches to SSD/NVMe/RAM (/dev/shm) and persists in bfloat16
  6. Pre-indexes foreground tumor voxel coordinates for O(1) GPU patch sampling
"""

import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, CropForegroundd, SpatialPadd, EnsureTyped
)

_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

from mama_mia_5fold import get_5fold_splits, TARGET_SPACING, PATCH_SIZE


def get_preproc_transforms(is_train: bool):
    keys = ["image_0", "image_1", "image_2"]
    tfs = [
        LoadImaged(keys=keys + ["label"]),
        EnsureChannelFirstd(keys=keys + ["label"]),
        Orientationd(keys=keys + ["label"], axcodes="RAS"),
        Spacingd(keys=keys + ["label"], pixdim=TARGET_SPACING, mode=tuple(["bilinear"] * 3 + ["nearest"])),
        NormalizeIntensityd(keys=keys, nonzero=True),
    ]
    if is_train:
        tfs.append(CropForegroundd(keys=keys + ["label"], source_key="image_0"))
    tfs.append(SpatialPadd(keys=keys + ["label"], spatial_size=PATCH_SIZE))
    tfs.append(EnsureTyped(keys=keys + ["label"], dtype=torch.float32))
    return Compose(tfs)


def process_one_patient(item, is_train: bool, persist_dir: Path, shm_dir: Path):
    pid = item["patient_id"]
    p_file = persist_dir / f"{pid}_{'train' if is_train else 'val'}.pt"
    s_file = shm_dir / f"{pid}_{'train' if is_train else 'val'}.pt"

    # If already in persist_dir, copy to shm_dir if available
    if p_file.exists():
        if shm_dir.exists() and not s_file.exists():
            data = torch.load(p_file, map_location="cpu")
            torch.save(data, s_file)
        return pid, True, 0.0

    t0 = time.time()
    tf = get_preproc_transforms(is_train=is_train)
    res = tf(item)

    # Build 5 channels: [Pre, Post1, Post2, Post1-Pre, Post2-Pre]
    img0 = res["image_0"]
    img1 = res["image_1"]
    img2 = res["image_2"]
    sub1 = img1 - img0
    sub2 = img2 - img0
    image_5c = torch.cat([img0, img1, img2, sub1, sub2], dim=0).to(torch.bfloat16)

    label = (res["label"] > 0.5).to(torch.uint8)

    # Pre-index foreground tumor voxel coordinates for O(1) positive patch sampling
    fg_indices = torch.nonzero(label[0] > 0).to(torch.int32)

    payload = {
        "patient_id": pid,
        "is_train": is_train,
        "image": image_5c,
        "label": label,
        "fg_indices": fg_indices,
        "shape": list(image_5c.shape),
    }

    persist_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p_file)

    if shm_dir.exists():
        torch.save(payload, s_file)

    elapsed = time.time() - t0
    return pid, False, elapsed


def main():
    parser = argparse.ArgumentParser(description="MAMA-MIA Parallel Preprocessor and Cache")
    parser.add_argument("--data-root", type=str, default=os.environ.get("MAMAMIA_ROOT", "./data/MAMA-MIA"))
    parser.add_argument("--split-csv", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="./data/preprocessed_5chan")
    parser.add_argument("--shm-dir", type=str, default="/dev/shm/mama_mia_5chan")
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    persist_dir = Path(args.out_dir)
    shm_dir = Path(args.shm_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    if os.path.exists("/dev/shm"):
        shm_dir.mkdir(parents=True, exist_ok=True)

    print(f"[preproc] Gathering Fold {args.fold} splits from {args.data_root}...")
    folds, test_files = get_5fold_splits(
        in_channels=5,
        seed=42,
        dataset_root=Path(args.data_root),
        split_csv=Path(args.split_csv) if args.split_csv else None
    )
    train_files, val_files = folds[args.fold]

    if args.limit:
        train_files = train_files[:args.limit]
        val_files = val_files[:args.limit]

    print(f"[preproc] Preprocessing: {len(train_files)} Train | {len(val_files)} Val | Workers: {args.num_workers}")
    print(f"[preproc] Storing persistent tensors to: {persist_dir}")
    print(f"[preproc] Storing RAM disk tensors to: {shm_dir}")

    tasks = [(item, True) for item in train_files] + [(item, False) for item in val_files]
    t_start = time.time()

    completed = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_one_patient, item, is_train, persist_dir, shm_dir): item["patient_id"]
            for item, is_train in tasks
        }
        for future in as_completed(futures):
            pid = futures[future]
            try:
                pid_done, cached, el = future.result()
                completed += 1
                status = "CACHED" if cached else f"{el:.1f}s"
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"[{completed:04d}/{len(tasks):04d}] Done: {pid_done} ({status}) | Total Elapsed: {time.time()-t_start:.1f}s", flush=True)
            except Exception as e:
                print(f"[ERROR] Failed patient {pid}: {e}", flush=True)

    total_time = time.time() - t_start
    print(f"\n[preproc] Finished all {len(tasks)} patients in {total_time:.1f}s ({total_time/60:.2f} mins).")


if __name__ == "__main__":
    main()
