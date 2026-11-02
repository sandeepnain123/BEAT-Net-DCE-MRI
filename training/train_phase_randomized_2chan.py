"""
Phase-Randomised 2-Channel Training across all 1200 MAMA-MIA Cases
===================================================================
Protocol Invariance Route for DCE-MRI:
  - Trains a 2-channel [I0, post_contrast] network on all 1200 development cases.
  - Phase Randomization: During training, channel 1 is dynamically and uniformly
    sampled between Phase 1 (arterial wash-in, I1) and Phase 2 (delayed wash-out, I2).
  - Pretrained ResNet-50 backbone.
  - 500 Epochs, AdamW, CosineAnnealingLR, DiceFocalLoss.
  - Saves final-epoch checkpoint to final_model.pt at Epoch 500.
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.losses import DiceFocalLoss

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.attention_55_model import AttentionAllDecoderUNet
from data.mama_mia_5fold import PATCH_SIZE


class PhaseRandomizedTrainDataset(Dataset):
    """
    Samples 2-channel patches where:
      Channel 0 = Pre-contrast (I0)
      Channel 1 = Random choice between Phase 1 (I1) and Phase 2 (I2)
    """
    def __init__(self, patient_dirs, patch_size=(128, 128, 64)):
        self.patient_dirs = patient_dirs
        self.patch_size = patch_size
        self.SH, self.SW, self.SD = patch_size

    def __len__(self):
        return len(self.patient_dirs)

    def _crop_and_randomize_phase(self, img, lbl, hc, wc, dc, H, W, D):
        h0 = int(np.clip(hc - self.SH // 2, 0, H - self.SH))
        w0 = int(np.clip(wc - self.SW // 2, 0, W - self.SW))
        d0 = int(np.clip(dc - self.SD // 2, 0, D - self.SD))

        # Sample patch [5, SH, SW, SD]
        patch_5c = img[:, h0:h0 + self.SH, w0:w0 + self.SW, d0:d0 + self.SD]
        patch_lbl = torch.from_numpy(lbl[:, h0:h0 + self.SH, w0:w0 + self.SW, d0:d0 + self.SD].copy())

        # Phase randomization: 50% Phase 1 (arterial), 50% Phase 2 (delayed)
        phase_idx = 1 if np.random.rand() < 0.5 else 2
        ch0 = torch.from_numpy(patch_5c[0:1].copy())
        ch1 = torch.from_numpy(patch_5c[phase_idx:phase_idx + 1].copy())
        patch_2c = torch.cat([ch0, ch1], dim=0)

        return patch_2c, patch_lbl

    def __getitem__(self, idx):
        pdir = self.patient_dirs[idx]
        img = np.load(pdir / "image.npy", mmap_mode="r")
        lbl = np.load(pdir / "label.npy", mmap_mode="r")
        fg = np.load(pdir / "fg_coords.npy") if (pdir / "fg_coords.npy").exists() else np.empty((0, 3), dtype=int)

        H, W, D = img.shape[1], img.shape[2], img.shape[3]

        # Positive patch (tumor-centered)
        if len(fg) > 0:
            c = fg[np.random.randint(len(fg))]
            hc = c[0] + np.random.randint(-16, 17)
            wc = c[1] + np.random.randint(-16, 17)
            dc = c[2] + np.random.randint(-8, 9)
        else:
            hc = np.random.randint(self.SH // 2, max(self.SH // 2 + 1, H - self.SH // 2))
            wc = np.random.randint(self.SW // 2, max(self.SW // 2 + 1, W - self.SW // 2))
            dc = np.random.randint(self.SD // 2, max(self.SD // 2 + 1, D - self.SD // 2))
        pos_img, pos_lbl = self._crop_and_randomize_phase(img, lbl, hc, wc, dc, H, W, D)

        # Negative / random patch
        hc_neg = np.random.randint(self.SH // 2, max(self.SH // 2 + 1, H - self.SH // 2))
        wc_neg = np.random.randint(self.SW // 2, max(self.SW // 2 + 1, W - self.SW // 2))
        dc_neg = np.random.randint(self.SD // 2, max(self.SD // 2 + 1, D - self.SD // 2))
        neg_img, neg_lbl = self._crop_and_randomize_phase(img, lbl, hc_neg, wc_neg, dc_neg, H, W, D)

        imgs = torch.stack([pos_img, neg_img], dim=0)
        lbls = torch.stack([pos_lbl, neg_lbl], dim=0)
        return imgs, lbls


def fast_collate_fn(batch):
    imgs = torch.cat([b[0] for b in batch], dim=0)
    lbls = torch.cat([b[1] for b in batch], dim=0)
    return imgs, lbls


def apply_gpu_augmentations(imgs, lbls):
    if torch.rand(1).item() > 0.5:
        imgs = torch.flip(imgs, dims=[2])
        lbls = torch.flip(lbls, dims=[2])
    if torch.rand(1).item() > 0.5:
        imgs = torch.flip(imgs, dims=[3])
        lbls = torch.flip(lbls, dims=[3])
    if torch.rand(1).item() > 0.5:
        imgs = torch.flip(imgs, dims=[4])
        lbls = torch.flip(lbls, dims=[4])
    if torch.rand(1).item() < 0.2:
        imgs = imgs + 0.1 * torch.randn_like(imgs)
    return imgs, lbls


def get_all_1200_train_dirs(split_csv: Path, mmap_dir: Path, shm_dir: Path):
    with open(split_csv) as f:
        train_pids = [r["train_split"] for r in csv.DictReader(f) if r.get("train_split")]

    base = shm_dir if shm_dir.exists() and len(list(shm_dir.glob("*_train"))) >= 1000 else mmap_dir
    train_dirs = []
    for pid in train_pids:
        d1 = base / f"{pid}_train"
        d2 = base / f"{pid}_val"
        if d1.exists():
            train_dirs.append(d1)
        elif d2.exists():
            train_dirs.append(d2)
    return train_dirs, base


def main():
    p = argparse.ArgumentParser(description="Phase-Randomized 2-Channel Network Training")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--mmap-dir", type=str, default="./data/preprocessed_mmap")
    p.add_argument("--shm-dir", type=str, default="/dev/shm/mama_mia_mmap")
    p.add_argument("--split-csv", type=str, default="./data/metadata/train_test_splits.csv")
    p.add_argument("--pretrained-ckpt", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="./results/phase_randomized_2chan_seed0")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==========================================================================")
    print(f"  PHASE-RANDOMISED 2-CHANNEL TRAINING (All 1200 Cases, Seed {args.seed})")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("==========================================================================")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    split_csv = Path(args.split_csv)
    mmap_dir = Path(args.mmap_dir)
    shm_dir = Path(args.shm_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dirs, active_base = get_all_1200_train_dirs(split_csv, mmap_dir, shm_dir)
    print(f"[data] Found {len(train_dirs)} training patient dirs in {active_base}")

    train_ds = PhaseRandomizedTrainDataset(train_dirs, patch_size=PATCH_SIZE)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size // 2, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=fast_collate_fn
    )

    model = AttentionAllDecoderUNet(in_channels=2, out_channels=1).to(device)
    if args.pretrained_ckpt and os.path.exists(args.pretrained_ckpt):
        print(f"[ckpt] Loading pretrained weights from {args.pretrained_ckpt}")
        ckpt = torch.load(args.pretrained_ckpt, map_location="cpu")
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=True)

    loss_fn = DiceFocalLoss(sigmoid=True, lambda_dice=1.0, lambda_focal=1.0)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=(device.type == "cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    latest_ckpt_path = out_dir / "latest_checkpoint.pt"
    final_ckpt_path = out_dir / "final_model.pt"

    start_epoch = 1
    if latest_ckpt_path.exists():
        ck = torch.load(latest_ckpt_path, map_location=device)
        start_epoch = ck["epoch"] + 1
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        print(f"[resume] Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        step_count = 0

        for imgs, lbls in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            imgs, lbls = apply_gpu_augmentations(imgs, lbls)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(imgs)
                loss = loss_fn(preds, lbls)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step_count += 1

        scheduler.step()
        epoch_loss /= max(1, step_count)
        el = time.time() - t0

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {epoch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {el:.1f}s", flush=True)

        if epoch % 25 == 0 or epoch == args.epochs:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": epoch_loss,
            }, latest_ckpt_path)

    torch.save({
        "epoch": args.epochs,
        "seed": args.seed,
        "model_state_dict": model.state_dict(),
        "total_params": sum(p.numel() for p in model.parameters()),
        "in_channels": 2,
    }, final_ckpt_path)
    print(f"[PROTOCOL] Successfully saved final_model.pt at Epoch {args.epochs} to {final_ckpt_path}")


if __name__ == "__main__":
    main()
