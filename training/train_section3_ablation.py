#!/usr/bin/env python3
"""
Section 3: Input Channel and Pretraining Ablations Protocol
============================================================
Protocol:
  - Train on all 1200 training cases (no folds).
  - Evaluate on the 306 test cases at Epoch 500 with final_model.pt.
  - Seeds: 0 and 1.
  - Epochs: 500.
  - Model: FlexibleUNet with standard scSE attention blocks in all 5 decoder stages.
  - Loss: DiceFocalLoss (lambda_dice=1.0, lambda_focal=1.0, sigmoid=True).
  - Optimizer: AdamW lr=2e-4, weight_decay=1e-5, fused=True.
  - Scheduler: CosineAnnealingLR T_max=500, eta_min=1e-6.

Experiments:
  3.0 Reference: standard_scse, 5-channel [I0,I1,I2,S1,S2], seeds 0, 1
  3.1 Input decomposition:
      - raw3ch: [I0,I1,I2] (raw phases, channels 0, 1, 2)
      - sub2ch: [S1,S2] only (subtractions, channels 3, 4)
  3.2 Encoder pretraining: standard_scse 5-ch with RANDOMLY initialised encoder
  3.3 Encoder depth:
      - resnet18: pretrained ResNet-18, standard_scse, 5-ch
      - resnet34: pretrained ResNet-34, standard_scse, 5-ch
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.networks.nets import FlexibleUNet
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.attention_isolated_model import IsolatedSCSEBlock
from data.mama_mia_5fold import PATCH_SIZE, TARGET_SPACING


class Section3AblationUNet(FlexibleUNet):
    """FlexibleUNet with standard scSE in all 5 decoder stages and selectable backbone"""
    def __init__(self, in_channels: int = 5, out_channels: int = 1, backbone: str = "resnet50"):
        decoder_channels = (512, 256, 128, 64, 32)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            backbone=backbone,
            spatial_dims=3,
            decoder_channels=decoder_channels,
            upsample="deconv",
            norm=("instance", {"affine": True}),
        )
        self.backbone_name = backbone
        self.attention_blocks = nn.ModuleList([
            IsolatedSCSEBlock(ch, reduction=16, mode="standard") for ch in decoder_channels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        skips = [f for f in features[:-1] if f is not None][::-1]
        x = features[-1]

        for i, decoder_block in enumerate(self.decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            x = self.attention_blocks[i](x)

        x = self.segmentation_head(x)
        return x


class FullCohortDataset(Dataset):
    def __init__(self, patient_dirs, channel_indices=(0, 1, 2, 3, 4), patch_size=(128, 128, 64)):
        self.patient_dirs = patient_dirs
        self.channel_indices = list(channel_indices)
        self.patch_size = patch_size
        self.SH, self.SW, self.SD = patch_size

    def __len__(self):
        return len(self.patient_dirs)

    def _crop(self, img, lbl, hc, wc, dc, H, W, D):
        h0 = int(np.clip(hc - self.SH // 2, 0, H - self.SH))
        w0 = int(np.clip(wc - self.SW // 2, 0, W - self.SW))
        d0 = int(np.clip(dc - self.SD // 2, 0, D - self.SD))
        patch_img = torch.from_numpy(img[self.channel_indices, h0:h0 + self.SH, w0:w0 + self.SW, d0:d0 + self.SD].copy())
        patch_lbl = torch.from_numpy(lbl[:, h0:h0 + self.SH, w0:w0 + self.SW, d0:d0 + self.SD].copy())
        return patch_img, patch_lbl

    def __getitem__(self, idx):
        pdir = self.patient_dirs[idx]
        img = np.load(pdir / "image.npy", mmap_mode="r")
        lbl = np.load(pdir / "label.npy", mmap_mode="r")
        fg = np.load(pdir / "fg_coords.npy") if (pdir / "fg_coords.npy").exists() else np.empty((0, 3), dtype=int)

        H, W, D = img.shape[1], img.shape[2], img.shape[3]

        if len(fg) > 0:
            c = fg[np.random.randint(len(fg))]
            hc = c[0] + np.random.randint(-16, 17)
            wc = c[1] + np.random.randint(-16, 17)
            dc = c[2] + np.random.randint(-8, 9)
        else:
            hc = np.random.randint(self.SH // 2, max(self.SH // 2 + 1, H - self.SH // 2))
            wc = np.random.randint(self.SW // 2, max(self.SW // 2 + 1, W - self.SW // 2))
            dc = np.random.randint(self.SD // 2, max(self.SD // 2 + 1, D - self.SD // 2))
        pos_img, pos_lbl = self._crop(img, lbl, hc, wc, dc, H, W, D)

        hc_neg = np.random.randint(self.SH // 2, max(self.SH // 2 + 1, H - self.SH // 2))
        wc_neg = np.random.randint(self.SW // 2, max(self.SW // 2 + 1, W - self.SW // 2))
        dc_neg = np.random.randint(self.SD // 2, max(self.SD // 2 + 1, D - self.SD // 2))
        neg_img, neg_lbl = self._crop(img, lbl, hc_neg, wc_neg, dc_neg, H, W, D)

        imgs = torch.stack([pos_img, neg_img], dim=0)
        lbls = torch.stack([pos_lbl, neg_lbl], dim=0)
        return imgs, lbls


class TestEvalDataset(Dataset):
    def __init__(self, patient_dirs, channel_indices=(0, 1, 2, 3, 4)):
        self.patient_dirs = patient_dirs
        self.channel_indices = list(channel_indices)

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        pdir = self.patient_dirs[idx]
        img = np.load(pdir / "image.npy", mmap_mode="r")[self.channel_indices]
        lbl = np.load(pdir / "label.npy", mmap_mode="r")
        pid = pdir.name.replace("_test", "")
        return torch.from_numpy(img.copy()), torch.from_numpy(lbl.copy()), pid


def collate_fn(batch):
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


def load_transfer_weights(model, ckpt_path, in_channels=5):
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[warn] Checkpoint not found at {ckpt_path}. Training from scratch.")
        return model

    print(f"[ckpt] Loading pretrained weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model_sd = model.state_dict()

    for k, v in sd.items():
        if k == "encoder.conv1.weight":
            if in_channels == 5:
                new_w = torch.zeros(64, 5, 7, 7, 7, device=v.device, dtype=v.dtype)
                new_w[:, :2] = v
                new_w[:, 2:] = v.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1, 1)
                model_sd[k] = new_w
            elif in_channels == 3:
                new_w = torch.zeros(64, 3, 7, 7, 7, device=v.device, dtype=v.dtype)
                new_w[:, :2] = v
                new_w[:, 2] = v.mean(dim=1)
                model_sd[k] = new_w
            elif in_channels == 2:
                model_sd[k] = v
        elif k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v

    model.load_state_dict(model_sd, strict=False)
    return model


def compute_metrics(p_bin, g_bin, spacing=TARGET_SPACING):
    p = p_bin.bool()
    g = g_bin.bool()
    p_sum = p.sum().item()
    g_sum = g.sum().item()
    inter = (p & g).sum().item()

    if p_sum == 0 and g_sum == 0:
        return 100.0, 100.0, 100.0, 100.0, 0.0, 0.0
    if p_sum == 0 or g_sum == 0:
        diag = float(np.sqrt(sum((s * d) ** 2 for s, d in zip(spacing, p_bin.shape))))
        return 0.0, 0.0, (100.0 if g_sum == 0 else 0.0), (100.0 if p_sum == 0 else 0.0), diag, diag

    union = p_sum + g_sum - inter
    dsc = (2.0 * inter) / (p_sum + g_sum) * 100.0
    iou = (inter / union) * 100.0 if union > 0 else 0.0
    prec = (inter / p_sum) * 100.0 if p_sum > 0 else 0.0
    rec = (inter / g_sum) * 100.0 if g_sum > 0 else 0.0

    idx = torch.where(p | g)
    pad = 4
    hmin, hmax = max(0, idx[0].min().item() - pad), min(p_bin.shape[0], idx[0].max().item() + pad + 1)
    wmin, wmax = max(0, idx[1].min().item() - pad), min(p_bin.shape[1], idx[1].max().item() + pad + 1)
    dmin, dmax = max(0, idx[2].min().item() - pad), min(p_bin.shape[2], idx[2].max().item() + pad + 1)

    p_crop = p[hmin:hmax, wmin:wmax, dmin:dmax].unsqueeze(0).unsqueeze(0).float()
    g_crop = g[hmin:hmax, wmin:wmax, dmin:dmax].unsqueeze(0).unsqueeze(0).float()

    try:
        hd95 = compute_hausdorff_distance(p_crop, g_crop, include_background=True, percentile=95.0, spacing=spacing).item()
    except Exception:
        hd95 = float("nan")
    try:
        assd = compute_average_surface_distance(p_crop, g_crop, include_background=True, symmetric=True, spacing=spacing).item()
    except Exception:
        assd = float("nan")

    return dsc, iou, prec, rec, hd95, assd


def evaluate_test_set(model, test_loader, device):
    model.eval()
    per_case = []
    with torch.no_grad():
        for val_imgs, val_labels, pid in test_loader:
            val_imgs = val_imgs.to(device)
            val_labels = val_labels.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = sliding_window_inference(val_imgs, PATCH_SIZE, sw_batch_size=8, predictor=model, overlap=0.5, mode="gaussian")
            p_bin = (torch.sigmoid(logits[0, 0]) > 0.5).to(torch.uint8)
            g_bin = (val_labels[0, 0] > 0.5).to(torch.uint8)
            dsc, iou, prec, rec, hd95, assd = compute_metrics(p_bin.cpu(), g_bin.cpu())
            per_case.append({"pid": pid[0], "dsc": dsc, "iou": iou, "precision": prec, "recall": rec, "hd95": hd95, "assd": assd})

    dscs = [c["dsc"] for c in per_case]
    ious = [c["iou"] for c in per_case]
    precs = [c["precision"] for c in per_case]
    recs = [c["recall"] for c in per_case]
    hd95s = [c["hd95"] for c in per_case if not np.isnan(c["hd95"])]
    assds = [c["assd"] for c in per_case if not np.isnan(c["assd"])]

    summary = {
        "dsc_mean": float(np.mean(dscs)), "dsc_std": float(np.std(dscs)), "dsc_median": float(np.median(dscs)),
        "iou_mean": float(np.mean(ious)), "iou_std": float(np.std(ious)), "iou_median": float(np.median(ious)),
        "precision_mean": float(np.mean(precs)), "recall_mean": float(np.mean(recs)),
        "hd95_mean": float(np.mean(hd95s)), "assd_mean": float(np.mean(assds)),
    }
    return summary, per_case


def main():
    parser = argparse.ArgumentParser(description="Section 3 Input Channel & Pretraining Ablation Trainer")
    parser.add_argument("--exp", type=str, required=True, choices=["3.0_ref", "3.1_raw3ch", "3.1_sub2ch", "3.2_random_init", "3.3_resnet18", "3.3_resnet34"])
    parser.add_argument("--seed", type=int, required=True, choices=[0, 1])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--mmap-dir", type=str, default="./data/preprocessed_mmap")
    parser.add_argument("--pretrained-ckpt", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="./results/ablation_section3")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==========================================================================")
    print(f"  STARTING SECTION 3 ABLATION: Exp={args.exp} | Seed={args.seed}")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("==========================================================================")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.exp == "3.0_ref":
        in_channels = 5
        channel_indices = [0, 1, 2, 3, 4]
        backbone = "resnet50"
        load_weights = True
    elif args.exp == "3.1_raw3ch":
        in_channels = 3
        channel_indices = [0, 1, 2]
        backbone = "resnet50"
        load_weights = True
    elif args.exp == "3.1_sub2ch":
        in_channels = 2
        channel_indices = [3, 4]
        backbone = "resnet50"
        load_weights = True
    elif args.exp == "3.2_random_init":
        in_channels = 5
        channel_indices = [0, 1, 2, 3, 4]
        backbone = "resnet50"
        load_weights = False
    elif args.exp == "3.3_resnet18":
        in_channels = 5
        channel_indices = [0, 1, 2, 3, 4]
        backbone = "resnet18"
        load_weights = False
    elif args.exp == "3.3_resnet34":
        in_channels = 5
        channel_indices = [0, 1, 2, 3, 4]
        backbone = "resnet34"
        load_weights = False

    out_folder = Path(args.out_dir) / f"{args.exp}_seed{args.seed}"
    out_folder.mkdir(parents=True, exist_ok=True)
    final_ckpt_path = out_folder / "final_model.pt"
    metrics_path = out_folder / "metrics.json"
    per_case_csv_path = out_folder / f"test_per_case_{args.exp}_seed{args.seed}.csv"

    mmap_dir = Path(args.mmap_dir)
    train_dirs = sorted([d for d in mmap_dir.iterdir() if d.is_dir() and (d.name.endswith("_train") or d.name.endswith("_val"))])
    test_dirs = sorted([d for d in mmap_dir.iterdir() if d.is_dir() and d.name.endswith("_test")])
    print(f"[data] Loaded {len(train_dirs)} Training Cases (all 1200) | {len(test_dirs)} Test Cases (306)")

    train_ds = FullCohortDataset(train_dirs, channel_indices=channel_indices, patch_size=PATCH_SIZE)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size // 2, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )

    model = Section3AblationUNet(in_channels=in_channels, out_channels=1, backbone=backbone).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    if load_weights and args.pretrained_ckpt:
        model = load_transfer_weights(model, args.pretrained_ckpt, in_channels=in_channels)
    else:
        print("[model] Initialized encoder weights randomly.")

    loss_fn = DiceFocalLoss(sigmoid=True, lambda_dice=1.0, lambda_focal=1.0)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=(device.type == "cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    latest_ckpt_path = out_folder / "latest_checkpoint.pt"
    start_epoch = 1
    history = []

    if latest_ckpt_path.exists():
        ck = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        start_epoch = ck["epoch"] + 1
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        history = ck.get("history", [])

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
                "history": history,
            }, latest_ckpt_path)

    torch.save({
        "epoch": args.epochs,
        "exp": args.exp,
        "seed": args.seed,
        "model_state_dict": model.state_dict(),
        "total_params": total_params,
        "in_channels": in_channels,
        "backbone": backbone,
    }, final_ckpt_path)
    print(f"\n[PROTOCOL] Saved final model checkpoint at epoch {args.epochs} to {final_ckpt_path}")

    # Single held-out test evaluation conducted strictly post-training at epoch 500
    if len(test_dirs) > 0:
        print("[eval] Running single final evaluation on 306 held-out test cases...")
        test_ds = TestEvalDataset(test_dirs, channel_indices=channel_indices)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        final_summary, final_per_case = evaluate_test_set(model, test_loader, device)
        df_per_case = pd.DataFrame(final_per_case)
        df_per_case.to_csv(per_case_csv_path, index=False)
        print(f"[done] Saved final per-case CSV to {per_case_csv_path}")

        with open(metrics_path, "w") as fp:
            json.dump({
                "exp": args.exp, "seed": args.seed, "backbone": backbone, "in_channels": in_channels,
                "total_params": total_params, "last_logged_epoch": args.epochs,
                "final_summary": final_summary, "history": history
            }, fp, indent=2)


if __name__ == "__main__":
    main()
