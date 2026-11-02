"""
MAMA-MIA 5-Fold Trainer for AttentionAllDecoderUNet (BEAT-Net / scSE decoder)
=============================================================================
Fast, resilient 5-fold cross validation trainer:
- Zero initial caching delay (starts training in 2 seconds)
- Auto-resume from intermediate checkpoints if preempted by Slurm
- Saves final_model.pt (Epoch 500) and best_model.pt per fold
- Adheres to PROTOCOL.md: evaluation strictly on final_model.pt at Epoch 500
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.data import DataLoader, Dataset
from monai.transforms import AsDiscrete

# Add repository root to path for imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.attention_55_model import AttentionAllDecoderUNet
from data.mama_mia_5fold import get_5fold_splits, get_transforms, process_5chan_batch, PATCH_SIZE


def parse_args():
    p = argparse.ArgumentParser(description="MAMA-MIA 5-Fold Cross Validation Trainer")
    p.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4], help="Cross-validation fold index")
    p.add_argument("--in-channels", type=int, default=5, choices=[2, 5], help="Input channel count (2 or 5)")
    p.add_argument("--epochs", type=int, default=500, help="Total training epochs (default: 500)")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    p.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay (default: 1e-5)")
    p.add_argument("--val-interval", type=int, default=5, help="Validation interval (epochs)")
    p.add_argument("--num-workers", type=int, default=4, help="Data loader workers")
    p.add_argument("--data-root", type=str, default=os.environ.get("MAMAMIA_ROOT", "./data/MAMA-MIA"))
    p.add_argument("--split-csv", type=str, default=None, help="Custom split CSV path")
    p.add_argument("--pretrained-ckpt", type=str, default=None, help="Path to pretrained weights checkpoint")
    p.add_argument("--out-dir", type=str, default="./results", help="Directory to save checkpoints and logs")
    return p.parse_args()


def load_pretrained_weights(model, ckpt_path, in_channels):
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"[warn] Pretrained checkpoint not found or not provided ({ckpt_path}), training from scratch.")
        return model

    print(f"[ckpt] Loading pretrained weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    if in_channels == 2:
        model.load_state_dict(sd, strict=True)
        print("[ckpt] Successfully loaded strict 2-channel pretrained weights!")
    elif in_channels == 5:
        model_sd = model.state_dict()
        for k, v in sd.items():
            if k == "encoder.conv1.weight":
                new_w = torch.zeros(64, 5, 7, 7, 7, device=v.device, dtype=v.dtype)
                new_w[:, :2] = v
                mean_channel = v.mean(dim=1, keepdim=True)
                new_w[:, 2:] = mean_channel.repeat(1, 3, 1, 1, 1)
                model_sd[k] = new_w
            elif k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v
        model.load_state_dict(model_sd, strict=False)
        print("[ckpt] Successfully transferred pretrained weights to 5-channel model!")

    return model


def train_fold(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Fast 5-Fold Training: Fold {args.fold} | In-Channels: {args.in_channels} | Device: {device} ===")

    out_folder = Path(args.out_dir) / f"5fold_{args.in_channels}chan" / f"fold_{args.fold}"
    out_folder.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_folder / "best_model.pt"
    last_ckpt_path = out_folder / "latest_checkpoint.pt"
    final_ckpt_path = out_folder / "final_model.pt"
    metrics_path = out_folder / "metrics.json"

    data_root = Path(args.data_root)
    split_csv = Path(args.split_csv) if args.split_csv else None

    folds, test_data = get_5fold_splits(
        in_channels=args.in_channels,
        seed=42,
        dataset_root=data_root,
        split_csv=split_csv
    )
    train_files, val_files = folds[args.fold]
    print(f"[data] Fold {args.fold} - Train patients: {len(train_files)} | Val patients: {len(val_files)}")

    train_tf = get_transforms(in_channels=args.in_channels, is_train=True)
    val_tf = get_transforms(in_channels=args.in_channels, is_train=False)

    train_ds = Dataset(data=train_files, transform=train_tf)
    val_ds = Dataset(data=val_files, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = AttentionAllDecoderUNet(in_channels=args.in_channels, out_channels=1).to(device)
    model = load_pretrained_weights(model, args.pretrained_ckpt, args.in_channels)

    loss_fn = DiceFocalLoss(sigmoid=True, lambda_dice=1.0, lambda_focal=1.0)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    best_val_dice = 0.0
    history = []

    # Auto-resume if interrupted
    resume_ckpt = None
    if last_ckpt_path.exists():
        resume_ckpt = last_ckpt_path
    elif best_ckpt_path.exists():
        resume_ckpt = best_ckpt_path

    if resume_ckpt is not None:
        print(f"[resume] Found existing checkpoint {resume_ckpt}, resuming...")
        last_ckpt = torch.load(resume_ckpt, map_location=device)
        start_epoch = last_ckpt["epoch"] + 1
        model.load_state_dict(last_ckpt["model_state_dict"])
        if "optimizer_state_dict" in last_ckpt:
            optimizer.load_state_dict(last_ckpt["optimizer_state_dict"])
        best_val_dice = last_ckpt.get("best_val_dice", last_ckpt.get("val_dice", 0.0))
        history = last_ckpt.get("history", [])
        print(f"[resume] Resumed from epoch {start_epoch} (best_val_dice: {best_val_dice:.4f})")
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6, last_epoch=start_epoch - 1)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(threshold=0.5)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        step_count = 0

        for batch in train_loader:
            batch = process_5chan_batch(batch)
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step_count += 1

        scheduler.step()
        epoch_loss /= max(1, step_count)
        elapsed = time.time() - t0

        if epoch % args.val_interval == 0 or epoch == args.epochs:
            model.eval()
            dice_metric.reset()

            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = process_5chan_batch(val_batch)
                    val_imgs = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        val_outputs = sliding_window_inference(
                            val_imgs, PATCH_SIZE, sw_batch_size=4, predictor=model, overlap=0.5
                        )

                    val_preds = post_pred(torch.sigmoid(val_outputs))
                    dice_metric(y_pred=val_preds, y=val_labels)

            val_dice = dice_metric.aggregate().item()
            dice_metric.reset()

            print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {epoch_loss:.4f} | Val Dice: {val_dice:.4f} | time: {elapsed:.1f}s", flush=True)

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "fold": args.fold,
                    "in_channels": args.in_channels
                }, best_ckpt_path)
                print(f"  [* BEST] Saved new best fold {args.fold} model (val_dice: {val_dice:.4f}) to {best_ckpt_path}", flush=True)

            history.append({
                "epoch": epoch, "train_loss": epoch_loss,
                "val_dice": val_dice, "best_val_dice": best_val_dice
            })

            # Save latest checkpoint for resume safety
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "history": history,
                "fold": args.fold,
                "in_channels": args.in_channels
            }, last_ckpt_path)

    # Strictly save final_model.pt at completion as mandated by PROTOCOL.md
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "final_val_dice": val_dice,
        "best_val_dice": best_val_dice,
        "fold": args.fold,
        "in_channels": args.in_channels,
    }, final_ckpt_path)
    print(f"[PROTOCOL] Successfully saved final_model.pt at Epoch {args.epochs} to {final_ckpt_path}")

    metrics_path.write_text(json.dumps({
        "fold": args.fold,
        "in_channels": args.in_channels,
        "best_val_dice": best_val_dice,
        "final_val_dice": val_dice,
        "last_logged_epoch": args.epochs,
        "history": history
    }, indent=2))
    print(f"=== Completed Fold {args.fold} Training | Best Val Dice: {best_val_dice:.4f} ===")


if __name__ == "__main__":
    args = parse_args()
    train_fold(args)
