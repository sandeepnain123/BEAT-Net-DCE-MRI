import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from monai.networks.nets import FlexibleUNet
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance
from attention_isolated_model import IsolatedSCSEBlock
TARGET_SPACING=(0.8,0.8,1.0)
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

class TestEvalDataset(Dataset):
    def __init__(self, patient_dirs, channel_indices=None):
        self.patient_dirs = patient_dirs
        self.channel_indices = channel_indices

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        pdir = self.patient_dirs[idx]
        img = np.load(pdir / "image.npy", mmap_mode="r")
        lbl = np.load(pdir / "label.npy", mmap_mode="r")
        if self.channel_indices is not None:
            img = img[self.channel_indices]
        pid = pdir.name.replace("_test", "")
        return {
            "image": torch.from_numpy(img.copy()).float(),
            "label": torch.from_numpy(lbl.copy()).float(),
            "pid": pid,
        }

def compute_metrics_pair(p_bin, g_bin, spacing=TARGET_SPACING):
    p = torch.from_numpy(p_bin).bool()
    g = torch.from_numpy(g_bin).bool()
    p_sum = p.sum().item()
    g_sum = g.sum().item()
    inter = (p & g).sum().item()

    if p_sum == 0 and g_sum == 0:
        return 100.0, 100.0, 100.0, 100.0, 0.0, 0.0
    if p_sum == 0 or g_sum == 0:
        diag = float(np.sqrt(sum((s * d) ** 2 for s, d in zip(spacing, p.shape))))
        return 0.0, 0.0, 100.0 if p_sum == 0 else 0.0, 100.0 if g_sum == 0 else 0.0, diag, diag

    dsc = (2.0 * inter) / (p_sum + g_sum) * 100.0
    iou = inter / (p_sum + g_sum - inter) * 100.0
    prec = inter / p_sum * 100.0
    rec = inter / g_sum * 100.0

    p_4d = p.unsqueeze(0).unsqueeze(0).float()
    g_4d = g.unsqueeze(0).unsqueeze(0).float()
    try:
        hd95 = compute_hausdorff_distance(p_4d, g_4d, include_background=True, percentile=95.0, spacing=spacing).item()
    except Exception:
        hd95 = float("nan")
    try:
        assd = compute_average_surface_distance(p_4d, g_4d, include_background=True, symmetric=True, spacing=spacing).item()
    except Exception:
        assd = float("nan")

    return dsc, iou, prec, rec, hd95, assd
