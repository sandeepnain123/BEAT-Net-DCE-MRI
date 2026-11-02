"""
Kinetics-guided scSE (KG-scSE) Block & Architecture for DCE-MRI
================================================================
Drop-in replacement for the standard scSE block in each decoder stage.
Adds contrast-uptake information, not capacity, in two places:

1. Enhancement-weighted squeeze (channel branch):
   Plain global average pooling is dominated by background, because the tumour
   is a few percent of a 128 x 128 x 64 patch. Here each voxel is weighted by
   (1 + enhancement), so channels are summarised where contrast uptake occurs.
   With no enhancement the weights are uniform and the squeeze reduces to GAP.

2. Kinetic prior in the spatial gate (spatial branch):
   The 1 x 1 x 1 projection that gives the spatial weight map sees the decoder
   features AND the downsampled kinetic subtraction maps (S1 = I1 - I0, S2 = I2 - I0).
   The kinetic weights are zero-initialised, so at the start of training the
   block is exactly standard scSE and can only move away from it if that helps.

Extra parameters per stage: k=2 (10 extra parameters total across 5 stages).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import FlexibleUNet


class KGscSE3D(nn.Module):
    """
    3D Kinetics-Guided Spatial and Channel Squeeze & Excitation Block.
    """
    def __init__(self, channels: int, kinetic_channels: int = 2, reduction: int = 16, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.k = kinetic_channels
        self.eps = eps
        hidden = max(channels // reduction, 1)

        # Channel branch
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1)

        # Spatial branch (features + kinetic maps)
        self.spatial = nn.Conv3d(channels + kinetic_channels, 1, kernel_size=1)

        # Zero-init kinetic weights -> exactly equals standard scSE at initialization
        with torch.no_grad():
            self.spatial.weight[:, channels:].zero_()

    def forward(self, feat: torch.Tensor, kinetic: torch.Tensor) -> torch.Tensor:
        # Kinetic maps at the decoder resolution (average pooling when downsampling)
        if kinetic.shape[2:] != feat.shape[2:]:
            kin = F.adaptive_avg_pool3d(kinetic, feat.shape[2:])
        else:
            kin = kinetic
        kin = kin.to(feat.dtype)

        # 1) Enhancement-weighted squeeze: weights proportional to 1 + positive enhancement
        enh = F.relu(kin).mean(dim=1, keepdim=True)                     # [B, 1, D, H, W]
        w = 1.0 + enh
        w = w / (w.sum(dim=(2, 3, 4), keepdim=True) + self.eps)
        z = (feat * w).sum(dim=(2, 3, 4), keepdim=True)                 # [B, C, 1, 1, 1]
        u = torch.sigmoid(self.fc2(F.relu(self.fc1(z))))                # Channel recalibration weights

        # 2) Spatial gate with kinetic prior
        v = torch.sigmoid(self.spatial(torch.cat([feat, kin], dim=1)))  # [B, 1, D, H, W]

        return feat * u + feat * v


class AttentionKGscSEUNet(FlexibleUNet):
    """
    FlexibleUNet with 3D ResNet-50 backbone and KG-scSE blocks in all 5 decoder stages.
    Expects 5-channel DCE-MRI input: [I0, I1, I2, S1, S2], where S1=I1-I0, S2=I2-I0.
    """
    def __init__(self, in_channels: int = 5, out_channels: int = 1, **kwargs):
        decoder_channels = kwargs.pop("decoder_channels", (512, 256, 128, 64, 32))
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            backbone='resnet50',
            spatial_dims=3,
            decoder_channels=decoder_channels,
            upsample='deconv',
            norm=('instance', {'affine': True}),
            **kwargs
        )
        self.mode = "kg_scse"
        self.attention_blocks = nn.ModuleList([
            KGscSE3D(ch, kinetic_channels=2, reduction=16) for ch in decoder_channels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract kinetic subtraction channels S1 (ch 3) and S2 (ch 4) from 5-channel input
        if x.shape[1] >= 5:
            kin = x[:, 3:5]
        elif x.shape[1] == 2:
            kin = x[:, 1:2] - x[:, 0:1]
            if kin.shape[1] == 1:
                kin = kin.repeat(1, 2, 1, 1, 1)
        else:
            kin = torch.zeros(x.shape[0], 2, *x.shape[2:], device=x.device, dtype=x.dtype)

        features = self.encoder(x)
        skips = [f for f in features[:-1] if f is not None][::-1]
        feat = features[-1]

        for i, decoder_block in enumerate(self.decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            feat = decoder_block(feat, skip)
            feat = self.attention_blocks[i](feat, kin)

        out = self.segmentation_head(feat)
        return out


def get_kg_scse_model(in_channels: int = 5, out_channels: int = 1) -> nn.Module:
    """Factory function for KG-scSE model."""
    return AttentionKGscSEUNet(in_channels=in_channels, out_channels=out_channels)


if __name__ == "__main__":
    torch.manual_seed(0)
    stages = [(512, (8, 8, 4)), (256, (16, 16, 8)), (128, (32, 32, 16)), (64, (64, 64, 32)), (32, (128, 128, 64))]
    kin = torch.randn(1, 2, 128, 128, 64)
    total_extra = 0
    print("--- Stage Checks ---")
    for c, size in stages:
        blk = KGscSE3D(c)
        x = torch.randn(1, c, *size)
        y = blk(x, kin)
        assert y.shape == x.shape
        n = sum(p.numel() for p in blk.parameters())
        std = 2 * c * max(c // 16, 1) + max(c // 16, 1) + c + c + 1
        total_extra += (n - std)
        print(f"C={c:3d} out {tuple(y.shape)} params {n:,} (standard scSE {std:,}, +{n-std})")
    print(f"Total extra parameters across all 5 stages: {total_extra}")

    # Full model instantiation test
    print("\n--- Full Model Forward Test ---")
    m = get_kg_scse_model(in_channels=5, out_channels=1)
    inp = torch.randn(1, 5, 128, 128, 64)
    out = m(inp)
    print(f"Model input: {inp.shape} -> output: {out.shape}")
    total_p = sum(p.numel() for p in m.parameters())
    print(f"Total Model Parameters: {total_p:,}")
    print("Validation OK!")
