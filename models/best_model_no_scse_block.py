"""
No-scSE 3D Architecture for DCE-MRI Decoder Ablation
=====================================================
Baseline architecture without any spatial or channel attention blocks.
Pure 3D U-Net with ResNet-50 encoder backbone (MONAI FlexibleUNet).

Parameters:
  - ResNet-50 Encoder + 5-stage deconv Decoder
  - Attention Blocks: None (Identity pass-through in all decoder stages)
  - Total parameters: 109,087,009 (~109.09M), exactly 45,699 fewer than standard scSE.
"""

import torch
import torch.nn as nn
from monai.networks.nets import FlexibleUNet


class NoSCSEUNet(FlexibleUNet):
    """
    3D U-Net with ResNet-50 backbone and NO attention blocks in decoder stages.
    Serves as the un-recalibrated baseline to quantify the exact contribution
    of attention recalibration (No-scSE vs Standard scSE vs KG-scSE vs BEAT-Net).
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
        self.mode = "no_scse"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        skips = [f for f in features[:-1] if f is not None][::-1]
        feat = features[-1]

        for i, decoder_block in enumerate(self.decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            feat = decoder_block(feat, skip)
            # No attention block: pure feed-forward deconvolution

        out = self.segmentation_head(feat)
        return out


def get_no_scse_model(in_channels: int = 5, out_channels: int = 1) -> nn.Module:
    """Factory function for No-scSE model."""
    return NoSCSEUNet(in_channels=in_channels, out_channels=out_channels)


if __name__ == "__main__":
    m = get_no_scse_model(in_channels=5, out_channels=1)
    inp = torch.randn(1, 5, 128, 128, 64)
    out = m(inp)
    total_p = sum(p.numel() for p in m.parameters())
    print(f"No-scSE Model Input: {inp.shape} -> Output: {out.shape}")
    print(f"Total Parameters: {total_p:,} (Std scSE: 109,132,708, delta: -{109132708 - total_p:,})")
