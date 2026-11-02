"""
AttentionAllDecoderUNet Architecture (FlexibleUNet + scSE decoder)
==================================================================
3D ResNet-50 backbone with 5 Spatial & Channel Squeeze-and-Excitation (scSE) Attention Blocks.
Supports in_channels=2 and in_channels=5.
"""

import torch
import torch.nn as nn
from monai.networks.nets import FlexibleUNet

class SCSEBlock(nn.Module):
    """Spatial and Channel Squeeze & Excitation Block (Roy et al., 2018)"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.channel_excitation = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, max(1, in_channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(1, in_channels // reduction), in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_excitation = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chn_se = self.channel_excitation(x) * x
        spa_se = self.spatial_excitation(x) * x
        return chn_se + spa_se

class AttentionAllDecoderUNet(FlexibleUNet):
    """FlexibleUNet with scSE attention blocks in all 5 decoder stages"""
    def __init__(self, in_channels: int = 2, out_channels: int = 1, **kwargs):
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
        self.attention_blocks = nn.ModuleList([
            SCSEBlock(ch) for ch in decoder_channels
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

def get_attention55_model(in_channels: int = 2, out_channels: int = 1) -> nn.Module:
    """Returns initialized AttentionAllDecoderUNet model."""
    return AttentionAllDecoderUNet(in_channels=in_channels, out_channels=out_channels)
