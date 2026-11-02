"""
Isolated Attention Block Architectures for Ablation Study
===========================================================
Defines IsolatedSCSEBlock with distinct operational modes:
  1. 'standard': Roy et al. (2018) spatial excitation (Conv3d 1x1x1, C -> 1, +0M params)
  2. 'boundary_full': Full boundary-aware spatial excitation (Conv3d 3x3x3, C -> C -> 1, +9.43M params)
  3. 'boundary_matched': Capacity-matched bottlenecked spatial excitation (Conv3d 3x3x3, C -> C/r -> 1, +0.59M params)
  4. 'pointwise_deep': 1x1x1 conv with exact same depth & nonlinearity as boundary_full (C -> C -> 1)
  5. 'no_scse': Identity pass-through (no attention)
  6. 'kg_scse': Kinetics-guided scSE block
  7. 'segresnet': MONAI SegResNet baseline
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from monai.networks.nets import FlexibleUNet

# Allow importing sibling modules whether imported as package or directly
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

try:
    from .best_model_no_scse_block import get_no_scse_model
    from .best_model_KG_scSE_block import get_kg_scse_model
except ImportError:
    from best_model_no_scse_block import get_no_scse_model
    from best_model_KG_scSE_block import get_kg_scse_model


class IsolatedSCSEBlock(nn.Module):
    """Spatial and Channel Squeeze & Excitation Block with Isolated Boundary Variants"""
    def __init__(self, in_channels: int, reduction: int = 16, mode: str = "standard"):
        super().__init__()
        self.mode = mode
        reduced_c = max(1, in_channels // reduction)

        # 1. Channel Squeeze & Excitation (cSE) - identical across all variants
        self.channel_excitation = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, reduced_c, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_c, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 2. Spatial Squeeze & Excitation (sSE) - isolated variations
        if mode == "standard":
            # Baseline: 1x1x1 conv (Roy et al., 2018)
            self.spatial_excitation = nn.Sequential(
                nn.Conv3d(in_channels, 1, kernel_size=1),
                nn.Sigmoid()
            )
        elif mode == "boundary_full":
            # Full 3x3x3 conv: in_channels -> in_channels -> 1 (+9.43M params)
            self.spatial_excitation = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(in_channels, affine=True),
                nn.PReLU(),
                nn.Conv3d(in_channels, 1, kernel_size=1),
                nn.Sigmoid()
            )
        elif mode == "boundary_matched":
            # Capacity-matched: in_channels -> in_channels//reduction -> 1
            self.spatial_excitation = nn.Sequential(
                nn.Conv3d(in_channels, reduced_c, kernel_size=3, padding=1),
                nn.InstanceNorm3d(reduced_c, affine=True),
                nn.PReLU(),
                nn.Conv3d(reduced_c, 1, kernel_size=1),
                nn.Sigmoid()
            )
        elif mode == "pointwise_deep":
            # Key Control: 1x1x1 conv with exact same depth & nonlinearity as boundary_full (C -> C -> 1)
            # Isolates whether the 3D spatial receptive field (3x3x3) vs depth/PReLU drives the gain
            self.spatial_excitation = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=1),
                nn.InstanceNorm3d(in_channels, affine=True),
                nn.PReLU(),
                nn.Conv3d(in_channels, 1, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown block mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chn_se = self.channel_excitation(x) * x
        spa_se = self.spatial_excitation(x) * x
        return chn_se + spa_se


class IsolatedAttentionUNet(FlexibleUNet):
    """FlexibleUNet with configurable IsolatedSCSEBlock across all 5 decoder stages"""
    def __init__(self, in_channels: int = 5, out_channels: int = 1, mode: str = "standard", **kwargs):
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
        self.mode = mode
        self.attention_blocks = nn.ModuleList([
            IsolatedSCSEBlock(ch, reduction=16, mode=mode) for ch in decoder_channels
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


def get_isolated_model(mode: str = "standard", in_channels: int = 5, out_channels: int = 1) -> nn.Module:
    """Factory function for isolated ablation models and baselines"""
    if mode == "segresnet":
        from monai.networks.nets import SegResNet
        model = SegResNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels, init_filters=32)
        model.mode = "segresnet"
        return model
    elif mode == "no_scse":
        return get_no_scse_model(in_channels=in_channels, out_channels=out_channels)
    elif mode == "kg_scse":
        return get_kg_scse_model(in_channels=in_channels, out_channels=out_channels)
    return IsolatedAttentionUNet(in_channels=in_channels, out_channels=out_channels, mode=mode)
