"""
models.py
---------
Model definitions for SAR-to-EO translation:
  - CBAM attention (ChannelAttention, SpatialAttention, CBAMBlock)
  - Lightweight Generator built from DepthwiseSeparableConv + InvertedResidualBlock
  - PatchGAN Discriminator with spectral normalization
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


# =============================================================================
# CBAM Attention
# =============================================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention module (CBAM).
    Uses both average-pool and max-pool branches through a shared MLP.
    """

    def __init__(self, in_channels: int, ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    """
    Spatial Attention module (CBAM).
    Combines channel-wise mean and max, then convolves to produce a spatial mask.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv    = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        x_cat   = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat)) * x


class CBAMBlock(nn.Module):
    """Full CBAM block: Channel Attention → Spatial Attention."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# =============================================================================
# Lightweight Building Blocks
# =============================================================================

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution block:
      Depthwise Conv (groups=in_channels) → BN → ReLU6
      Pointwise Conv (1×1)                → BN → ReLU6
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InvertedResidualBlock(nn.Module):
    """
    Inverted Residual Block (MobileNetV2-style):
      Expand (1×1) → Depthwise (3×3) → Project (1×1) + skip connection.
    """

    def __init__(self, in_channels: int, expansion_factor: int = 6):
        super().__init__()
        hidden_dim = in_channels * expansion_factor
        self.block = nn.Sequential(
            # Expand
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Depthwise
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Project
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)  # Residual connection


# =============================================================================
# Generator
# =============================================================================

class Generator(nn.Module):
    """
    Lightweight CycleGAN Generator with CBAM attention.

    Architecture:
      Input Projection (3 → 32)
      → Downsample ×2 (stride-2 DepthwiseSeparableConv)
      → Bottleneck (n_blocks × InvertedResidualBlock)
      → CBAM Attention
      → Upsample ×2 (Bilinear + DepthwiseSeparableConv)
      → Output Projection → Tanh

    Args:
        in_channels  (int): Input channels (3 for SAR VV/VH/ratio).
        out_channels (int): Output channels (3 for RGB/NIR-SWIR-RE, 4 for RGB+NIR).
        n_blocks     (int): Number of inverted residual bottleneck blocks.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, n_blocks: int = 4):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        ]

        # Downsample: 32 → 64 → 128
        curr_dim = 32
        for _ in range(2):
            layers.append(DepthwiseSeparableConv(curr_dim, curr_dim * 2, stride=2))
            curr_dim *= 2

        # Bottleneck
        for _ in range(n_blocks):
            layers.append(InvertedResidualBlock(curr_dim))

        # CBAM Attention after bottleneck
        layers.append(CBAMBlock(curr_dim))

        # Upsample: 128 → 64 → 32
        for _ in range(2):
            layers += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                DepthwiseSeparableConv(curr_dim, curr_dim // 2),
            ]
            curr_dim //= 2

        # Output projection
        layers += [
            nn.Conv2d(curr_dim, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# =============================================================================
# Discriminator
# =============================================================================

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator with spectral normalization for training stability.

    Produces a patch-wise real/fake prediction map rather than a single scalar.

    Args:
        in_channels (int): Input image channels (3 for RGB/NIR-SWIR-RE, 4 for RGB+NIR).
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        def _block(in_c: int, out_c: int, norm: bool = True) -> nn.Sequential:
            layers = [spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1))]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            _block(in_channels, 32,  norm=False),
            _block(32,          64),
            _block(64,          128),
            spectral_norm(nn.Conv2d(128, 1, kernel_size=4, padding=1)),  # PatchGAN output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
