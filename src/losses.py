"""
losses.py
---------
Loss functions used for training the SAR-to-EO CycleGAN generator:

  - CharbonnierLoss  : Smooth L1 approximation, more robust than MSE.
  - perceptual_loss  : VGG16 feature-space L1 loss for texture consistency.
  - ms_ssim_loss     : Multi-Scale SSIM loss for structural preservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from pytorch_msssim import ms_ssim


# =============================================================================
# Charbonnier Loss
# =============================================================================

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss — a smooth L1 approximation:
        loss = mean( sqrt((x - y)^2 + ε^2) )

    More robust to outliers than L2; smoother gradient than L1.

    Args:
        epsilon (float): Smoothing constant. Default 1e-3.
    """

    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((x - y) ** 2 + self.epsilon ** 2))


# =============================================================================
# Perceptual Loss (VGG16)
# =============================================================================

def build_vgg_feature_extractor(device: torch.device) -> nn.Module:
    """
    Build a frozen VGG16 feature extractor (first 9 layers).

    Returns:
        A frozen nn.Module that maps images to VGG feature maps.
    """
    vgg = vgg16(pretrained=True).features[:9].eval().to(device)
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg


def perceptual_loss(
    fake: torch.Tensor,
    real: torch.Tensor,
    vgg_net: nn.Module,
) -> torch.Tensor:
    """
    VGG-based perceptual loss — L1 distance between VGG feature maps.

    Args:
        fake    : Generated image tensor in [-1, 1], shape [B, C, H, W].
        real    : Ground-truth image tensor in [-1, 1], shape [B, C, H, W].
        vgg_net : Frozen VGG16 feature extractor (from build_vgg_feature_extractor).

    Returns:
        Scalar perceptual loss.
    """
    return F.l1_loss(vgg_net(fake), vgg_net(real))


# =============================================================================
# MS-SSIM Loss
# =============================================================================

def ms_ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Multi-Scale SSIM loss.
    Inputs must be in [0, 1] (data_range=1.0).

    Args:
        pred   : Predicted image tensor, shape [B, C, H, W].
        target : Ground-truth image tensor, shape [B, C, H, W].

    Returns:
        Scalar loss = 1 - MS-SSIM score.
    """
    return 1.0 - ms_ssim(pred, target, data_range=1.0, size_average=True)
