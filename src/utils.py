"""
utils.py
--------
Shared utility functions for the SAR-to-EO CycleGAN project:
  - Weight initialization
  - Tensor denormalization
  - PSNR / SSIM metrics
  - NDVI computation (used for RGB+NIR variant)
  - Visualization helpers (save_sample, show_sample, save_side_by_side_images)
  - Checkpoint export (zip_checkpoints)
"""

import os
import zipfile

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


# =============================================================================
# Weight Initialization
# =============================================================================

def init_weights(net: nn.Module) -> None:
    """
    Initialize Conv2d, ConvTranspose2d, and BatchNorm2d layers with
    N(0, 0.02) weights and zero biases — standard GAN initialization.
    """
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0)


# =============================================================================
# Tensor Normalization / Denormalization
# =============================================================================

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor from [-1, 1] to [0, 1], clamped.

    Args:
        tensor: Any float tensor with values in [-1, 1].

    Returns:
        Tensor with values in [0, 1].
    """
    return (tensor * 0.5 + 0.5).clamp(0, 1)


# =============================================================================
# Metrics
# =============================================================================

def compute_psnr(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two tensors.

    Inputs are denormalized from [-1,1] to [0,1] before computation.

    Args:
        tensor1: Predicted tensor (any shape).
        tensor2: Ground-truth tensor (same shape).

    Returns:
        PSNR value (float). Returns inf if MSE=0.
    """
    with torch.no_grad():
        t1 = denormalize(tensor1).clamp(0, 1)
        t2 = denormalize(tensor2).clamp(0, 1)
        mse = torch.mean((t1 - t2) ** 2)
        if mse == 0:
            return float('inf')
        return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()


def compute_multiband_ssim(fake: torch.Tensor, real: torch.Tensor) -> float:
    """
    Compute SSIM across all bands.

    Args:
        fake: Predicted tensor [C, H, W] or [1, C, H, W] in [0, 1].
        real: Ground-truth tensor, same shape.

    Returns:
        SSIM score (float in [0, 1]).
    """
    if fake.ndim == 3:
        fake = fake.unsqueeze(0)
        real = real.unsqueeze(0)
    return ssim(fake, real, data_range=1.0, size_average=True).item()


# =============================================================================
# NDVI Computation (RGB+NIR variant)
# =============================================================================

def compute_ndvi(image: torch.Tensor, red_idx: int = 0, nir_idx: int = 3) -> torch.Tensor:
    """
    Compute NDVI from a predicted EO image tensor.

    NDVI = (NIR - Red) / (NIR + Red + ε)

    Args:
        image   : Tensor [C, H, W] in [0, 1].
        red_idx : Channel index for Red band (default 0 for rgb_nir config).
        nir_idx : Channel index for NIR band (default 3 for rgb_nir config).

    Returns:
        NDVI map as tensor [H, W] with values in [-1, 1].
    """
    red = image[red_idx]
    nir = image[nir_idx]
    return (nir - red) / (nir + red + 1e-6)


# =============================================================================
# Visualization
# =============================================================================

def save_sample(
    real_A: torch.Tensor,
    fake_B: torch.Tensor,
    real_B: torch.Tensor,
    path: str,
) -> None:
    """
    Save a side-by-side grid: VV | VH | Generated EO | Ground Truth EO.

    Args:
        real_A : SAR input batch  [B, 3, H, W] (channels: VV, VH, ratio).
        fake_B : Generated EO     [B, C, H, W].
        real_B : Ground-truth EO  [B, C, H, W].
        path   : Output file path (.png).
    """
    vv = real_A[:, 0:1].repeat(1, 3, 1, 1)   # [B, 3, H, W]
    vh = real_A[:, 1:2].repeat(1, 3, 1, 1)

    # Handle 4-channel EO — take first 3 for display
    fb_display = fake_B[:, :3]
    rb_display = real_B[:, :3]

    grid = torch.cat([vv, vh, fb_display, rb_display], dim=3)
    save_image(grid, path, normalize=True)


def show_sample(
    real_A: torch.Tensor,
    fake_B: torch.Tensor,
    real_B: torch.Tensor,
    idx: int = 0,
) -> None:
    """
    Display a single sample using matplotlib.

    Args:
        real_A : SAR batch [B, 3, H, W].
        fake_B : Generated EO batch [B, C, H, W].
        real_B : Ground-truth EO batch [B, C, H, W].
        idx    : Sample index within the batch to display.
    """
    with torch.no_grad():
        vv = denormalize(real_A[idx:idx+1, 0:1].repeat(1, 3, 1, 1))[0].cpu()
        vh = denormalize(real_A[idx:idx+1, 1:2].repeat(1, 3, 1, 1))[0].cpu()
        fake = denormalize(fake_B[idx:idx+1, :3])[0].cpu()
        real = denormalize(real_B[idx:idx+1, :3])[0].cpu()

    images = [vv, vh, fake, real]
    titles = ['SAR VV', 'SAR VH', 'Generated EO', 'Ground Truth EO']

    plt.figure(figsize=(16, 4))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 4, i + 1)
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def save_side_by_side_images(
    sar_batch: torch.Tensor,
    fake_batch: torch.Tensor,
    real_batch: torch.Tensor,
    output_dir: str = 'generated_grids',
) -> None:
    """
    Save per-sample triplet grids: SAR | Generated EO | Ground Truth EO.

    Args:
        sar_batch  : SAR batch [B, 3, H, W].
        fake_batch : Generated EO batch [B, C, H, W].
        real_batch : Ground-truth EO batch [B, C, H, W].
        output_dir : Directory to save images to.
    """
    os.makedirs(output_dir, exist_ok=True)

    sar_d  = denormalize(sar_batch.cpu())[:, :3]
    fake_d = denormalize(fake_batch.cpu())[:, :3]
    real_d = denormalize(real_batch.cpu())[:, :3]

    for i in range(sar_d.size(0)):
        triplet = torch.stack([sar_d[i], fake_d[i], real_d[i]])  # [3, C, H, W]
        grid = make_grid(triplet, nrow=3, padding=2)
        save_image(grid, os.path.join(output_dir, f'sample_{i:03d}.png'))

    print(f"Saved {sar_d.size(0)} side-by-side image triplets in '{output_dir}/'")


# =============================================================================
# Checkpoint Export
# =============================================================================

def zip_checkpoints(checkpoint_dir: str = 'checkpoints', zip_filename: str = 'model_checkpoint.zip') -> str:
    """
    Compress all checkpoint files into a single .zip archive.

    Args:
        checkpoint_dir : Directory containing checkpoint files.
        zip_filename   : Output .zip file path.

    Returns:
        Path to the created zip file.
    """
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(checkpoint_dir):
            for file in tqdm(files, desc='Zipping checkpoints'):
                filepath = os.path.join(root, file)
                arcname  = os.path.relpath(filepath, start=checkpoint_dir)
                zipf.write(filepath, arcname=arcname)
    print(f"Checkpoints zipped to '{zip_filename}'")
    return zip_filename
