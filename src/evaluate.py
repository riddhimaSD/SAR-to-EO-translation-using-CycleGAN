"""
evaluate.py  (src/)
-------------------
Evaluator class for running inference on the test set after training.

Computes:
  - Average PSNR across all test samples
  - Average SSIM across all test samples
  - NDVI analysis (only for 'rgb_nir' band_config)

Saves side-by-side triplet images (SAR | Generated EO | Ground Truth EO).
"""

import os
import random
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.dataset import SARToEODataset
from src.models import Generator
from src.utils import (
    denormalize,
    compute_psnr,
    compute_multiband_ssim,
    compute_ndvi,
    show_sample,
    save_side_by_side_images,
)


class Evaluator:
    """
    Loads a trained Generator checkpoint and evaluates it on the test set.

    Args:
        config (dict): Config dict (same format as used for training).
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg    = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Evaluator] Using device: {self.device}")

        band_config  = config['band_config']
        out_channels = 4 if band_config == 'rgb_nir' else 3

        # Build and load model
        checkpoint_dir   = config.get('checkpoint_dir', './checkpoints')
        checkpoint_path  = os.path.join(checkpoint_dir, 'G_best.pt')
        self.G = Generator(in_channels=3, out_channels=out_channels).to(self.device)
        self.G.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.G.eval()
        print(f"[Evaluator] Loaded checkpoint: {checkpoint_path}")

        # Build test dataloader
        self.test_loader = self._build_test_loader(config)
        self.band_config = band_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, float]:
        """
        Run full test-set evaluation.

        Returns:
            dict with keys: 'avg_psnr', 'avg_ssim',
            and optionally 'avg_ndvi_diff' for rgb_nir.
        """
        total_psnr = 0.0
        total_ssim = 0.0
        total_ndvi_diff = 0.0
        count = 0

        with torch.no_grad():
            for batch_idx, (real_A, real_B) in enumerate(self.test_loader):
                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)
                fake_B = self.G(real_A)

                for i in range(real_A.size(0)):
                    pred = denormalize(fake_B[i].cpu())
                    gt   = denormalize(real_B[i].cpu())

                    total_psnr += compute_psnr(pred, gt)
                    total_ssim += compute_multiband_ssim(pred, gt)

                    if self.band_config == 'rgb_nir':
                        ndvi_pred = compute_ndvi(pred, red_idx=0, nir_idx=3)
                        ndvi_real = compute_ndvi(gt,   red_idx=0, nir_idx=3)
                        total_ndvi_diff += (ndvi_pred - ndvi_real).abs().mean().item()

                    count += 1

                # On the first batch, display and save a visual sample
                if batch_idx == 0:
                    idx = random.randint(0, real_A.size(0) - 1)
                    show_sample(real_A, fake_B, real_B, idx=idx)

                    output_dir = self.cfg.get('output_dir', './outputs')
                    save_side_by_side_images(
                        real_A.cpu(), fake_B.cpu(), real_B.cpu(),
                        output_dir=os.path.join(output_dir, 'eval_grids'),
                    )

        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count

        results = {'avg_psnr': avg_psnr, 'avg_ssim': avg_ssim}

        print(f"\n[Results] Average PSNR: {avg_psnr:.2f} dB")
        print(f"[Results] Average SSIM: {avg_ssim:.4f}")

        if self.band_config == 'rgb_nir':
            avg_ndvi = total_ndvi_diff / count
            results['avg_ndvi_diff'] = avg_ndvi
            print(f"[Results] Average |NDVI diff|: {avg_ndvi:.4f}")

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_test_loader(self, config: dict) -> DataLoader:
        data_root = config['data_root']
        band_cfg  = config['band_config']
        n_test    = config.get('test_samples', 400)
        batch_size = config.get('batch_size', 8)

        test_tf = transforms.Resize((256, 256))
        test_full = SARToEODataset(
            sar_dir=os.path.join(data_root, 'test', 'SAR'),
            eo_dir=os.path.join(data_root,  'test', 'EO'),
            transform=test_tf, band_config=band_cfg,
        )
        test_ds = Subset(test_full, list(range(min(n_test, len(test_full)))))
        print(f"[Evaluator] Test: {len(test_ds)} samples")
        return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
