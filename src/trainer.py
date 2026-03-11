"""
trainer.py
----------
Trainer class for the SAR-to-EO CycleGAN model.

Handles:
  - Model and optimizer setup from a config dict
  - DataLoader construction
  - Full training loop with validation and early stopping
  - Checkpoint saving (every epoch + best model)
  - Learning rate scheduling (linear decay after half-way point)
"""

import os
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from tqdm import tqdm
from pytorch_msssim import ms_ssim

from src.dataset import SARToEODataset
from src.models import Generator, Discriminator
from src.losses import CharbonnierLoss, build_vgg_feature_extractor, perceptual_loss, ms_ssim_loss
from src.utils import init_weights, compute_psnr, save_sample, denormalize


class Trainer:
    """
    End-to-end trainer for the SAR-to-EO CycleGAN.

    Args:
        config (dict): Configuration dictionary, typically loaded from a YAML file.
                       See configs/config_part_a.yaml for all required keys.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Trainer] Using device: {self.device}")

        # Determine output channels from band_config
        band_config = config['band_config']
        out_channels = 4 if band_config == 'rgb_nir' else 3

        # --- Directories ---
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        self.output_dir     = config.get('output_dir',     './outputs')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # --- Models ---
        self.G = Generator(in_channels=3, out_channels=out_channels).to(self.device)
        self.D = Discriminator(in_channels=out_channels).to(self.device)
        init_weights(self.G)
        init_weights(self.D)

        # --- Loss Functions ---
        self.criterion_GAN = nn.MSELoss()
        self.criterion_L1  = CharbonnierLoss()
        self.vgg_net       = build_vgg_feature_extractor(self.device)

        # --- Loss Weights ---
        self.lam_gan    = config.get('lambda_gan',    1.0)
        self.lam_l1     = config.get('lambda_l1',    50.0)
        self.lam_msssim = config.get('lambda_msssim', 0.5)
        self.lam_perc   = config.get('lambda_perc',   0.05)

        # --- Optimizers ---
        lr = config.get('lr', 2e-4)
        self.optimizer_G = AdamW(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = AdamW(self.D.parameters(), lr=lr, betas=(0.5, 0.999))

        # --- Schedulers ---
        num_epochs     = config.get('num_epochs', 10)
        self.num_epochs = num_epochs
        lr_lambda      = lambda epoch: 1.0 - max(0, epoch - num_epochs // 2) / float(num_epochs // 2)
        self.scheduler_G = LambdaLR(self.optimizer_G, lr_lambda=lr_lambda)
        self.scheduler_D = LambdaLR(self.optimizer_D, lr_lambda=lr_lambda)

        # --- DataLoaders ---
        self.train_loader, self.val_loader = self._build_dataloaders(config)

        # --- Tracking ---
        self.g_losses   = []
        self.d_losses   = []
        self.psnr_list  = []
        self.val_psnrs  = []
        self.val_ssims  = []
        self.val_l1s    = []
        self.val_total_losses = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop with validation and early stopping."""
        patience      = self.cfg.get('patience', 7)
        counter       = 0
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            # ----- Training -----
            self.G.train()
            self.D.train()
            self._train_epoch(epoch)

            # ----- Validation -----
            val_psnr, val_ssim_val, val_l1, val_total = self._validate()
            self.val_psnrs.append(val_psnr)
            self.val_ssims.append(val_ssim_val)
            self.val_l1s.append(val_l1)
            self.val_total_losses.append(val_total)

            print(
                f"Epoch {epoch+1}/{self.num_epochs} — "
                f"PSNR: {val_psnr:.2f} | SSIM: {val_ssim_val:.4f} | "
                f"L1: {val_l1:.4f} | Val Loss: {val_total:.4f}"
            )

            # ----- Early Stopping + Best Checkpoint -----
            if val_total < best_val_loss:
                best_val_loss = val_total
                torch.save(self.G.state_dict(), os.path.join(self.checkpoint_dir, 'G_best.pt'))
                print(f"  Best model saved (epoch {epoch+1}, val_loss={val_total:.4f})")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"  Early stopping triggered at epoch {epoch+1}")
                    break

            # ----- Per-Epoch Checkpoints -----
            torch.save(self.G.state_dict(), os.path.join(self.checkpoint_dir, f'G_epoch{epoch}.pt'))
            torch.save(self.D.state_dict(), os.path.join(self.checkpoint_dir, f'D_epoch{epoch}.pt'))

            # ----- LR Schedules -----
            self.scheduler_G.step()
            self.scheduler_D.step()

        print("Training complete.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> None:
        """Single training epoch: update D then G for each batch."""
        loop = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
        for i, batch in enumerate(loop):
            real_A = batch[0].to(self.device)
            real_B = batch[1].to(self.device)

            # ---- Train Discriminator ----
            self.optimizer_D.zero_grad()
            fake_B    = self.G(real_A).detach()
            pred_real = self.D(real_B)
            pred_fake = self.D(fake_B)

            loss_D_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))
            loss_D_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D      = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            self.optimizer_D.step()

            # ---- Train Generator ----
            self.optimizer_G.zero_grad()
            fake_B    = self.G(real_A)
            pred_fake = self.D(fake_B)

            # Ensure inputs are in [0, 1] for MS-SSIM
            fake_B_01 = denormalize(fake_B)
            real_B_01 = denormalize(real_B)

            loss_G_GAN  = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake)) * self.lam_gan
            loss_G_L1   = self.criterion_L1(fake_B, real_B) * self.lam_l1
            loss_G_SSIM = ms_ssim_loss(fake_B_01, real_B_01) * self.lam_msssim
            loss_G_PERC = perceptual_loss(fake_B[:, :3], real_B[:, :3], self.vgg_net) * self.lam_perc

            loss_G = loss_G_GAN + loss_G_L1 + loss_G_SSIM + loss_G_PERC
            loss_G.backward()
            self.optimizer_G.step()

            # ---- Logging ----
            self.d_losses.append(loss_D.item())
            self.g_losses.append(loss_G.item())
            self.psnr_list.append(compute_psnr(fake_B, real_B))

            loop.set_postfix(G=f'{loss_G.item():.3f}', D=f'{loss_D.item():.3f}')

            if i % 100 == 0:
                save_sample(
                    real_A[:1], fake_B[:1], real_B[:1],
                    os.path.join(self.output_dir, f'sample_e{epoch}_i{i}.png')
                )

    def _validate(self):
        """Run validation loop. Returns (psnr, ssim, l1, total_loss)."""
        self.G.eval()
        total_psnr = total_ssim = total_l1 = 0.0

        with torch.no_grad():
            for val_batch in self.val_loader:
                val_A  = val_batch[0].to(self.device)
                val_B  = val_batch[1].to(self.device)
                fake_B = self.G(val_A)

                fake_01 = denormalize(fake_B)
                real_01 = denormalize(val_B)

                total_psnr += compute_psnr(fake_B, val_B)
                total_ssim += (1 - ms_ssim(fake_01, real_01, data_range=1.0, size_average=True)).item()
                total_l1   += self.criterion_L1(fake_B, val_B).item()

        n = len(self.val_loader)
        val_psnr = total_psnr / n
        val_ssim = total_ssim / n
        val_l1   = total_l1   / n
        val_loss = self.lam_l1 * val_l1 + self.lam_msssim * val_ssim
        return val_psnr, val_ssim, val_l1, val_loss

    def _build_dataloaders(self, config: dict):
        """Build train and val DataLoaders from config."""
        data_root  = config['data_root']
        band_cfg   = config['band_config']
        batch_size = config.get('batch_size', 8)
        n_train    = config.get('train_samples', 2000)
        n_val      = config.get('val_samples',   400)

        train_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        val_tf = transforms.Resize((256, 256))

        train_full = SARToEODataset(
            sar_dir=os.path.join(data_root, 'train', 'SAR'),
            eo_dir=os.path.join(data_root,  'train', 'EO'),
            transform=train_tf, band_config=band_cfg,
        )
        val_full = SARToEODataset(
            sar_dir=os.path.join(data_root, 'val', 'SAR'),
            eo_dir=os.path.join(data_root,  'val', 'EO'),
            transform=val_tf, band_config=band_cfg,
        )

        train_ds = Subset(train_full, list(range(min(n_train, len(train_full)))))
        val_ds   = Subset(val_full,   list(range(min(n_val,   len(val_full)))))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        print(f"[Trainer] Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
        return train_loader, val_loader
