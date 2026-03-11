# System Design — SAR-to-EO Image Translation using Lightweight CycleGAN + CBAM

> **Project:** SAR-to-EO Image Translation
> **Authors:** Roopanshu Gupta, Riddhima Bhargava — DTU
> **Last Updated:** March 2026

---

## Table of Contents

1. [Project Goals](#1-project-goals)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Module Responsibilities](#3-module-responsibilities)
4. [Data Flow](#4-data-flow)
5. [Model Architecture](#5-model-architecture)
6. [Loss Engine](#6-loss-engine)
7. [Configuration System](#7-configuration-system)
8. [Experiment Tracking](#8-experiment-tracking)
9. [Limitations and Future Work](#9-limitations-and-future-work)

---

## 1. Project Goals

### Primary Goal

Translate SAR (Synthetic Aperture Radar) satellite imagery into realistic EO (Electro-Optical) imagery using a lightweight deep learning model, enabling optical-quality analysis in cloud-covered or low-light conditions.

### Success Criteria

| Metric | Target |
|---|---|
| PSNR on RGB test set | >= 18 dB |
| SSIM on RGB test set | >= 0.55 |
| NDVI consistency (RGB+NIR) | Low |NDVI_pred - NDVI_real| |
| Training time | Reasonable within Kaggle GPU session |
| Generator parameters | Fewer than standard ResNet-9 CycleGAN |

### Non-Goals

- Real-time inference (not required for satellite imagery)
- Reverse translation (EO to SAR) — not implemented in v1
- Mobile or edge deployment

---

## 2. High-Level Architecture

```
+---------------------------------------------------------------+
|                        CLI Entry Points                       |
|            train.py                 evaluate.py               |
+---------------------------------------+-----------------------+
                                        |
                              YAML Config File
                       configs/config_part_{a,b,c}.yaml
                                        |
               +------------------------+------------------------+
               |                                                 |
     +---------v---------+                          +------------v----------+
     |   src/trainer.py  |                          |   src/evaluate.py     |
     |   Trainer class   |                          |   Evaluator class     |
     +--+--------+-------+                          +---------+------+------+
        |        |                                            |      |
   +----v---+ +--v----------+                        +--------v-+ +--v------+
   | Dataset| | G + D       |                        | Dataset  | | Metrics |
   | Loader | | Models      |                        | Loader   | | PSNR    |
   |        | |             |                        | (test)   | | SSIM    |
   +----+---+ +------+------+                        +----------+ | NDVI    |
        |            |                                            +---------+
      .tif    +------+------+
      files   | Loss Engine |
              | Charbonnier |
              | MS-SSIM     |
              | Perceptual  |
              | GAN (MSE)   |
              +-------------+
```

---

## 3. Module Responsibilities

### `src/dataset.py` — SARToEODataset

Reads, preprocesses, and serves paired SAR/EO `.tif` image samples.

| Concern | Implementation |
|---|---|
| File discovery | `glob` over `.tif` files in nested directories |
| SAR loading | Read VV, VH; compute ratio `VV / (VH + 1e-6)` |
| EO loading | Select bands by rasterio metadata description |
| Band selection fallback | Hardcoded indices for `rgb` if metadata absent |
| Normalization | Per-image dynamic, clip-range, or none — scaled to `[-1, 1]` |
| Augmentation | Configurable via `torchvision.transforms` |

---

### `src/models.py` — Generator and Discriminator

**Generator design goal**: Fewer parameters than ResNet-9 CycleGAN while maintaining competitive perceptual quality.

| Component | Details |
|---|---|
| Input projection | Conv 3x3, 3 -> 32 ch |
| Downsampling | 2x DepthwiseSeparableConv, stride=2 (32->64->128) |
| Bottleneck | 4x InvertedResidualBlock (MobileNetV2, expansion=6) |
| Attention | CBAMBlock (ChannelAttention + SpatialAttention) after bottleneck |
| Upsampling | 2x Bilinear upsample + DepthwiseSeparableConv (128->64->32) |
| Output | Conv 3x3 + Tanh, 32 -> out_channels (3 or 4) |

**Discriminator**: PatchGAN with spectral normalization for local spatial realism.

| Layer | Out channels | Stride | Norm |
|---|---|---|---|
| Block 1 | 32 | 2 | None |
| Block 2 | 64 | 2 | BatchNorm |
| Block 3 | 128 | 2 | BatchNorm |
| Output | 1 | conv4 | SpectralNorm |

---

### `src/losses.py` — Loss Engine

| Loss | Type | Input Range | Purpose |
|---|---|---|---|
| GAN | MSE (LSGAN) | any | Adversarial signal |
| Charbonnier | Smooth L1 | any | Pixel-level accuracy |
| MS-SSIM | 1 - ms_ssim | [0, 1] | Structural consistency |
| Perceptual | VGG16 L1 | [-1, 1] | Texture/feature fidelity |

For 4-channel (RGB+NIR) output, perceptual loss is computed on the first 3 channels only — VGG16 requires exactly 3-channel input.

---

### `src/utils.py` — Shared Utilities

| Function | Purpose |
|---|---|
| `init_weights` | Normal(0, 0.02) GAN weight init |
| `denormalize` | `[-1, 1] -> [0, 1]` clamped |
| `compute_psnr` | Peak Signal-to-Noise Ratio |
| `compute_multiband_ssim` | SSIM across all channels |
| `compute_ndvi` | `(NIR - Red) / (NIR + Red + eps)` |
| `save_sample` | VV / VH / Generated / Real grid |
| `show_sample` | Matplotlib inline display |
| `save_side_by_side_images` | Per-sample triplet grids |
| `zip_checkpoints` | Pack checkpoints for download |

---

### `src/trainer.py` — Trainer

Training loop pseudocode:

```
for epoch in range(num_epochs):

    # --- Training ---
    for batch in train_loader:

        # Discriminator step
        fake_B = G(real_A).detach()
        loss_D = 0.5 * (MSE(D(real_B), 1) + MSE(D(fake_B), 0))
        optimizer_D.step()

        # Generator step
        fake_B = G(real_A)
        loss_G = lam_gan  * GAN_loss(D(fake_B), 1)
               + lam_l1   * Charbonnier(fake_B, real_B)
               + lam_ssim * (1 - MS_SSIM(denorm(fake_B), denorm(real_B)))
               + lam_perc * L1(VGG(fake_B[:, :3]), VGG(real_B[:, :3]))
        optimizer_G.step()

    # --- Validation ---
    val_loss = lam_l1 * val_l1 + lam_ssim * val_ssim
    if val_loss < best_val_loss:
        save G_best.pt
    elif patience exceeded:
        break

    # --- LR Decay ---
    scheduler_G.step(); scheduler_D.step()
```

LR schedule: linear decay from `num_epochs // 2` to final epoch.  
Early stopping: monitors combined validation loss, configurable patience (default 7).

---

### `src/evaluate.py` — Evaluator

```
Load G from checkpoints/G_best.pt

for batch in test_loader:
    fake_B = G(real_A)
    for each sample:
        accumulate PSNR(fake, gt)
        accumulate SSIM(fake, gt)
        if band_config == 'rgb_nir':
            accumulate |NDVI(fake) - NDVI(gt)|

Report: avg_psnr, avg_ssim, [avg_ndvi_diff]
```

---

## 4. Data Flow

```
Raw .tif (SAR)
  |-- VV band, VH band
  |-- ratio = VV / (VH + eps)
  --> Tensor [3, H, W]
  --> Normalize to [-1, 1]
  --> Resize to 256x256
  --> [Train only] RandomHorizontalFlip

Raw .tif (EO)
  |-- Select bands by band_config (via rasterio metadata)
  --> Tensor [C, H, W]  (C = 3 or 4)
  --> Normalize to [-1, 1]
  --> Resize to 256x256

Training:
  (SAR, EO) -> Trainer -> G, D updated per batch
  -> Checkpoint saved per epoch (G_epoch{n}.pt, D_epoch{n}.pt)
  -> Best model saved (G_best.pt)

Inference:
  SAR -> G (G_best.pt) -> Generated EO
  -> PSNR, SSIM, [NDVI diff] reported
  -> Triplet grids saved to outputs/eval_grids/
```

---

## 5. Model Architecture

### Generator — Expanded View

```
Input SAR [ 3 x 256 x 256 ]
  |
  Conv2d 3->32 + BN + ReLU6
  |
  DepthwiseSeparableConv 32->64 (stride=2)          [128x128]
  DepthwiseSeparableConv 64->128 (stride=2)          [ 64x64]
  |
  InvertedResidualBlock (128ch) x4                   [ 64x64]
  |
  CBAMBlock: ChannelAttention -> SpatialAttention
  |
  Upsample(2x) + DepthwiseSeparableConv 128->64      [128x128]
  Upsample(2x) + DepthwiseSeparableConv 64->32       [256x256]
  |
  Conv2d 32->out_channels + Tanh
  |
Generated EO [ out_ch x 256 x 256 ]
  (out_ch = 3 for rgb / nir_swir_red_edge, 4 for rgb_nir)
```

### CBAM Attention — Detail

```
  Input x [C, H, W]
      |
  ChannelAttention:
    avg_pool(x) + max_pool(x)
    -> shared MLP (C -> C//8 -> C)
    -> sigmoid
    -> scale x
      |
  SpatialAttention:
    channel-wise mean + max -> [2, H, W]
    -> Conv2d(2, 1, 7x7) + sigmoid
    -> scale x
      |
  Output [C, H, W]
```

---

## 6. Loss Engine

Generator total loss:

```
L_G = lam_gan  * MSE(D(fake_B), 1)              [GAN: realism]
    + lam_l1   * Charbonnier(fake_B, real_B)     [pixel: sharpness]
    + lam_ssim * (1 - MS_SSIM(fake_01, real_01)) [structure: SSIM]
    + lam_perc * L1(VGG(fake[:3]), VGG(real[:3]))[texture: features]
```

Default weights: `lam_gan=1.0, lam_l1=50.0, lam_ssim=0.5, lam_perc=0.05`

Discriminator loss (LSGAN):

```
L_D = 0.5 * (MSE(D(real_B), 1) + MSE(D(fake_B), 0))
```

---

## 7. Configuration System

All hyperparameters are externalized to YAML files. The three experiments differ only in band-level settings:

| Parameter | Part A | Part B | Part C |
|---|---|---|---|
| `band_config` | `rgb` | `nir_swir_red_edge` | `rgb_nir` |
| `out_channels` | 3 | 3 | 4 |
| `train_samples` | 2000 | 800 | 800 |
| `val_samples` | 400 | 200 | 200 |
| `test_samples` | 400 | 200 | 200 |

Shared parameters (`lr`, `batch_size`, `num_epochs`, `patience`, `lambda_*`) are the same across all three configs.

CLI overrides are supported:
```bash
python train.py --config configs/config_part_a.yaml --num_epochs 20 --lr 1e-4
```

---

## 8. Experiment Tracking

Per-epoch metrics stored on the `Trainer` object after training, available for plotting:

| Attribute | Content |
|---|---|
| `trainer.g_losses` | Generator loss per training batch |
| `trainer.d_losses` | Discriminator loss per training batch |
| `trainer.psnr_list` | PSNR per training batch |
| `trainer.val_psnrs` | Validation PSNR per epoch |
| `trainer.val_ssims` | Validation SSIM per epoch |
| `trainer.val_l1s` | Validation Charbonnier L1 per epoch |
| `trainer.val_total_losses` | Combined validation loss per epoch |

Sample visualizations are saved every 100 training iterations to `output_dir/`.

---

## 9. Limitations and Future Work

| Limitation | Potential Improvement |
|---|---|
| No cycle-consistency loss | Add reverse G: EO -> SAR, enforce consistency loss |
| Single-direction translation | Extend to fully bidirectional SAR <-> EO |
| Static loss weights | Adaptive weighting or uncertainty-based weighting |
| No multi-scale discriminator | Add projected or multi-scale discriminator |
| Limited NDVI visualization | Add NDVI side-by-side maps to evaluate.py output |
| No mixed-precision training | Add `torch.cuda.amp` for 2x GPU memory efficiency |
| Single dataset only | Evaluate cross-dataset generalization |

---

*This document should be updated whenever major architectural decisions are made.*
