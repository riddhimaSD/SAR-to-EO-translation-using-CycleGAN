# SAR-to-EO Image Translation using Lightweight CycleGAN + CBAM Attention

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF)](https://kaggle.com/)

---

## Authors

| Name | Email |
|---|---|
| Roopanshu Gupta | roopanshugupta_se24b03_018@dtu.ac.in |
| Riddhima Bhargava | riddhimabhargava_se24b03_011@dtu.ac.in |

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [What Makes This Project Unique](#what-makes-this-project-unique)
3. [Repository Structure](#repository-structure)
4. [Setup and Installation](#setup-and-installation)
5. [Dataset Preparation](#dataset-preparation)
6. [Usage](#usage)
7. [Architecture](#architecture)
8. [Data Preprocessing](#data-preprocessing)
9. [Loss Functions](#loss-functions)
10. [Results](#results)
11. [Tools and Frameworks](#tools-and-frameworks)

---

## Project Overview

This project translates **SAR (Synthetic Aperture Radar)** images into **EO (Electro-Optical)** images using a custom-built lightweight CycleGAN architecture augmented with CBAM attention. The objective is to generate realistic EO outputs from radar inputs, particularly in remote sensing scenarios where optical data is unavailable due to cloud cover or low-light conditions.

Three EO translation settings are validated:

| Part | Configuration | Bands |
|---|---|---|
| A | SAR to EO (RGB) | B4, B3, B2 |
| B | SAR to EO (NIR-SWIR-RedEdge) | B8, B11, B5 |
| C | SAR to EO (RGB + NIR) | B4, B3, B2, B8 with NDVI integration |

---

## What Makes This Project Unique

**Multiple Spectral Target Variants**
Supports EO image synthesis in RGB, NIR-SWIR-RedEdge, and RGB+NIR formats from a single unified codebase.

**Lightweight Generator Architecture**
Built from scratch using depthwise separable convolutions and inverted residual blocks, ensuring faster training and fewer parameters compared to standard ResNet-based CycleGAN generators.

**CBAM Attention Integration**
Integrates both channel and spatial attention after the bottleneck to prioritize key spectral and spatial features during translation.

**Hybrid Multi-Loss Strategy**
Combines four complementary loss terms:
- Charbonnier loss for pixel-level robustness
- MS-SSIM for structural preservation
- VGG-based perceptual loss for texture consistency
- GAN loss for photorealism

**NDVI Integration (RGB+NIR Variant)**
NDVI maps are derived from predicted Red and NIR bands during evaluation to validate ecological consistency of generated EO outputs.

---

## Repository Structure

```
SAR-to-EO-image-translation-using-CycleGAN/
|
|-- src/                        # Core Python package
|   |-- __init__.py
|   |-- dataset.py              # SARToEODataset — data loading and preprocessing
|   |-- models.py               # CBAM, Generator, Discriminator
|   |-- losses.py               # CharbonnierLoss, perceptual loss, MS-SSIM loss
|   |-- utils.py                # Metrics, visualization, NDVI, checkpoint export
|   |-- trainer.py              # Training loop with validation and early stopping
|   |-- evaluate.py             # Test-set evaluation and NDVI analysis
|
|-- configs/                    # Per-experiment YAML configuration files
|   |-- config_part_a.yaml      # RGB (B4, B3, B2)
|   |-- config_part_b.yaml      # NIR-SWIR-RedEdge (B8, B11, B5)
|   |-- config_part_c.yaml      # RGB+NIR (B4, B3, B2, B8)
|
|-- Generated_EO_Part_A/        # Generated EO samples — RGB variant
|-- Generated_EO_Part_B/        # Generated EO samples — NIR-SWIR-RE variant
|-- Generated_EO_Part_C/        # Generated EO samples — RGB+NIR variant
|-- Plots_Part_A/               # Training loss and PSNR curves — Part A
|-- Plots_Part_B/               # Training loss and PSNR curves — Part B
|-- Plots_Part_C/               # Training loss and PSNR curves — Part C
|
|-- train.py                    # CLI entry-point for training
|-- evaluate.py                 # CLI entry-point for evaluation
|-- setup.sh                    # Environment setup script
|-- requirements.txt            # Python dependencies
|-- Project1-A.ipynb            # Kaggle notebook — RGB variant
|-- Project1-B.ipynb            # Kaggle notebook — NIR-SWIR-RedEdge variant
|-- Project1-C.ipynb            # Kaggle notebook — RGB+NIR variant
|-- system_design.md            # System architecture documentation
|-- README.md
```

---

## Setup and Installation

### Option 1: Automated Setup

```bash
bash setup.sh
```

### Option 2: Manual Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate.bat       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Kaggle (Notebook Environment)

The notebooks install dependencies automatically via:
```python
!pip install rasterio pytorch-msssim
```

---

## Dataset Preparation

Expected folder structure under the dataset root:

```
CycleGAN_dataset/
|-- train/
|   |-- SAR/       # .tif files (VV, VH polarizations)
|   |-- EO/        # .tif files (multispectral bands)
|-- val/
|   |-- SAR/
|   |-- EO/
|-- test/
|   |-- SAR/
|   |-- EO/
```

- File format: `.tif` (GeoTIFF)
- SAR inputs: VV and VH polarization bands (2-band .tif)
- EO targets: multispectral GeoTIFF; bands selected automatically from metadata or fallback indices
- All files must be paired by filename within each split

Update `data_root` in the relevant config file to point to your dataset root before training.

---

## Usage

### Training

```bash
# Part A — RGB
python train.py --config configs/config_part_a.yaml

# Part B — NIR-SWIR-RedEdge
python train.py --config configs/config_part_b.yaml

# Part C — RGB+NIR with NDVI
python train.py --config configs/config_part_c.yaml
```

Override any config parameter at the command line:

```bash
python train.py --config configs/config_part_a.yaml --num_epochs 20 --lr 1e-4 --data_root /path/to/data
```

### Evaluation

```bash
python evaluate.py --config configs/config_part_a.yaml
```

For Part C, NDVI difference (|NDVI_pred - NDVI_real|) is computed and reported automatically.

### Output

- Checkpoints saved to `./checkpoints/G_epoch{n}.pt` and `./checkpoints/G_best.pt`
- Sample grids saved to `./outputs/sample_e{epoch}_i{iter}.png`
- Evaluation grids saved to `./outputs/eval_grids/`

---

## Architecture

### Generator

A lightweight encoder-decoder with integrated CBAM attention:

```
Input (SAR, 3-ch)
  |
  v
Input Projection         (Conv 3x3, 3 -> 32)
  |
Downsample x2            (DepthwiseSeparableConv, stride=2 each)
  |                      32 -> 64 -> 128 channels
  v
Bottleneck               (4x InvertedResidualBlock, MobileNetV2-style)
  |
CBAM Attention           (ChannelAttention + SpatialAttention)
  |
Upsample x2              (Bilinear + DepthwiseSeparableConv)
  |                      128 -> 64 -> 32 channels
  v
Output Projection        (Conv 3x3 -> Tanh)
  |
  v
Output (EO, 3 or 4-ch)
```

### Discriminator

PatchGAN-style CNN with spectral normalization:

```
Input (EO image, 3 or 4-ch)
  |
Block 1: SpectralNorm Conv4x4 (no BN) -> LeakyReLU
Block 2: SpectralNorm Conv4x4 + BN -> LeakyReLU
Block 3: SpectralNorm Conv4x4 + BN -> LeakyReLU
Output:  SpectralNorm Conv4x4 (patch-wise prediction map)
```

---

## Data Preprocessing

### SAR Inputs

1. Read VV and VH polarization channels from `.tif`
2. Compute ratio channel: `VV / (VH + 1e-6)`
3. Stack to 3-channel tensor: `[VV, VH, ratio]`

### EO Outputs

Band selection via `band_config`:

| Config | Bands | Channels |
|---|---|---|
| `rgb` | Red, Green, Blue (B4, B3, B2) | 3 |
| `nir_swir_red_edge` | NIR, SWIR, Red Edge (B8, B11, B5) | 3 |
| `rgb_nir` | Red, Green, Blue, NIR (B4, B3, B2, B8) | 4 |

Bands are selected from rasterio metadata descriptions; falls back to hardcoded indices if metadata is unavailable.

### Normalization

All inputs and targets normalized to `[-1, 1]`:
- `dynamic`: per-image min-max then scale to `[-1, 1]`
- `clip`: clip to a fixed range then scale to `[-1, 1]`
- `none`: no normalization

### Augmentation

- Training: random horizontal flip (p=0.5) after resize to 256x256
- Validation / Test: resize to 256x256 only

---

## Loss Functions

The generator is optimized with a weighted combination of four losses:

| Loss | Weight | Purpose |
|---|---|---|
| GAN Loss (MSE) | 1.0 | Adversarial realism |
| Charbonnier Loss | 50.0 | Pixel-level accuracy (smooth L1) |
| MS-SSIM Loss | 0.5 | Structural similarity |
| Perceptual Loss (VGG16) | 0.05 | Texture and feature consistency |

The discriminator uses standard LSGAN loss (MSE) with spectral normalization for training stability.

**Optimizer**: AdamW (betas = 0.5, 0.999)
**LR Scheduler**: Linear decay from `num_epochs // 2` to end

---

## Results

### Quantitative Metrics (averaged over test set)

| Configuration | PSNR (avg) | SSIM (avg) |
|---|---|---|
| SAR to EO (RGB) | ~19 dB | ~0.60 |
| SAR to EO (RGB+NIR) | ~16.7 dB | ~0.45 |
| SAR to EO (NIR-SWIR-RedEdge) | ~18 dB | ~0.60 |

### Qualitative Observations

- The RGB+NIR variant produces realistic NDVI patterns, indicating biologically plausible EO synthesis.
- CBAM improves detail in spatially complex regions such as urban textures and vegetation edges.
- Charbonnier loss outperforms standard L1 in early training stability and final sharpness.
- MS-SSIM and perceptual losses significantly improve structure and color alignment.
- The lightweight model generalizes well even on small training datasets (800-2000 samples).

### Sample Output Layout

Each saved image grid shows (left to right):

1. VV channel SAR image
2. VH channel SAR image
3. Generated EO image (model output)
4. Real EO image (ground truth)

---

## Tools and Frameworks

| Tool / Library | Version | Purpose |
|---|---|---|
| PyTorch | >= 2.0 | Core deep learning framework |
| Torchvision | >= 0.15 | Transforms, VGG feature extractor |
| pytorch-msssim | >= 1.0 | MS-SSIM and SSIM metrics |
| rasterio | >= 1.3 | Multiband GeoTIFF reading |
| matplotlib | >= 3.7 | Metric curve plotting |
| NumPy | >= 1.24 | Numerical operations |
| tqdm | >= 4.65 | Progress bars |
| PyYAML | >= 6.0 | Config file parsing |
| Kaggle Notebooks | — | GPU execution environment |
