"""
dataset.py
----------
Custom PyTorch Dataset for loading paired SAR and EO satellite imagery.

Supports three EO band configurations:
  - 'rgb'               : Red, Green, Blue (B4, B3, B2)
  - 'nir_swir_red_edge' : NIR, SWIR, Red Edge (B8, B11, B5)
  - 'rgb_nir'           : Red, Green, Blue, NIR (B4, B3, B2, B8)
"""

import os
import glob

import torch
import rasterio
from torch.utils.data import Dataset


class SARToEODataset(Dataset):
    """
    A custom PyTorch Dataset for loading paired SAR and EO satellite imagery.

    Args:
        sar_dir (str): Directory containing SAR .tif images.
        eo_dir (str): Directory containing EO .tif images.
        transform (callable, optional): Transform applied to both SAR and EO images.
        band_config (str): EO band selection. One of 'rgb', 'rgb_nir', 'nir_swir_red_edge'.
        normalize (str): Normalization strategy — 'dynamic', 'clip', or 'none'.
        clip_range (tuple): (min, max) for 'clip' normalization mode.
    """

    BAND_CONFIG_MAP = {
        'rgb':               ['red', 'green', 'blue'],
        'rgb_nir':           ['red', 'green', 'blue', 'nir'],
        'nir_swir_red_edge': ['nir', 'swir', 'red edge'],
    }

    def __init__(
        self,
        sar_dir: str,
        eo_dir: str,
        transform=None,
        band_config: str = 'rgb',
        normalize: str = 'dynamic',
        clip_range: tuple = (0, 1),
    ):
        self.sar_dir = sar_dir
        self.eo_dir = eo_dir
        self.transform = transform
        self.band_config = band_config.lower()
        self.normalize = normalize
        self.clip_range = clip_range

        self.sar_files = sorted(glob.glob(os.path.join(sar_dir, '**', '*.tif'), recursive=True))
        self.eo_files  = sorted(glob.glob(os.path.join(eo_dir,  '**', '*.tif'), recursive=True))

        print(f"[Dataset] SAR files: {len(self.sar_files)}, EO files: {len(self.eo_files)}")
        assert len(self.sar_files) == len(self.eo_files), \
            f"Mismatch between SAR ({len(self.sar_files)}) and EO ({len(self.eo_files)}) file counts."

        self.paired_files = list(zip(self.sar_files, self.eo_files))

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx: int):
        sar_path, eo_path = self.paired_files[idx]

        sar_img = self._load_sar_with_ratio_channel(sar_path)
        eo_img  = self._load_raster(eo_path, band_config=self.band_config)

        sar_img = self._normalize_image(sar_img)
        eo_img  = self._normalize_image(eo_img)

        if self.transform:
            sar_img = self.transform(sar_img)
            eo_img  = self.transform(eo_img)

        return sar_img, eo_img

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_raster(self, path: str, band_config: str = None) -> torch.Tensor:
        """Read a multi-band .tif file, selecting bands by config name."""
        with rasterio.open(path) as src:
            if band_config:
                indices = self._select_bands_by_names(src, band_config)
                bands = src.read(indices, out_dtype='float32') if indices else src.read(out_dtype='float32')
            else:
                bands = src.read(out_dtype='float32')
        return torch.from_numpy(bands)

    def _load_sar_with_ratio_channel(self, path: str) -> torch.Tensor:
        """
        Load a SAR .tif file and compute a 3-channel tensor:
          [VV, VH, VV / (VH + ε)]
        """
        with rasterio.open(path) as src:
            sar = src.read(out_dtype='float32')

        vv    = torch.from_numpy(sar[0])
        vh    = torch.from_numpy(sar[1])
        ratio = vv / (vh + 1e-6)

        return torch.stack([vv, vh, ratio], dim=0)  # [3, H, W]

    def _select_bands_by_names(self, src, config: str) -> list:
        """
        Map a band_config string to rasterio 1-indexed band numbers,
        using band description metadata where available.
        Falls back to hardcoded indices for 'rgb'.
        """
        desired_names = self.BAND_CONFIG_MAP.get(config, None)
        if not desired_names:
            return []

        band_indices = []
        for idx in range(1, src.count + 1):
            raw_desc = src.descriptions[idx - 1]
            desc = raw_desc.lower() if raw_desc else ""
            if any(name in desc for name in desired_names):
                band_indices.append(idx)

        # Fallback for RGB if metadata is missing
        if not band_indices and config == 'rgb':
            band_indices = [3, 2, 1]  # B4, B3, B2

        return band_indices

    def _normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Normalize image tensor to [-1, 1].

        Modes:
          'dynamic' : per-image min-max scaling
          'clip'    : clip to clip_range then scale
          'none'    : no normalization
        """
        if self.normalize == 'dynamic':
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = torch.zeros_like(img)
        elif self.normalize == 'clip':
            lo, hi = self.clip_range
            img = torch.clamp(img, min=lo, max=hi)
            img = (img - lo) / (hi - lo)
        elif self.normalize == 'none':
            return img
        else:
            raise ValueError(f"Unknown normalization mode: '{self.normalize}'. "
                             "Choose from 'dynamic', 'clip', 'none'.")
        return img * 2.0 - 1.0  # Scale to [-1, 1]
