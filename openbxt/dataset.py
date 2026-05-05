"""Synthetic training dataset for OpenBXT.

Reads a folder of *sharp* linear astronomical images and applies random,
realistic blur+aberration on the fly. Returns (blurry, sharp, psf, star_mask)
tuples ready for training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .aberrations import synthesize_blurred
from .utils import load_image, auto_stretch
from .star_detect import detect_stars, stellar_mask


_IMG_SUFFIXES = (".fits", ".fit", ".fts", ".tif", ".tiff", ".png", ".jpg", ".jpeg")


def _list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in _IMG_SUFFIXES])


class SyntheticAstroDataset(Dataset):
    """Each item: dict with 'blurry', 'sharp', 'psf', 'star_mask'."""

    def __init__(
        self,
        root: str | Path,
        crop: int = 256,
        psf_size: int = 33,
        grid: tuple[int, int] = (3, 3),
        n_channels: int = 3,
        seeing_range: tuple[float, float] = (1.2, 3.5),
        aberration_strength: float = 0.3,
        samples_per_image: int = 4,
        seed: int | None = None,
    ):
        self.files = _list_images(root)
        if not self.files:
            raise FileNotFoundError(f"No images under {root}")
        self.crop = crop
        self.psf_size = psf_size
        self.grid = grid
        self.n_channels = n_channels
        self.seeing_range = seeing_range
        self.aberration_strength = aberration_strength
        self.samples_per_image = samples_per_image
        self._base_seed = seed

    def __len__(self) -> int:
        return len(self.files) * self.samples_per_image

    def __getitem__(self, idx: int):
        file_idx = idx // self.samples_per_image
        path = self.files[file_idx]

        rng = torch.Generator()
        if self._base_seed is not None:
            rng.manual_seed(self._base_seed + idx)

        arr = load_image(path)  # (C, H, W) np
        arr, _, _ = auto_stretch(arr)
        arr = np.clip(arr, 0.0, 1.5)

        if arr.shape[0] == 1 and self.n_channels == 3:
            arr = np.repeat(arr, 3, axis=0)
        elif arr.shape[0] >= 3 and self.n_channels == 1:
            arr = arr.mean(axis=0, keepdims=True)
        else:
            arr = arr[: self.n_channels]

        c, h, w = arr.shape
        crop = self.crop
        if h < crop or w < crop:
            # pad reflectively rather than fail
            pad_h = max(0, crop - h)
            pad_w = max(0, crop - w)
            arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            c, h, w = arr.shape

        # Random crop
        ry = int(torch.randint(0, h - crop + 1, (1,), generator=rng).item())
        rx = int(torch.randint(0, w - crop + 1, (1,), generator=rng).item())
        sharp_np = arr[:, ry : ry + crop, rx : rx + crop].copy()
        sharp = torch.from_numpy(sharp_np)

        # Random horizontal/vertical flip
        if torch.rand(1, generator=rng).item() < 0.5:
            sharp = torch.flip(sharp, dims=[-1])
        if torch.rand(1, generator=rng).item() < 0.5:
            sharp = torch.flip(sharp, dims=[-2])

        # Synthesize blur
        meta = synthesize_blurred(
            sharp,
            grid=self.grid,
            psf_size=self.psf_size,
            seeing_range=self.seeing_range,
            aberration_strength=self.aberration_strength,
            return_meta=True,
            generator=rng,
        )
        blurry = meta["blurred"]
        psf = meta["psf"]

        # Build the star mask from the *sharp* image (ground truth stars)
        coords = detect_stars(sharp, fwhm=2.5, threshold_sigma=4.0, max_stars=300)
        mask = stellar_mask(sharp, coords, radius=3.0)[None]

        return {
            "blurry": blurry.float(),
            "sharp": sharp.float(),
            "psf": psf.float(),
            "star_mask": mask.float(),
        }


def collate(batch: Sequence[dict]) -> dict:
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
