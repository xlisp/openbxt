"""Tiled inference for arbitrarily large astronomical images.

The network is trained on small crops, but real frames can be 6k x 4k or
larger. We tile, run the network on each tile with overlap, blend with a
cosine window, and return the result.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .star_detect import detect_stars, estimate_psf
from .model import OpenBXT


def _cosine_window(size: int, overlap: int, device, dtype) -> torch.Tensor:
    """1-D blending window: 1 in the central plateau, cosine ramps at edges."""
    w = torch.ones(size, device=device, dtype=dtype)
    if overlap > 0:
        ramp = 0.5 - 0.5 * torch.cos(
            torch.linspace(0, math.pi, overlap, device=device, dtype=dtype)
        )
        w[:overlap] = ramp
        w[-overlap:] = ramp.flip(0)
    return w


@torch.no_grad()
def deconvolve(
    model: OpenBXT,
    image: torch.Tensor,
    psf: torch.Tensor | None = None,
    tile: int = 256,
    overlap: int = 32,
    sharpen_nonstellar: float = 0.9,
    sharpen_stellar: float = 0.5,
    auto_psf: bool = True,
    device: str | torch.device | None = None,
) -> dict:
    """Run OpenBXT on a full-size image.

    image: (C, H, W) torch tensor, linear values roughly in [0, 1].
    psf:   (K, K) tensor; if None and auto_psf, the PSF is estimated from
           detected stars in the input image.
    Returns dict {sharp, residual, star_map, psf_used}.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    image = image.to(device).float()
    C, H, W = image.shape

    if psf is None and auto_psf:
        coords = detect_stars(image, fwhm=3.0, threshold_sigma=5.0, max_stars=200)
        psf = estimate_psf(image, coords, psf_size=model.psf_encoder.psf_size)
    if psf is None:
        psf = torch.zeros(
            (model.psf_encoder.psf_size, model.psf_encoder.psf_size),
            device=device,
        )
    psf = psf.to(device).float()

    stride = tile - overlap
    n_y = max(1, math.ceil((H - overlap) / stride))
    n_x = max(1, math.ceil((W - overlap) / stride))

    accum = torch.zeros((C, H, W), device=device)
    accum_star = torch.zeros((1, H, W), device=device)
    weight = torch.zeros((1, H, W), device=device)

    win_y = _cosine_window(tile, overlap, device, image.dtype)
    win_x = _cosine_window(tile, overlap, device, image.dtype)
    win = (win_y[:, None] * win_x[None, :])[None]  # (1, tile, tile)

    for iy in range(n_y):
        for ix in range(n_x):
            y0 = min(iy * stride, H - tile) if H >= tile else 0
            x0 = min(ix * stride, W - tile) if W >= tile else 0
            th = min(tile, H)
            tw = min(tile, W)
            patch = image[:, y0 : y0 + th, x0 : x0 + tw]

            # Pad to (tile, tile) if at edges
            pad_h = tile - patch.shape[-2]
            pad_w = tile - patch.shape[-1]
            if pad_h or pad_w:
                patch = F.pad(patch, (0, pad_w, 0, pad_h), mode="reflect")

            out = model(
                patch[None],
                psf=psf[None],
                stellar_strength=sharpen_stellar,
                nonstellar_strength=sharpen_nonstellar,
            )
            sharp = out["sharp"][0]
            star_map = out["star_map"][0]

            # Crop back to actual size
            sharp = sharp[:, :th, :tw]
            star_map = star_map[:, :th, :tw]
            w_local = win[:, :th, :tw]

            accum[:, y0 : y0 + th, x0 : x0 + tw] += sharp * w_local
            accum_star[:, y0 : y0 + th, x0 : x0 + tw] += star_map * w_local
            weight[:, y0 : y0 + th, x0 : x0 + tw] += w_local

    weight = weight.clamp(min=1e-6)
    sharp_full = accum / weight
    star_full = accum_star / weight

    return {
        "sharp": sharp_full,
        "residual": sharp_full - image,
        "star_map": star_full,
        "psf_used": psf,
    }
