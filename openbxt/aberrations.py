"""Spatially-varying aberration / blur application.

Real telescope PSFs vary across the field of view: stars in the corners
typically have more coma, astigmatism, and defocus than stars in the center.
We model this by interpolating PSFs between control points on a sparse grid.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .psf import aberrated_psf, gaussian_psf, motion_psf, random_aberrated_psf


def _convolve_per_tile(image: torch.Tensor, kernels: torch.Tensor, tile: int) -> torch.Tensor:
    """Apply a different PSF kernel to each tile of the image.

    image:   (C, H, W)
    kernels: (Gy, Gx, K, K)  — kernel per tile (Gy x Gx grid)
    tile:    tile size in pixels — H and W must be divisible by tile

    Returns the convolved image, smoothly blended at tile boundaries via
    bilinear weights between the four nearest tile centers.
    """
    C, H, W = image.shape
    Gy, Gx, K, _ = kernels.shape
    pad = K // 2

    # Pre-compute every tile-PSF convolution as a separate full-image conv,
    # then blend with bilinear weights based on per-pixel coords.
    # Cheap because grids are small (~3x3 or 5x5).
    out = torch.zeros_like(image)
    weight_sum = torch.zeros((1, H, W), device=image.device, dtype=image.dtype)

    yy = torch.arange(H, device=image.device, dtype=image.dtype)
    xx = torch.arange(W, device=image.device, dtype=image.dtype)

    # tile center coords (in pixels)
    cy = torch.linspace(tile / 2, H - tile / 2, Gy, device=image.device)
    cx = torch.linspace(tile / 2, W - tile / 2, Gx, device=image.device)

    img_pad = F.pad(image[None], (pad, pad, pad, pad), mode="reflect")
    for gy in range(Gy):
        for gx in range(Gx):
            k = kernels[gy, gx][None, None].expand(C, 1, K, K)
            conv = F.conv2d(img_pad, k, groups=C)[0]  # (C, H, W)

            # Bilinear weight peaked at (cy[gy], cx[gx]), width = tile
            wy = (1.0 - (yy - cy[gy]).abs() / tile).clamp(min=0.0)
            wx = (1.0 - (xx - cx[gx]).abs() / tile).clamp(min=0.0)
            w = wy[:, None] * wx[None, :]  # (H, W)

            out = out + conv * w
            weight_sum = weight_sum + w

    return out / weight_sum.clamp(min=1e-6)


def synthesize_blurred(
    sharp: torch.Tensor,
    grid: tuple[int, int] = (3, 3),
    psf_size: int = 33,
    seeing_range: tuple[float, float] = (1.2, 3.5),
    aberration_strength: float = 0.3,
    edge_aberration_boost: float = 1.5,
    motion_prob: float = 0.3,
    chromatic_lateral: float = 0.4,
    chromatic_seeing_jitter: float = 0.15,
    return_meta: bool = False,
    generator: torch.Generator | None = None,
):
    """Apply realistic spatially-varying blur + aberrations to a sharp image.

    sharp: (C, H, W) float tensor in [0, 1] linear units. C in {1, 3}.
    Returns: (blurred, central_psf) or dict if return_meta.
    """
    C, H, W = sharp.shape
    Gy, Gx = grid
    # tile size such that H, W are divisible
    th, tw = H // Gy, W // Gx
    assert th == tw, "Square tiles for now"
    tile = th
    device = sharp.device

    # Build a per-tile PSF: aberrations scale with radius from image center.
    kernels = torch.empty((Gy, Gx, psf_size, psf_size), device=device)
    seeing = float(torch.empty(1).uniform_(*seeing_range, generator=generator))

    use_motion = bool(torch.rand(1, generator=generator).item() < motion_prob)
    motion_len = float(torch.empty(1).uniform_(0.8, 3.0, generator=generator)) if use_motion else 0.0
    motion_ang = float(torch.empty(1).uniform_(0, math.pi, generator=generator)) if use_motion else 0.0

    for gy in range(Gy):
        for gx in range(Gx):
            # normalized radial distance from image center, [0, ~1.41]
            ny = (gy + 0.5) / Gy * 2 - 1
            nx = (gx + 0.5) / Gx * 2 - 1
            r = math.sqrt(ny ** 2 + nx ** 2)

            scale = 1.0 + edge_aberration_boost * r
            coeffs = {}
            for j in (4, 5, 6, 7, 8, 9, 10, 11):
                coeffs[j] = float(
                    torch.empty(1).normal_(0.0, aberration_strength * scale, generator=generator)
                )
            # Coma typically points outward (radial)
            radial_coma = aberration_strength * scale * 0.8
            coeffs[7] += radial_coma * ny
            coeffs[8] += radial_coma * nx

            psf = aberrated_psf(psf_size, coeffs, seeing_fwhm=seeing, device=device)

            if use_motion:
                mk = motion_psf(psf_size, motion_len, motion_ang, device=device)
                psf = F.conv2d(
                    psf[None, None], mk[None, None], padding=psf_size // 2
                )[0, 0]
                psf = psf / psf.sum().clamp(min=1e-8)

            kernels[gy, gx] = psf

    # Per-channel chromatic effects
    if C == 3:
        out_channels = []
        for ci in range(C):
            shift_y = float(torch.empty(1).normal_(0, chromatic_lateral, generator=generator))
            shift_x = float(torch.empty(1).normal_(0, chromatic_lateral, generator=generator))
            jitter = float(torch.empty(1).normal_(0, chromatic_seeing_jitter, generator=generator))

            ch = sharp[ci : ci + 1]
            if abs(jitter) > 0.05:
                gk = gaussian_psf(7, max(0.4, 0.6 + jitter), device=device)
                ch = F.conv2d(ch[None], gk[None, None], padding=3)[0]

            blurred_ch = _convolve_per_tile(ch, kernels, tile)
            # Sub-pixel translation for lateral chromatic aberration
            if abs(shift_x) > 0.02 or abs(shift_y) > 0.02:
                blurred_ch = _translate(blurred_ch, shift_y, shift_x)
            out_channels.append(blurred_ch)
        blurred = torch.cat(out_channels, dim=0)
    else:
        blurred = _convolve_per_tile(sharp, kernels, tile)

    # Add Poisson-like + Gaussian read noise
    noise_sigma = float(torch.empty(1).uniform_(0.001, 0.01, generator=generator))
    photon_scale = float(torch.empty(1).uniform_(50.0, 500.0, generator=generator))
    shot = torch.poisson(blurred.clamp(min=0) * photon_scale, generator=generator) / photon_scale
    blurred = shot + torch.randn(blurred.shape, generator=generator, device=device) * noise_sigma
    blurred = blurred.clamp(0.0, 1.5)

    central_psf = kernels[Gy // 2, Gx // 2]

    if return_meta:
        return {
            "blurred": blurred,
            "psf": central_psf,
            "kernels": kernels,
            "seeing": seeing,
            "noise_sigma": noise_sigma,
        }
    return blurred, central_psf


def _translate(image: torch.Tensor, dy: float, dx: float) -> torch.Tensor:
    """Sub-pixel translate via grid_sample."""
    C, H, W = image.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=image.device),
        torch.linspace(-1, 1, W, device=image.device),
        indexing="ij",
    )
    grid = torch.stack([xx + 2 * dx / W, yy + 2 * dy / H], dim=-1)[None]
    return F.grid_sample(image[None], grid, mode="bilinear", padding_mode="reflection", align_corners=True)[0]
