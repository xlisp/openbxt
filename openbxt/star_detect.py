"""Star detection and PSF estimation from a real image.

Used at inference time to (a) build a stellar mask, and (b) estimate an
empirical average PSF that conditions the network.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _laplacian_of_gaussian(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Compute LoG response — peaks correspond to blob centers of size ~sigma."""
    size = max(7, int(6 * sigma) | 1)
    c = (size - 1) / 2
    yy, xx = torch.meshgrid(
        torch.arange(size, device=image.device, dtype=image.dtype) - c,
        torch.arange(size, device=image.device, dtype=image.dtype) - c,
        indexing="ij",
    )
    r2 = xx ** 2 + yy ** 2
    s2 = sigma ** 2
    log = ((r2 - 2 * s2) / (s2 ** 2)) * torch.exp(-r2 / (2 * s2))
    log = log - log.mean()
    log = log / log.abs().sum()

    if image.ndim == 2:
        image = image[None, None]
    elif image.ndim == 3:
        image = image[None]

    pad = size // 2
    return F.conv2d(image, log[None, None].expand(image.shape[1], 1, size, size),
                    padding=pad, groups=image.shape[1])


def detect_stars(
    image: torch.Tensor,
    fwhm: float = 3.0,
    threshold_sigma: float = 5.0,
    max_stars: int = 500,
) -> torch.Tensor:
    """Return star centroid coords as a (N, 2) tensor (y, x).

    image: (C, H, W) or (H, W) — uses luminance if multi-channel.
    """
    if image.ndim == 3:
        lum = image.mean(dim=0)
    else:
        lum = image
    lum = lum - lum.median()

    sigma = fwhm / 2.355
    response = -_laplacian_of_gaussian(lum, sigma)[0, 0]  # negate so stars are positive

    # Robust noise estimate
    sigma_n = response[response.abs() < response.abs().quantile(0.95)].std()
    thr = threshold_sigma * sigma_n

    # Local maxima via max-pool == self
    win = max(3, int(fwhm) | 1)
    pooled = F.max_pool2d(response[None, None], kernel_size=win, stride=1, padding=win // 2)[0, 0]
    peaks = (response == pooled) & (response > thr)

    coords = peaks.nonzero(as_tuple=False)  # (N, 2) yx
    if coords.numel() == 0:
        return coords

    # Sort by brightness desc, keep top max_stars
    vals = response[coords[:, 0], coords[:, 1]]
    order = vals.argsort(descending=True)[:max_stars]
    return coords[order]


def estimate_psf(
    image: torch.Tensor,
    star_coords: torch.Tensor,
    psf_size: int = 33,
    saturate_thresh: float = 0.95,
) -> torch.Tensor:
    """Build an averaged empirical PSF from star cutouts.

    image: (C, H, W) or (H, W). Returns (psf_size, psf_size) tensor.
    """
    if image.ndim == 3:
        lum = image.mean(dim=0)
    else:
        lum = image
    H, W = lum.shape
    half = psf_size // 2
    accum = torch.zeros((psf_size, psf_size), device=image.device, dtype=image.dtype)
    n_used = 0
    for c in star_coords:
        y, x = int(c[0]), int(c[1])
        if y - half < 0 or y + half + 1 > H or x - half < 0 or x + half + 1 > W:
            continue
        cut = lum[y - half : y + half + 1, x - half : x + half + 1]
        peak = cut.max()
        if peak > saturate_thresh:
            continue
        # Background subtract using the cutout corners
        bg = torch.cat([cut[0, :], cut[-1, :], cut[:, 0], cut[:, -1]]).median()
        cut = (cut - bg).clamp(min=0)
        s = cut.sum()
        if s < 1e-6:
            continue
        accum = accum + cut / s
        n_used += 1
    if n_used == 0:
        # Fallback: gaussian
        c = (psf_size - 1) / 2
        yy, xx = torch.meshgrid(
            torch.arange(psf_size, device=image.device) - c,
            torch.arange(psf_size, device=image.device) - c,
            indexing="ij",
        )
        psf = torch.exp(-(xx ** 2 + yy ** 2) / (2 * 1.5 ** 2))
        return psf / psf.sum()
    psf = accum / n_used
    return psf / psf.sum().clamp(min=1e-8)


def stellar_mask(
    image: torch.Tensor,
    star_coords: torch.Tensor,
    radius: float = 4.0,
) -> torch.Tensor:
    """Soft mask (H, W) that is 1 inside star apertures, 0 elsewhere."""
    if image.ndim == 3:
        H, W = image.shape[-2:]
    else:
        H, W = image.shape
    mask = torch.zeros((H, W), device=image.device, dtype=torch.float32)
    if star_coords.numel() == 0:
        return mask
    yy, xx = torch.meshgrid(
        torch.arange(H, device=image.device, dtype=torch.float32),
        torch.arange(W, device=image.device, dtype=torch.float32),
        indexing="ij",
    )
    for c in star_coords:
        y, x = float(c[0]), float(c[1])
        d2 = (yy - y) ** 2 + (xx - x) ** 2
        mask = torch.maximum(mask, torch.exp(-d2 / (2 * radius ** 2)))
    return mask
