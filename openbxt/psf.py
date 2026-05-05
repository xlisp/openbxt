"""PSF synthesis: Moffat, Gaussian, and Zernike-based aberrated PSFs.

All kernels are returned as torch tensors normalized to sum=1.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _grid(size: int, device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    c = (size - 1) / 2.0
    y = torch.arange(size, device=device, dtype=dtype) - c
    x = torch.arange(size, device=device, dtype=dtype) - c
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return xx, yy


def gaussian_psf(size: int, sigma: float, device=None) -> torch.Tensor:
    xx, yy = _grid(size, device)
    k = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def moffat_psf(size: int, fwhm: float, beta: float = 3.5, device=None) -> torch.Tensor:
    """Moffat profile, a better star model than a Gaussian.

    fwhm: full-width-half-max in pixels
    beta: shape parameter (typical 2.5-4.0 for ground-based seeing)
    """
    alpha = fwhm / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))
    xx, yy = _grid(size, device)
    r2 = xx ** 2 + yy ** 2
    k = (1.0 + r2 / (alpha ** 2)) ** (-beta)
    return k / k.sum()


def motion_psf(size: int, length: float, angle_rad: float, device=None) -> torch.Tensor:
    """Linear motion blur (guiding error)."""
    xx, yy = _grid(size, device)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    along = xx * cos_a + yy * sin_a
    perp = -xx * sin_a + yy * cos_a
    k = torch.exp(-perp ** 2 / 0.5) * (along.abs() <= length / 2).float()
    s = k.sum()
    if s < 1e-8:
        return gaussian_psf(size, 0.6, device)
    return k / s


# ---------------------------------------------------------------------------
# Zernike polynomials over the unit disk (Noll indexing 1..15 covered).
# ---------------------------------------------------------------------------

def zernike(j: int, rho: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Noll-indexed Zernike polynomials (low orders)."""
    if j == 1:  # piston
        return torch.ones_like(rho)
    if j == 2:  # tip
        return 2 * rho * torch.cos(theta)
    if j == 3:  # tilt
        return 2 * rho * torch.sin(theta)
    if j == 4:  # defocus
        return math.sqrt(3) * (2 * rho ** 2 - 1)
    if j == 5:  # oblique astigmatism
        return math.sqrt(6) * rho ** 2 * torch.sin(2 * theta)
    if j == 6:  # vertical astigmatism
        return math.sqrt(6) * rho ** 2 * torch.cos(2 * theta)
    if j == 7:  # vertical coma
        return math.sqrt(8) * (3 * rho ** 3 - 2 * rho) * torch.sin(theta)
    if j == 8:  # horizontal coma
        return math.sqrt(8) * (3 * rho ** 3 - 2 * rho) * torch.cos(theta)
    if j == 9:  # vertical trefoil
        return math.sqrt(8) * rho ** 3 * torch.sin(3 * theta)
    if j == 10:  # oblique trefoil
        return math.sqrt(8) * rho ** 3 * torch.cos(3 * theta)
    if j == 11:  # primary spherical
        return math.sqrt(5) * (6 * rho ** 4 - 6 * rho ** 2 + 1)
    raise ValueError(f"Zernike index {j} not implemented")


def aberrated_psf(
    size: int,
    coeffs: dict[int, float],
    pupil_diameter: float = 0.95,
    seeing_fwhm: float = 1.6,
    device=None,
) -> torch.Tensor:
    """Build a PSF from Zernike wavefront aberrations + atmospheric seeing.

    coeffs: mapping {Noll index -> rms wavefront amplitude in waves}.
            Useful keys: 4 (defocus), 5/6 (astig), 7/8 (coma), 9/10 (trefoil), 11 (sph)
    """
    # Pupil plane sampling
    N = size * 2  # zero-pad for finer angular sampling
    xx, yy = _grid(N, device)
    rr = torch.sqrt(xx ** 2 + yy ** 2) / (N / 2 * pupil_diameter)
    theta = torch.atan2(yy, xx)
    pupil = (rr <= 1.0).to(torch.float32)

    phase = torch.zeros_like(rr)
    for j, c in coeffs.items():
        if c == 0:
            continue
        phase = phase + c * zernike(j, rr.clamp(max=1.0), theta) * pupil

    field = pupil * torch.exp(2j * math.pi * phase.to(torch.float32))
    psf = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field))).abs() ** 2

    # crop to requested size
    s = (N - size) // 2
    psf = psf[s : s + size, s : s + size]

    # convolve with seeing (Gaussian) — small kernel
    sigma_seeing = seeing_fwhm / 2.355
    if sigma_seeing > 0.1:
        gk_size = max(5, int(6 * sigma_seeing) | 1)
        gk = gaussian_psf(min(gk_size, size), sigma_seeing, device)
        psf = F.conv2d(
            psf[None, None],
            gk[None, None],
            padding=gk.shape[-1] // 2,
        )[0, 0]

    psf = psf.clamp(min=0)
    s = psf.sum()
    if s < 1e-8:
        return gaussian_psf(size, max(sigma_seeing, 0.8), device)
    return psf / s


def random_aberrated_psf(
    size: int = 33,
    seeing_range: tuple[float, float] = (1.0, 3.5),
    aberration_strength: float = 0.25,
    device=None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a random realistic PSF for training augmentation."""
    g = generator
    seeing = float(torch.empty(1).uniform_(*seeing_range, generator=g))
    coeffs = {}
    for j in (4, 5, 6, 7, 8, 9, 10, 11):
        coeffs[j] = float(torch.empty(1).normal_(0.0, aberration_strength, generator=g))
    return aberrated_psf(size, coeffs, seeing_fwhm=seeing, device=device)
