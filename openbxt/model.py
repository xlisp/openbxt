"""OpenBXT network architecture.

The network is a U-Net that takes:
    - the input (blurry, linear) image, C in {1, 3}
    - an optional PSF kernel (K x K), encoded as a global conditioning vector

It produces:
    - a residual sharpened image (added to the input, bounded)
    - a stellar probability map (used at inference to apply different
      sharpening to stars vs nebulosity)

Training combines a regression loss against the synthetic sharp target with
a BCE loss against a synthetic stellar mask. Coordinate channels (CoordConv)
let the network condition on field position to handle spatially-varying
aberrations.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _act():
    return nn.SiLU(inplace=True)


class ResBlock(nn.Module):
    def __init__(self, ch: int, cond_dim: int = 0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = _act()
        self.cond = nn.Linear(cond_dim, ch * 2) if cond_dim > 0 else None

    def forward(self, x, c=None):
        h = self.conv1(self.act(self.norm1(x)))
        if self.cond is not None and c is not None:
            scale, shift = self.cond(c).chunk(2, dim=-1)
            h = h * (1 + scale[..., None, None]) + shift[..., None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class PSFEncoder(nn.Module):
    """Encode a (K x K) PSF kernel into a feature vector."""

    def __init__(self, psf_size: int = 33, dim: int = 128):
        super().__init__()
        self.psf_size = psf_size
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), _act(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), _act(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), _act(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, dim),
            _act(),
            nn.Linear(dim, dim),
        )

    def forward(self, psf):
        # psf: (B, K, K) or (B, 1, K, K)
        if psf.ndim == 3:
            psf = psf[:, None]
        # Resize to canonical size if needed
        if psf.shape[-1] != self.psf_size:
            psf = F.interpolate(psf, size=self.psf_size, mode="bilinear", align_corners=False)
        return self.net(psf)


def coord_channels(B: int, H: int, W: int, device, dtype) -> torch.Tensor:
    """CoordConv-style spatial coordinates in [-1, 1]."""
    yy = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    xx = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    rr = torch.sqrt(yy ** 2 + xx ** 2)
    return torch.stack([yy, xx, rr], dim=0).expand(B, 3, H, W).contiguous()


class OpenBXT(nn.Module):
    """U-Net with PSF conditioning, dual residual + star-map heads."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 48,
        psf_size: int = 33,
        cond_dim: int = 128,
        residual_scale: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.residual_scale = residual_scale

        self.psf_encoder = PSFEncoder(psf_size=psf_size, dim=cond_dim)

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # Input: image + 3 coord channels
        self.in_conv = nn.Conv2d(in_channels + 3, c1, 3, padding=1)

        self.enc1 = nn.Sequential(ResBlock(c1, cond_dim), ResBlock(c1, cond_dim))
        self.down1 = Down(c1, c2)
        self.enc2 = nn.Sequential(ResBlock(c2, cond_dim), ResBlock(c2, cond_dim))
        self.down2 = Down(c2, c3)
        self.enc3 = nn.Sequential(ResBlock(c3, cond_dim), ResBlock(c3, cond_dim))
        self.down3 = Down(c3, c4)
        self.bottleneck = nn.Sequential(ResBlock(c4, cond_dim), ResBlock(c4, cond_dim))

        self.up3 = Up(c4, c3)
        self.dec3 = nn.Sequential(ResBlock(c3 * 2, cond_dim), ResBlock(c3 * 2, cond_dim))
        self.dec3_out = nn.Conv2d(c3 * 2, c3, 1)

        self.up2 = Up(c3, c2)
        self.dec2 = nn.Sequential(ResBlock(c2 * 2, cond_dim), ResBlock(c2 * 2, cond_dim))
        self.dec2_out = nn.Conv2d(c2 * 2, c2, 1)

        self.up1 = Up(c2, c1)
        self.dec1 = nn.Sequential(ResBlock(c1 * 2, cond_dim), ResBlock(c1 * 2, cond_dim))
        self.dec1_out = nn.Conv2d(c1 * 2, c1, 1)

        self.head_residual = nn.Sequential(
            nn.GroupNorm(8, c1),
            _act(),
            nn.Conv2d(c1, in_channels, 3, padding=1),
        )
        self.head_starmap = nn.Sequential(
            nn.GroupNorm(8, c1),
            _act(),
            nn.Conv2d(c1, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, psf: torch.Tensor | None = None,
                stellar_strength: float = 1.0,
                nonstellar_strength: float = 1.0):
        """
        x:   (B, C, H, W) input image (linear, ~[0, 1])
        psf: (B, K, K) optional PSF — if None, uses zero kernel as "unknown"
        stellar_strength / nonstellar_strength: inference-time controls
        Returns: dict {sharp, residual, star_map}
        """
        B, _, H, W = x.shape
        if psf is None:
            psf = torch.zeros((B, self.psf_encoder.psf_size, self.psf_encoder.psf_size),
                              device=x.device, dtype=x.dtype)
        c = self.psf_encoder(psf)

        coords = coord_channels(B, H, W, x.device, x.dtype)
        h0 = self.in_conv(torch.cat([x, coords], dim=1))

        # encoder
        e1 = self._apply_blocks(self.enc1, h0, c)
        e2 = self._apply_blocks(self.enc2, self.down1(e1), c)
        e3 = self._apply_blocks(self.enc3, self.down2(e2), c)
        b = self._apply_blocks(self.bottleneck, self.down3(e3), c)

        # decoder w/ skip connections
        d3 = self.up3(b)
        d3 = self._apply_blocks(self.dec3, torch.cat([d3, e3], dim=1), c)
        d3 = self.dec3_out(d3)

        d2 = self.up2(d3)
        d2 = self._apply_blocks(self.dec2, torch.cat([d2, e2], dim=1), c)
        d2 = self.dec2_out(d2)

        d1 = self.up1(d2)
        d1 = self._apply_blocks(self.dec1, torch.cat([d1, e1], dim=1), c)
        d1 = self.dec1_out(d1)

        residual = torch.tanh(self.head_residual(d1)) * self.residual_scale
        star_logits = self.head_starmap(d1)
        star_map = torch.sigmoid(star_logits)

        # Inference-time mixing: scale the residual differently for stars vs nebula
        if stellar_strength != 1.0 or nonstellar_strength != 1.0:
            mix = star_map * stellar_strength + (1 - star_map) * nonstellar_strength
            residual = residual * mix

        sharp = (x + residual).clamp(0.0, 1.5)

        return {
            "sharp": sharp,
            "residual": residual,
            "star_map": star_map,
            "star_logits": star_logits,
        }

    @staticmethod
    def _apply_blocks(blocks: nn.Sequential, x, c):
        for blk in blocks:
            if isinstance(blk, ResBlock):
                x = blk(x, c)
            else:
                x = blk(x)
        return x
