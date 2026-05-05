"""Loss functions for OpenBXT.

The combined loss has four parts:

  1. L1 reconstruction in linear units            — bulk fidelity
  2. Gradient (edge) loss                          — encourages sharp edges
  3. Local-flux conservation                       — prevents fabricated brightness
  4. Star-map BCE                                  — supervises the stellar head
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def l1(pred, target):
    return (pred - target).abs().mean()


def gradient_loss(pred, target):
    """Match Sobel-like first-order gradients of pred and target.

    Encourages the network to recover sharp edges where they exist in target.
    """
    def grad(t):
        gx = t[..., :, 1:] - t[..., :, :-1]
        gy = t[..., 1:, :] - t[..., :-1, :]
        return gx, gy
    pgx, pgy = grad(pred)
    tgx, tgy = grad(target)
    return (pgx - tgx).abs().mean() + (pgy - tgy).abs().mean()


def flux_conservation_loss(pred, target, kernel: int = 9):
    """Penalize differences in local average brightness.

    Deconvolution must conserve total flux; if our prediction shifts mean
    brightness in any local window, that's invented light or lost flux.
    """
    # avg_pool acts per-channel
    pad = kernel // 2
    pp = F.avg_pool2d(pred, kernel_size=kernel, stride=1, padding=pad)
    tp = F.avg_pool2d(target, kernel_size=kernel, stride=1, padding=pad)
    return (pp - tp).abs().mean()


def star_bce(star_logits, mask):
    return F.binary_cross_entropy_with_logits(star_logits, mask)


class OpenBXTLoss(nn.Module):
    def __init__(
        self,
        w_l1: float = 1.0,
        w_grad: float = 0.5,
        w_flux: float = 0.5,
        w_star: float = 0.2,
        star_weight_factor: float = 5.0,
    ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_grad = w_grad
        self.w_flux = w_flux
        self.w_star = w_star
        self.star_weight_factor = star_weight_factor

    def forward(self, output: dict, target: torch.Tensor, star_mask: torch.Tensor):
        sharp = output["sharp"]
        # Weighted L1: stars contribute more (small area, high importance)
        weight = 1.0 + self.star_weight_factor * star_mask
        l1_term = ((sharp - target).abs() * weight).mean()

        grad_term = gradient_loss(sharp, target)
        flux_term = flux_conservation_loss(sharp, target)
        star_term = star_bce(output["star_logits"], star_mask)

        total = (
            self.w_l1 * l1_term
            + self.w_grad * grad_term
            + self.w_flux * flux_term
            + self.w_star * star_term
        )
        return total, {
            "l1": l1_term.detach(),
            "grad": grad_term.detach(),
            "flux": flux_term.detach(),
            "star": star_term.detach(),
            "total": total.detach(),
        }
