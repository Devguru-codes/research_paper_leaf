"""Batched GPU augmentation (same transform family as the CPU path in dataset.py).

Per sample: rotation U(-180,180), scale U(0.85,1.15), translation U(-5%,5%), horizontal flip p=0.5,
applied jointly to image (bilinear, white fill), mask (nearest) and vein map (bilinear, zero fill),
resampled directly to the training size; then brightness/contrast/saturation U(0.8,1.2) and a small
hue rotation (+-0.02 turns, YIQ rotation) on the image only.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.data.dataset import MEAN, STD


def _affine(B, device, generator=None, rot: bool = True):
    r = lambda lo, hi: torch.empty(B, device=device).uniform_(lo, hi, generator=generator)  # noqa: E731
    ang = r(-math.pi, math.pi) if rot else torch.zeros(B, device=device)
    s = r(0.85, 1.15)
    tx, ty = r(-0.1, 0.1), r(-0.1, 0.1)  # +-5% of the side in normalised [-1,1] coords
    flip = torch.where(torch.rand(B, device=device, generator=generator) < 0.5, -1.0, 1.0)
    c, si = torch.cos(ang), torch.sin(ang)
    # output -> input mapping: inverse of (translate o rotate o scale o flip)
    A = torch.zeros(B, 2, 3, device=device)
    A[:, 0, 0] = flip * c / s
    A[:, 0, 1] = flip * si / s
    A[:, 1, 0] = -si / s
    A[:, 1, 1] = c / s
    A[:, 0, 2] = -(A[:, 0, 0] * tx + A[:, 0, 1] * ty)
    A[:, 1, 2] = -(A[:, 1, 0] * tx + A[:, 1, 1] * ty)
    return A


def gpu_augment(x_u8, m_u8, v_u8, size: int, generator=None, rot: bool = True):
    """x_u8 (B,3,H,W) uint8, m_u8 (B,H,W) uint8 {0,255}, v_u8 (B,H,W) uint8 -> normalised img, mask, vein."""
    B = x_u8.shape[0]
    dev = x_u8.device
    x = x_u8.float() / 255.0
    m = (m_u8.float() / 255.0)[:, None]
    v = (v_u8.float() / 255.0)[:, None]
    grid = F.affine_grid(_affine(B, dev, generator, rot), (B, 1, size, size), align_corners=False)
    valid = F.grid_sample(torch.ones_like(m), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + (1 - valid)
    m = F.grid_sample(m, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    v = F.grid_sample(v, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    # photometric (image only)
    u = lambda lo, hi: torch.empty(B, 1, 1, 1, device=dev).uniform_(lo, hi, generator=generator)  # noqa: E731
    x = x * u(0.8, 1.2)
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = gray.mean((2, 3), keepdim=True) + u(0.8, 1.2) * (x - gray.mean((2, 3), keepdim=True))
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = gray + u(0.8, 1.2) * (x - gray)
    h = u(-0.02, 0.02)[:, :, 0, 0] * 2 * math.pi  # hue rotation in YIQ
    T = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], device=dev)
    Ti = torch.linalg.inv(T)
    c, s = torch.cos(h)[:, 0], torch.sin(h)[:, 0]
    Rm = torch.zeros(B, 3, 3, device=dev)
    Rm[:, 0, 0] = 1
    Rm[:, 1, 1], Rm[:, 1, 2], Rm[:, 2, 1], Rm[:, 2, 2] = c, -s, s, c
    M = Ti[None] @ Rm @ T[None]
    x = torch.einsum("bij,bjhw->bihw", M, x).clamp(0, 1)
    x = (x - MEAN.to(dev)) / STD.to(dev)
    return x, m[:, 0], v[:, 0]
