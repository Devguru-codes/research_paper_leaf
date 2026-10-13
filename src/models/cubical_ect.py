r"""Differentiable, weighted Euler-characteristic sweep on a cubical grid.

Notation (paper, Sec. Method):
  grid pixels p with centres x_p (normalised by the blade centroid and RMS radius),
  0-cells = pixels, 1-cells = 4-adjacent pixel pairs, 2-cells = 2x2 squares,
  lower-star heights   h_theta(c) = max_{v in c} <x_v, theta>,
  cell weights         w_0(p) = a(p),  w_1(e) = min_{v in e} a(v),  w_2(q) = min_{v in q} a(v),
  relaxed Euler curve  chi_theta(t) = sum_d (-1)^d sum_{c in C_d} w_d(c) sigma(lambda (t - h_theta(c))),
  filtration increment Delta chi(theta, k) = chi_theta(t_k) - chi_theta(t_{k-1}),  k = 1..T,
with t_0 = -inf (chi = 0) and t_T = +inf (chi = total weighted Euler characteristic), and
T-1 inner thresholds uniform on (-R, R).

Because the outer thresholds are infinite, the soft slab memberships
  s_k(h) = sigma(lambda (t_k - h)) - sigma(lambda (t_{k-1} - h))
sum to exactly 1 over k, and Delta chi(theta, k) = sum_c (-1)^{d(c)} w(c) s_k(h_theta(c)).

Implementation: the geometry (heights) is fixed -- directions are fixed and coordinates come
from the mask -- so gradients are only needed w.r.t. the weights. Heights are therefore
hard-binned into `nbins` fine bins (resolution 2R/nbins), weights are scatter-added into a
per-direction histogram, and slab memberships are applied to bin centres with one matmul.
This makes any *linear* slab statistic (Euler increment, area, perimeter, moments) cost
O(B K N) memory with no sorting.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def directions(K: int, device=None, dtype=torch.float32) -> torch.Tensor:
    ang = torch.arange(K, device=device, dtype=dtype) * (2 * math.pi / K)
    return torch.stack([torch.cos(ang), torch.sin(ang)], -1)  # (K, 2)


def pixel_coords(H: int, W: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Pixel centres in a mathematically oriented frame: x = column, y = -row (unit square)."""
    ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
    xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, -yy], -1)  # (H, W, 2)


def normalise_frame(mask: torch.Tensor, eps: float = 1e-6):
    """Centroid and RMS radius of a (soft) mask. mask: (B,H,W) -> centre (B,2), scale (B,)."""
    B, H, W = mask.shape
    xy = pixel_coords(H, W, mask.device, mask.dtype)
    tot = mask.sum((1, 2)).clamp_min(eps)
    c = torch.einsum("bhw,hwc->bc", mask, xy) / tot[:, None]
    d2 = ((xy[None] - c[:, None, None]) ** 2).sum(-1)
    s = ((d2 * mask).sum((1, 2)) / tot).sqrt().clamp_min(eps)
    return c, s


def frame_coords(xy: torch.Tensor, centre: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Fixed coordinates xy (...,2) -> per-sample normalised coordinates (B,...,2)."""
    lead = (centre.shape[0],) + (1,) * (xy.dim() - 1)
    return (xy[None] - centre.view(*lead, 2)) / scale.view(*lead, 1)


class SweepGeometry(nn.Module):
    """Fixed directions, thresholds and slab-membership kernel."""

    def __init__(self, K: int = 32, T: int = 16, R: float = 2.0, kappa: float = 4.0, nbins: int = 256):
        super().__init__()
        self.K, self.T, self.R, self.nbins = K, T, R, nbins
        self.lam = kappa * T / (2 * R)  # sharpness relative to slab width
        self.register_buffer("dirs", directions(K), persistent=False)
        self.register_buffer("t_inner", torch.linspace(-R, R, T + 1)[1:-1], persistent=False)
        self._kernels: dict = {}  # (device, dtype) -> (nbins, T) slab kernel, built in float64

    def slab_membership(self, h: torch.Tensor) -> torch.Tensor:
        """Soft slab membership s_k(h) for heights h (...,) -> (..., T); rows sum to 1."""
        c = torch.sigmoid(self.lam * (self.t_inner.to(h) - h[..., None]))  # (..., T-1)
        zeros, ones = torch.zeros_like(c[..., :1]), torch.ones_like(c[..., :1])
        C = torch.cat([zeros, c, ones], -1)  # cumulative at t_0..t_T
        return C[..., 1:] - C[..., :-1]

    def bin_index(self, h: torch.Tensor) -> torch.Tensor:
        idx = ((h + self.R) * (self.nbins / (2 * self.R))).floor().long()
        return idx.clamp_(0, self.nbins - 1)

    def slab_sums(self, values: torch.Tensor, heights: torch.Tensor) -> torch.Tensor:
        """sum_c values(c) * s_k(h_theta(c)).

        values: (B, N) or (B, K, N); heights: (B, K, N) -> (B, K, T).
        """
        B, K, N = heights.shape
        if values.dim() == 2:
            values = values[:, None, :].expand(B, K, N)
        hist = values.new_zeros(B, K, self.nbins)
        hist = hist.scatter_add(2, self.bin_index(heights), values)
        return hist @ self.slab_kernel(hist.device, hist.dtype)

    def slab_kernel(self, device, dtype) -> torch.Tensor:
        key = (str(device), dtype)
        if key not in self._kernels:
            c = -self.R + (torch.arange(self.nbins, dtype=torch.float64) + 0.5) * (2 * self.R / self.nbins)
            t = torch.linspace(-self.R, self.R, self.T + 1, dtype=torch.float64)[1:-1]
            cum = torch.sigmoid(self.lam * (t - c[:, None]))
            C = torch.cat([torch.zeros_like(cum[:, :1]), cum, torch.ones_like(cum[:, :1])], 1)
            self._kernels[key] = (C[:, 1:] - C[:, :-1]).to(device=device, dtype=dtype)
        return self._kernels[key]


def cubical_cells_by_dim(a: torch.Tensor, h: torch.Tensor):
    """[(weights (B,N_d), heights (B,K,N_d)) for d = 0, 1, 2] of the weighted cubical complex."""
    wv, hv = a.flatten(1), h.flatten(2)
    we = torch.cat([torch.minimum(a[:, :, :-1], a[:, :, 1:]).flatten(1),
                    torch.minimum(a[:, :-1, :], a[:, 1:, :]).flatten(1)], 1)
    he = torch.cat([torch.maximum(h[..., :, :-1], h[..., :, 1:]).flatten(2),
                    torch.maximum(h[..., :-1, :], h[..., 1:, :]).flatten(2)], 2)
    wq = torch.minimum(torch.minimum(a[:, :-1, :-1], a[:, :-1, 1:]),
                       torch.minimum(a[:, 1:, :-1], a[:, 1:, 1:])).flatten(1)
    hq = torch.maximum(torch.maximum(h[..., :-1, :-1], h[..., :-1, 1:]),
                       torch.maximum(h[..., 1:, :-1], h[..., 1:, 1:])).flatten(2)
    return [(wv, hv), (we, he), (wq, hq)]


def valuation_increments(a: torch.Tensor, coords: torch.Tensor, geo: "SweepGeometry") -> torch.Tensor:
    """Filtration increments of the three planar intrinsic volumes (Hadwiger basis), (B,K,T,3):
    [area, perimeter, Euler characteristic], with H_d the weighted d-cell counts per slab:
        area = H2,   perimeter = 2 H1 - 4 H2,   chi = H0 - H1 + H2
    (binary set: number of unit squares, boundary length in pixel units, V - E + F).
    By Hadwiger's theorem every continuous rigid-motion-invariant valuation on planar convex
    bodies is a linear combination of these three."""
    h = vertex_heights(coords, geo.dirs.to(coords))
    H0, H1, H2 = (geo.slab_sums(w, hh) for w, hh in cubical_cells_by_dim(a, h))
    return torch.stack([H2, 2 * H1 - 4 * H2, H0 - H1 + H2], -1)


def cubical_cells(a: torch.Tensor, h: torch.Tensor):
    """Signed cell weights and lower-star heights of the cubical complex.

    a: (B,H,W) vertex weights in [0,1];  h: (B,K,H,W) vertex heights.
    Returns signed weights (B,N) with sign (-1)^dim and heights (B,K,N).
    """
    wv, hv = a.flatten(1), h.flatten(2)
    weh = torch.minimum(a[:, :, :-1], a[:, :, 1:]).flatten(1)
    heh = torch.maximum(h[..., :, :-1], h[..., :, 1:]).flatten(2)
    wev = torch.minimum(a[:, :-1, :], a[:, 1:, :]).flatten(1)
    hev = torch.maximum(h[..., :-1, :], h[..., 1:, :]).flatten(2)
    wsq = torch.minimum(torch.minimum(a[:, :-1, :-1], a[:, :-1, 1:]),
                        torch.minimum(a[:, 1:, :-1], a[:, 1:, 1:])).flatten(1)
    hsq = torch.maximum(torch.maximum(h[..., :-1, :-1], h[..., :-1, 1:]),
                        torch.maximum(h[..., 1:, :-1], h[..., 1:, 1:])).flatten(2)
    w = torch.cat([wv, -weh, -wev, wsq], 1)
    hh = torch.cat([hv, heh, hev, hsq], 2)
    return w, hh


def vertex_heights(coords: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """coords (B,H,W,2) normalised, dirs (K,2) -> (B,K,H,W)."""
    return torch.einsum("bhwc,kc->bkhw", coords, dirs)


def euler_increments(a: torch.Tensor, coords: torch.Tensor, geo: SweepGeometry) -> torch.Tensor:
    """Delta chi(theta,k) for weights a (B,H,W) on the normalised grid coords (B,H,W,2) -> (B,K,T)."""
    h = vertex_heights(coords, geo.dirs.to(coords))
    w, hh = cubical_cells(a, h)
    return geo.slab_sums(w, hh)


def exact_euler(binary: torch.Tensor) -> torch.Tensor:
    """Exact Euler characteristic V - E + F of the cubical complex of a binary image (B,H,W)."""
    a = binary.float()
    v = a.sum((1, 2))
    e = (a[:, :, :-1] * a[:, :, 1:]).sum((1, 2)) + (a[:, :-1, :] * a[:, 1:, :]).sum((1, 2))
    f = (a[:, :-1, :-1] * a[:, :-1, 1:] * a[:, 1:, :-1] * a[:, 1:, 1:]).sum((1, 2))
    return v - e + f


def slab_statistics(a: torch.Tensor, coords: torch.Tensor, geo: SweepGeometry) -> torch.Tensor:
    """Non-topological slab statistics (control for Euler increments), (B,K,T,3):
    area  = sum_p a(p) s_k,
    perimeter = sum_edges |a_i - a_j| s_k  (boundary length of the weighted set),
    spread = sum_p a(p) <x_p, theta_perp>^2 s_k  (second moment across the sweep).
    """
    dirs = geo.dirs.to(coords)
    h = vertex_heights(coords, dirs)  # (B,K,H,W)
    area = geo.slab_sums(a.flatten(1), h.flatten(2))
    pe_h = (a[:, :, :-1] - a[:, :, 1:]).abs().flatten(1)
    pe_v = (a[:, :-1, :] - a[:, 1:, :]).abs().flatten(1)
    hh = torch.cat([torch.maximum(h[..., :, :-1], h[..., :, 1:]).flatten(2),
                    torch.maximum(h[..., :-1, :], h[..., 1:, :]).flatten(2)], 2)
    perim = geo.slab_sums(torch.cat([pe_h, pe_v], 1), hh)
    perp = torch.stack([-dirs[:, 1], dirs[:, 0]], -1)
    hp = torch.einsum("bhwc,kc->bkhw", coords, perp).flatten(2)
    spread = geo.slab_sums(a.flatten(1)[:, None, :] * hp ** 2, h.flatten(2))
    return torch.stack([area, perim, spread], -1)
