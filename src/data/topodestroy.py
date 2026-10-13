"""Topology-destruction probes with matched, chi-preserving controls.

Blade edits (exact chi of the mask is verified):
  holes   : punch n interior disks to background (white)      -> chi decreases by n
  bites   : remove n disks of the same radius centred on the boundary -> chi unchanged
Venation edits (applied to the lamina; blade mask unchanged):
  sever   : inpaint n disks centred on strong vein pixels (pseudo-label)  -> breaks vein paths
  lamina  : inpaint n disks of the same radius on non-vein lamina pixels -> control
Each edit records pixel RMSE, colour-histogram distance (and LPIPS when available, computed by
the caller). Blade holes/bites are area-matched; vein severing necessarily changes more pixels
than lamina inpainting, so the analysis regresses the prediction change on visual distance
with a chi-changing indicator (i.e. compares at equal visual magnitude) instead of assuming
matched pairs.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.models.cubical_ect import exact_euler


def chi(mask: np.ndarray) -> int:
    import torch
    return int(exact_euler(torch.as_tensor(mask > 0)[None]).item())


def _disk(shape, cy, cx, r):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def _radius(mask, frac=0.004):
    return max(3, int(np.sqrt(frac * mask.sum() / np.pi)))


def blade_holes(img, mask, rng, n=3):
    r = _radius(mask)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((4 * r + 3, 4 * r + 3), np.uint8))
    ys, xs = np.nonzero(inner)
    out, m = img.copy(), mask.copy()
    placed = []
    for _ in range(200):
        if len(placed) == n or len(ys) == 0:
            break
        k = rng.integers(len(ys))
        cy, cx = ys[k], xs[k]
        if all((cy - py) ** 2 + (cx - px) ** 2 > (4 * r) ** 2 for py, px in placed):
            d = _disk(m.shape, cy, cx, r)
            out[d], m[d] = 255, 0
            placed.append((cy, cx))
    return out, m


def blade_bites(img, mask, rng, n=3):
    r = int(round(_radius(mask) * np.sqrt(2)))  # ~half of a boundary disk lies outside: match removed area
    edge = mask.astype(np.uint8) - cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(edge)
    out, m = img.copy(), mask.copy()
    c0 = chi(mask)
    placed = []
    for _ in range(400):
        if len(placed) == n or len(ys) == 0:
            break
        k = rng.integers(len(ys))
        cy, cx = ys[k], xs[k]
        if any((cy - py) ** 2 + (cx - px) ** 2 <= (4 * r) ** 2 for py, px in placed):
            continue
        d = _disk(m.shape, cy, cx, r)
        m2 = m.copy()
        m2[d] = 0
        if chi(m2) == c0:  # keep only bites that preserve topology
            out[d], m = 255, m2
            placed.append((cy, cx))
    return out, m


def _inpaint_disks(img, mask, centres, r):
    hole = np.zeros(mask.shape, np.uint8)
    for cy, cx in centres:
        hole[_disk(mask.shape, cy, cx, r)] = 1
    hole &= (mask > 0).astype(np.uint8)
    # inpaint from lamina only: temporarily paint background with the mean lamina colour
    base = img.copy()
    base[mask == 0] = img[mask > 0].mean(0).astype(np.uint8)
    out = cv2.inpaint(base, hole, 2 * r, cv2.INPAINT_TELEA)
    out[mask == 0] = img[mask == 0]
    return out


def _pick(ys, xs, rng, n, sep):
    placed = []
    for _ in range(400):
        if len(placed) == n or len(ys) == 0:
            break
        k = rng.integers(len(ys))
        if all((ys[k] - py) ** 2 + (xs[k] - px) ** 2 > sep ** 2 for py, px in placed):
            placed.append((ys[k], xs[k]))
    return placed


def vein_sever(img, mask, vein, rng, n=8, thr=0.5):
    r = _radius(mask, 0.002)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((2 * r + 3, 2 * r + 3), np.uint8))
    ys, xs = np.nonzero((vein >= thr) & (inner > 0))
    return _inpaint_disks(img, mask, _pick(ys, xs, rng, n, 6 * r), r), mask.copy()


def vein_control(img, mask, vein, rng, n=8, thr=0.1):
    r = _radius(mask, 0.002)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((2 * r + 3, 2 * r + 3), np.uint8))
    far = cv2.dilate((vein >= thr).astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) == 0
    ys, xs = np.nonzero(far & (inner > 0))
    return _inpaint_disks(img, mask, _pick(ys, xs, rng, n, 6 * r), r), mask.copy()


def visual_distance(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict:
    l2 = float(np.sqrt(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()))
    hd = 0.0
    for c in range(3):
        ha = np.histogram(a[..., c][mask > 0], 32, (0, 256))[0].astype(np.float64)
        hb = np.histogram(b[..., c][mask > 0], 32, (0, 256))[0].astype(np.float64)
        ha, hb = ha / max(ha.sum(), 1), hb / max(hb.sum(), 1)
        hd += 0.5 * np.abs(ha - hb).sum() / 3
    return dict(pixel_rmse=l2, hist_tv=float(hd))


EDITS = dict(blade_holes=("blade", True), blade_bites=("blade", False),
             vein_sever=("vein", True), vein_control=("vein", False))
