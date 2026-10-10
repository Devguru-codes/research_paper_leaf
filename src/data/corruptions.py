"""MedLeaf-C: corruption suite on 448px leaf crops (uint8 RGB + binary mask).

Photometric corruptions leave the (oracle) mask unchanged. Severity levels 1..3.
All randomness is driven by an explicit numpy Generator for reproducibility.
"""
from __future__ import annotations

import cv2
import numpy as np

SEVERITIES = (1, 2, 3)


def _clip(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def gaussian_noise(img, mask, s, rng):
    sig = (0.04, 0.08, 0.12)[s - 1] * 255
    return _clip(img.astype(np.float32) + rng.normal(0, sig, img.shape)), mask


def gaussian_blur(img, mask, s, rng):
    sig = (1.0, 2.0, 3.5)[s - 1]
    return cv2.GaussianBlur(img, (0, 0), sig), mask


def jpeg(img, mask, s, rng):
    q = (30, 15, 7)[s - 1]
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), mask


def low_resolution(img, mask, s, rng):
    r = (128, 80, 48)[s - 1]
    small = cv2.resize(img, (r, r), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, img.shape[:2][::-1], interpolation=cv2.INTER_LINEAR), mask


def darken(img, mask, s, rng):
    f = (0.7, 0.5, 0.35)[s - 1]
    out = img.astype(np.float32)
    out[mask > 0] *= f
    return _clip(out), mask


def contrast(img, mask, s, rng):
    f = (0.6, 0.4, 0.25)[s - 1]
    out = img.astype(np.float32)
    mu = out[mask > 0].mean(0)
    out[mask > 0] = mu + f * (out[mask > 0] - mu)
    return _clip(out), mask


def hue_shift(img, mask, s, rng):
    """Leaf 'yellowing/browning': rotate hue towards yellow-red on the leaf only."""
    d = (6, 12, 20)[s - 1]  # OpenCV hue units (0..180)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = np.where(mask > 0, (hsv[..., 0] - d) % 180, hsv[..., 0])
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB), mask


def desaturate(img, mask, s, rng):
    f = (0.5, 0.2, 0.0)[s - 1]  # 0 = grayscale: removes colour cues entirely
    g = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB).astype(np.float32)
    return _clip(g + f * (img.astype(np.float32) - g)), mask


def occlusion(img, mask, s, rng):
    """Random grey rectangles over the leaf covering ~10/20/30% of its bounding box."""
    frac = (0.10, 0.20, 0.30)[s - 1]
    out = img.copy()
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    area = (y1 - y0) * (x1 - x0) * frac / 2  # two rectangles of size side x 2*side
    side = int(np.sqrt(area / 2))
    for _ in range(2):
        cy, cx = rng.integers(y0, y1 + 1), rng.integers(x0, x1 + 1)
        out[max(cy - side // 2, 0):cy + (side + 1) // 2, max(cx - side, 0):cx + side] = 128
    return out, mask


def _procedural_bg(shape, s, rng):
    H, W = shape[:2]
    if s == 1:  # flat random colour
        return np.broadcast_to(rng.integers(60, 230, 3).astype(np.uint8), (H, W, 3)).copy()
    base = np.zeros((H, W, 3), np.float32)
    octaves = (4, 8) if s == 2 else (4, 8, 16, 32, 64)
    for o in octaves:
        n = rng.random((o, o, 3)).astype(np.float32)
        base += cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC) / len(octaves)
    lo = np.array([70, 55, 35]) if s == 3 else rng.integers(40, 120, 3)  # soil/foliage-like
    hi = np.array([150, 160, 90]) if s == 3 else rng.integers(140, 240, 3)
    return _clip(lo + base * (hi - lo))


def background(img, mask, s, rng):
    bg = _procedural_bg(img.shape, s, rng)
    m = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)[..., None]
    return _clip(m * img + (1 - m) * bg), mask


CORRUPTIONS = dict(gaussian_noise=gaussian_noise, gaussian_blur=gaussian_blur, jpeg=jpeg,
                   low_resolution=low_resolution, darken=darken, contrast=contrast, hue_shift=hue_shift,
                   desaturate=desaturate, occlusion=occlusion, background=background)
