"""Shape-texture cue-conflict stimuli: the silhouette of leaf A filled with the lamina of leaf B.

B is aligned to A by matching principal axes (rotation + anisotropic scaling along the axes +
centroid translation) and then enlarged until B's lamina covers >= 99% of A's silhouette, so
that B's own outline lies outside A and contributes no shape cue; the remaining uncovered
pixels are filled by inpainting from B's texture. Shape cue = class(A), texture
(incl. venation) cue = class(B).
"""
from __future__ import annotations

import cv2
import numpy as np


def _axes(mask):
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    cov = np.cov((pts - c).T)
    w, v = np.linalg.eigh(cov)
    v = v[:, ::-1]  # major axis first
    proj = (pts - c) @ v
    ext = np.percentile(np.abs(proj), 99, axis=0) + 1.0
    return c, v, ext


def cue_conflict(img_a, mask_a, img_b, mask_b, flip_major: bool = False):
    ca, va, ea = _axes(mask_a)
    cb, vb, eb = _axes(mask_b)
    if flip_major:
        vb = vb * np.array([-1.0, 1.0])
    # map A-frame coordinates -> B-frame coordinates (for inverse warping)
    H, W = mask_a.shape
    ma = mask_a > 0
    for grow in np.linspace(1.0, 3.0, 41):
        S = np.diag(eb / (ea * grow))  # shrinking the A->B map == enlarging B in A's frame
        M = vb @ S @ va.T
        t = cb - M @ ca
        A = np.hstack([M, t[:, None]]).astype(np.float32)
        cov = cv2.warpAffine(mask_b.astype(np.uint8), A, (W, H), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
        if (cov[ma] > 0).mean() >= 0.99:
            break
    tex = cv2.warpAffine(img_b, A, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    hole = ((mask_a > 0) & (cov == 0)).astype(np.uint8)
    if hole.any():
        tex = cv2.inpaint(tex, hole, 7, cv2.INPAINT_TELEA)
    out = np.full_like(img_a, 255)
    out[mask_a > 0] = tex[mask_a > 0]
    r = eb / ea
    return out, mask_a.copy(), dict(hole_frac=float(hole.sum() / max(mask_a.sum(), 1)), grow=float(grow),
                                    stretch=float(max(r) / min(r)))


def make_pairs(labels: np.ndarray, rng: np.random.Generator):
    """For each index i (shape donor A) pick a texture donor B of a different class."""
    n = len(labels)
    out = []
    for i in range(n):
        while True:
            j = int(rng.integers(n))
            if labels[j] != labels[i]:
                out.append((i, j))
                break
    return out
