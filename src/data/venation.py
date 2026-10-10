"""Venation pseudo-labels (multiscale ridge filtering) and the skeleton-graph ablation variant.

The pseudo-label only *supervises* the learned vein head; the model's venation field is the
head's prediction (a learned field, not an observed structure).
"""
import cv2
import numpy as np
from skimage.filters import sato
from skimage.morphology import skeletonize


def vein_pseudolabel(rgb: np.ndarray, mask: np.ndarray, sigmas=(1.0, 2.0, 3.0),
                     border_px: int = 6) -> np.ndarray:
    """Soft vein map in [0,1] (float32), zero outside the eroded mask.

    Veins on these leaves are mostly lighter than the lamina, so bright ridges are used on a
    CLAHE-normalised luminance channel. The mask is eroded so the blade outline is not
    mistaken for a vein.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    L = L.astype(np.float32) / 255.0
    inner = cv2.erode(mask.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border_px + 1,) * 2))
    L = np.where(mask > 0, L, float(np.median(L[mask > 0])) if mask.any() else 0.0)
    r = sato(L, sigmas=sigmas, black_ridges=False).astype(np.float32)
    r *= inner
    vals = r[inner > 0]
    if vals.size == 0:
        return np.zeros_like(r)
    lo, hi = np.percentile(vals, 50), np.percentile(vals, 99)
    r = np.clip((r - lo) / max(hi - lo, 1e-6), 0, 1) * inner
    return r.astype(np.float32)


def vein_skeleton(vein: np.ndarray, thr: float = 0.35) -> np.ndarray:
    """Hard threshold + skeletonize (ablation: the 'brittle' graph-style pipeline)."""
    return skeletonize(vein > thr).astype(np.uint8)
