"""Leaf masks from the white-background segmented images."""
import cv2
import numpy as np


def white_bg_mask(rgb: np.ndarray, thr: int = 235, min_frac: float = 0.002) -> np.ndarray:
    """Binary leaf mask (uint8 0/1) from an RGB image on a white background.

    A pixel is background when all channels exceed `thr`. Morphological closing/opening
    removes JPEG halo noise; the largest connected component and any component larger than
    `min_frac` of the image are kept (some leaves are compound / lobed).
    """
    return _clean((rgb.min(axis=2) < thr).astype(np.uint8), min_frac)


def bg_colour_mask(rgb: np.ndarray, border: float = 0.04) -> np.ndarray:
    """Leaf mask for photos on a plain but not white background (LeafSnap lab and field images):
    the background colour is the median of the image border, a pixel is leaf when its Lab distance
    to that colour exceeds an Otsu threshold. One component is kept: the one with the most leaf-coloured
    (yellow-green, saturated) pixels plus a small area term, so a larger colour-checker ruler or specks
    lose to the leaf."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = max(2, int(border * min(rgb.shape[:2])))
    edge = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3), lab[:, :b].reshape(-1, 3),
                           lab[:, -b:].reshape(-1, 3)])
    d = np.linalg.norm(lab - np.median(edge, 0), axis=2)
    d8 = np.clip(255 * d / max(d.max(), 1e-6), 0, 255).astype(np.uint8)
    _, fg = cv2.threshold(d8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out = _clean(fg.astype(np.uint8), min_frac=0.0)
    n, cc = cv2.connectedComponents(out, connectivity=8)
    if n <= 1:
        return np.ones_like(out)  # no leaf found: whole image (counted in the audit)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    leafy = (hsv[..., 0] >= 15) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 40)  # OpenCV hue: 30-180 deg
    score = np.bincount(cc.ravel(), leafy.ravel(), n) + 0.05 * np.bincount(cc.ravel(), minlength=n)
    return (cc == 1 + int(np.argmax(score[1:]))).astype(np.uint8)


def _clean(fg: np.ndarray, min_frac: float) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return fg
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, bool)
    keep[1 + int(np.argmax(areas))] = True
    keep[1:][areas >= min_frac * fg.size] = True
    out = keep[lab].astype(np.uint8)
    # fill pin-holes smaller than ~0.01% of the image (specular highlights) but keep real holes
    inv = (1 - out).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for i in range(1, n2):
        x, y, w, h, a = st2[i]
        touches = x == 0 or y == 0 or x + w == out.shape[1] or y + h == out.shape[0]
        if not touches and a < 1e-4 * out.size:
            out[lab2 == i] = 1
    return out


def crop_square(rgb: np.ndarray, mask: np.ndarray, margin: float = 0.04, size: int = 448,
                extra: list[np.ndarray] | None = None):
    """Square crop centred on the leaf whose *inscribed circle* contains the whole leaf
    (so any later rotation about the centre is lossless); pad with white / zeros, resize."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    rmax = float(np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2).max())) + 1
    side = int(2 * rmax * (1 + margin))
    top, left = cy - side // 2, cx - side // 2

    def _cut(a, fill):
        H, W = a.shape[:2]
        out = np.full((side, side) + a.shape[2:], fill, dtype=a.dtype)
        sy0, sx0 = max(top, 0), max(left, 0)
        sy1, sx1 = min(top + side, H), min(left + side, W)
        out[sy0 - top:sy1 - top, sx0 - left:sx1 - left] = a[sy0:sy1, sx0:sx1]
        return out

    rgb_c = cv2.resize(_cut(rgb, 255), (size, size), interpolation=cv2.INTER_AREA)
    m_c = cv2.resize(_cut(mask, 0).astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    m_c = (m_c > 0.5).astype(np.uint8)
    ex = []
    for e in extra or []:
        ex.append(cv2.resize(_cut(e, 0), (size, size), interpolation=cv2.INTER_AREA))
    return rgb_c, m_c, ex
