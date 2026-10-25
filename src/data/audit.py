"""Dataset audit + preprocessing cache.

  python -m src.data.audit            # writes cache/index.csv, cache/{img,mask,vein}/, cache/audit.json

* Files are indexed by (class folder, filename): Psidium guajava and Punica granatum both use
  the 'PG-S-xxx' prefix, so a bare filename is not unique.
* Exact duplicates (identical photographs saved twice; raw 1024-bit dHash distance 0) are
  dropped for the leakage-free protocol.
* Re-shots of the same physical leaf (re-photographed, sometimes under different lighting or
  orientation) are *grouped*, never split across folds. A pair is linked when either
    - 1024-bit dHash of the raw photographs <= 20 (same framing, re-shot), or
    - SIFT + ratio test + RANSAC (partial affine) gives >= 40 geometric inliers whose
      similarity transform has scale in [0.8, 1.25] (crops are size-normalised) and whose
      convex hull covers >= 25% of the leaf (rejects degenerate/hub matches on repetitive
      texture, e.g. Mentha)
      (thresholds chosen by visual inspection; over-grouping is conservative: it only
      constrains splits and cannot cause leakage). A dihedral-invariant hash on the
      leaf-centred crop was rejected: it links distinct lanceolate leaves (and even species)
      with the same silhouette.
  Groups are the connected components of the link graph.
"""
from __future__ import annotations

import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.data.masks import bg_colour_mask, crop_square, white_bg_mask
from src.data.taxonomy import CLASSES
from src.data.venation import vein_pseudolabel
from src.paths import cache_root, raw_root

SIZE = 448
RAW_DHASH_THR = 20
SIFT_INLIER_THR = 40


def dhash(gray: np.ndarray, n: int = 32) -> np.ndarray:
    a = np.asarray(Image.fromarray(gray).resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def _process(args):
    src, stem, out = args
    rgb = cv2.cvtColor(cv2.imread(str(src), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = bg_colour_mask(rgb) if os.environ.get("MEDLEAF_MASK") == "bgcolour" else white_bg_mask(rgb)
    area_frac = float(mask.mean())
    if not (out / "vein" / f"{stem}.png").exists():
        rgb_c, m_c, _ = crop_square(rgb, mask, size=SIZE)
        vein = vein_pseudolabel(rgb_c, m_c)
        cv2.imwrite(str(out / "img" / f"{stem}.jpg"), cv2.cvtColor(rgb_c, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / "mask" / f"{stem}.png"), m_c * 255)
        cv2.imwrite(str(out / "vein" / f"{stem}.png"), (vein * 255).astype(np.uint8))
    m_c = cv2.imread(str(out / "mask" / f"{stem}.png"), 0) > 127
    n_cc = cv2.connectedComponents(m_c.astype(np.uint8), connectivity=4)[0] - 1
    raw_hash = dhash(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    return stem, area_frac, int(n_cc), raw_hash


def _sift(args):
    stem, out = args
    g = cv2.imread(str(out / "img" / f"{stem}.jpg"), 0)
    m = cv2.erode(cv2.imread(str(out / "mask" / f"{stem}.png"), 0), np.ones((7, 7), np.uint8))
    kp, des = cv2.SIFT_create(nfeatures=800).detectAndCompute(g, m)
    return np.float32([k.pt for k in kp]), des, float((m > 127).sum())


def _sift_inliers(args):
    """Returns (n_inliers, similarity scale, inlier convex-hull area / leaf area)."""
    (p1, d1, a1), (p2, d2, _) = args
    if d1 is None or d2 is None or len(d1) < 8 or len(d2) < 8:
        return 0, 0.0, 0.0
    ms = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [a for a, b in (x for x in ms if len(x) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 8:
        return 0, 0.0, 0.0
    A = np.float32([p1[g.queryIdx] for g in good])
    B = np.float32([p2[g.trainIdx] for g in good])
    M, inl = cv2.estimateAffinePartial2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if M is None or inl is None or inl.sum() < 3:
        return 0, 0.0, 0.0
    inl = inl[:, 0].astype(bool)
    scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    hull = float(cv2.contourArea(cv2.convexHull(A[inl])) / max(a1, 1.0))
    return int(inl.sum()), scale, hull


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i, j):
        self.p[self.find(i)] = self.find(j)


def main():
    root, out = raw_root(), cache_root()
    for d in ("img", "mask", "vein"):
        (out / d).mkdir(parents=True, exist_ok=True)
    # class folders: the medicinal species, or (second dataset) whatever folders the raw root holds
    classes = CLASSES if (root / CLASSES[0]).exists() else sorted(d.name for d in root.iterdir() if d.is_dir())
    rows = []
    for label, cls in enumerate(classes):
        for f in sorted((root / cls).iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                rows.append(dict(rel_path=f"{cls}/{f.name}", class_name=cls, label=label,
                                 stem=f"{label:02d}_{f.stem}"))
    df = pd.DataFrame(rows)
    assert df.stem.is_unique
    N = len(df)

    with ProcessPoolExecutor() as ex:
        res = list(ex.map(_process, [(root / p, s, out) for p, s in zip(df.rel_path, df.stem)], chunksize=8))
        feats = list(ex.map(_sift, [(s, out) for s in df.stem], chunksize=16))
    df["leaf_area_frac"] = [r[1] for r in res]
    df["n_components"] = [r[2] for r in res]
    raw_h = np.stack([r[3] for r in res])

    # exact duplicates (raw photographs identical)
    h = raw_h.astype(np.float32)  # Hamming distance as two matmuls: N x N, not N x N x 1024 (29 GB for LeafSnap)
    D_raw = np.rint(h @ (1 - h).T + (1 - h) @ h.T).astype(np.int32)
    exact = _DSU(N)
    ei, ej = np.nonzero(np.triu(D_raw == 0, 1))
    for i, j in zip(ei, ej):
        exact.union(i, j)
    df["exact_group"] = [exact.find(i) for i in range(N)]
    df["keep"] = ~df.duplicated("exact_group", keep="first")

    # re-shot links
    links = _DSU(N)
    for i, j in zip(ei, ej):
        links.union(i, j)
    hi, hj = np.nonzero(np.triu(D_raw <= RAW_DHASH_THR, 1))
    hint = root / "specimen_groups.csv"
    # near-duplicate hashes only when the leaf is unknown: on mostly blank frames (LeafSnap lab) they link
    # unrelated leaves across species
    for i, j in zip(hi, hj) if not hint.exists() else ():
        links.union(i, j)
    if hint.exists():  # the dataset names the physical leaf of every photo: group by it, no SIFT search
        leaf = pd.read_csv(hint).set_index("rel_path").group
        for idx in df.groupby(df.rel_path.map(leaf).fillna(df.rel_path)).indices.values():
            for i in idx[1:]:
                links.union(int(idx[0]), int(i))
        pairs = []
    else:
        pairs = [(i, j) for _, g in df.groupby("label") for i, j in itertools.combinations(g.index, 2)]
    with ProcessPoolExecutor() as ex:
        inl = list(ex.map(_sift_inliers, ((feats[i], feats[j]) for i, j in pairs), chunksize=256))
    sift_links = [(i, j, n) for (i, j), (n, sc, hull) in zip(pairs, inl)
                  if n >= SIFT_INLIER_THR and 0.8 <= sc <= 1.25 and hull >= 0.25]
    for i, j, _ in sift_links:
        links.union(i, j)
    df["leaf_group"] = pd.Series([links.find(i) for i in range(N)]).astype("category").cat.codes
    df.to_csv(out / "index.csv", index=False)

    gsize = df[df.keep].groupby("leaf_group").size()
    counts = df.groupby("class_name").size()
    cross = sum(df.label[i] != df.label[j] for i, j in zip(hi, hj))
    audit = dict(
        n_images=N, n_classes=len(classes), n_exact_duplicate_pairs=int(len(ei)),
        n_after_exact_dedup=int(df.keep.sum()),
        exact_duplicate_pairs=[(df.rel_path[i], df.rel_path[j]) for i, j in zip(ei, ej)],
        n_raw_dhash_links=int(len(hi)), raw_dhash_cross_class_links=int(cross),
        n_sift_links=len(sift_links),
        sift_links=[(df.rel_path[i], df.rel_path[j], n) for i, j, n in sift_links],
        n_leaf_groups=int(len(gsize)), n_multi_image_groups=int((gsize > 1).sum()),
        n_images_in_multi_groups=int(gsize[gsize > 1].sum()), largest_group=int(gsize.max()),
        min_class=int(counts.min()), max_class=int(counts.max()),
        prefix_collisions=int(df.rel_path.str.split("/").str[1].duplicated(keep=False).sum()),
        multi_component_masks=int((df.n_components > 1).sum()),
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=1))
    print(json.dumps({k: v for k, v in audit.items() if not isinstance(v, list)}, indent=1))


if __name__ == "__main__":
    main()
