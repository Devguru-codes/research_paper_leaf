"""Static paper assets: taxonomy table and the method overview figure.

  python -m src.make_paper_assets --out paper/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.hazards import HAZARD_TIER  # noqa: E402
from src.data.taxonomy import CLASSES, TAXONOMY  # noqa: E402
from src.models.cubical_ect import (SweepGeometry, frame_coords, normalise_frame, pixel_coords,  # noqa: E402
                                    valuation_increments)
from src.paths import cache_root  # noqa: E402

TIER_NAME = {0: "", 1: "caution", 2: "high"}


def taxonomy_table(out: Path):
    idx = pd.read_csv(cache_root() / "index.csv")
    n = idx[idx.keep].groupby("label").size()
    rows = []
    for i, c in enumerate(CLASSES):
        o, f, g, s = TAXONOMY[c]
        common = c[c.find("(") + 1:c.find(")")] if "(" in c else ""
        rows.append((o, f, s, common, int(n.get(i, 0)), TIER_NAME[HAZARD_TIER[c]]))
    rows.sort()
    lines = ["\\begin{tabular}{lllrl}", "\\toprule", "Order / family & Species & Name & $n$ & Hazard \\\\", "\\midrule"]
    for o, f, s, common, k, h in rows:
        lines.append(f"{o} / {f} & \\textit{{{s}}} & {common} & {k} & {h} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "tab_taxonomy.tex").write_text("\n".join(lines), encoding="utf-8")


def overview_figure(out: Path, stem_class: str = "Ficus Religiosa (Peepal Tree)"):
    import cv2
    idx = pd.read_csv(cache_root() / "index.csv")
    r = idx[(idx.class_name == stem_class) & idx.keep].iloc[0]
    img = cv2.cvtColor(cv2.imread(str(cache_root() / "img" / f"{r.stem}.jpg")), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(cache_root() / "mask" / f"{r.stem}.png"), 0) > 127
    vein = cv2.imread(str(cache_root() / "vein" / f"{r.stem}.png"), 0) / 255.0
    G = 96
    m = torch.tensor(cv2.resize(mask.astype(np.float32), (G, G), interpolation=cv2.INTER_AREA) > 0.5).float()[None]
    v = torch.tensor(cv2.resize(vein.astype(np.float32), (G, G), interpolation=cv2.INTER_AREA))[None] * m
    geo = SweepGeometry(K=32, T=16).double()
    c, s = normalise_frame(m.double())
    coords = frame_coords(pixel_coords(G, G, dtype=torch.float64), c, s)
    vb = valuation_increments(m.double(), coords, geo)[0]  # (K,T,3)
    vv = valuation_increments(v.double(), coords, geo)[0]
    k = 0  # direction shown: theta = 0 (sweep along the image x axis)
    theta = 2 * np.pi * k / 32
    # slab bands at full resolution, in the same normalised frame as the module
    mf = torch.tensor(mask, dtype=torch.float64)[None]
    cf, sf = normalise_frame(mf)
    hf = frame_coords(pixel_coords(*mask.shape, dtype=torch.float64), cf, sf)[0].numpy()
    h = hf[..., 0] * np.cos(theta) + hf[..., 1] * np.sin(theta)
    band = np.clip(np.floor((h + 2) / 4 * 16), 0, 15)
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.0), gridspec_kw=dict(width_ratios=[1, 1, 1.3, 1.3]))
    ax[0].imshow(img)
    ax[0].imshow(np.ma.masked_where(~mask, band % 2), cmap="coolwarm", alpha=0.35)
    ax[0].contour(np.where(mask, h, np.nan), levels=np.linspace(-2, 2, 17)[1:-1], colors="k", linewidths=0.5)
    ax[0].set_title(r"(a) sweep slabs, $\theta=0^\circ$", fontsize=9)
    ax[1].imshow(vein * mask, cmap="magma")
    ax[1].set_title("(b) venation field", fontsize=9)
    t = np.arange(1, 17)
    ax[2].bar(t - 0.2, vb[k, :, 0], 0.4, label=r"$\Delta A$ blade", color="#4c78a8")
    ax[2].bar(t + 0.2, vb[k, :, 1], 0.4, label=r"$\Delta P$ blade", color="#f58518")
    ax[2].set_xlabel("slab $k$")
    ax[2].legend(fontsize=7)
    ax[2].set_title(r"(c) valuation increments, blade", fontsize=9)
    ax[3].bar(t - 0.2, vv[k, :, 0], 0.4, label=r"$\Delta A$ vein", color="#54a24b")
    ax[3].bar(t + 0.2, vv[k, :, 1], 0.4, label=r"$\Delta P$ vein", color="#e45756")
    ax[3].set_xlabel("slab $k$")
    ax[3].legend(fontsize=7)
    ax[3].set_title(r"(d) valuation increments, venation", fontsize=9)
    for a in ax[:2]:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out / "fig_overview.pdf")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    taxonomy_table(out)
    overview_figure(out)
    print("ok")
