"""Per-species field accuracy, gains and masker exclusions (appendix table).

  python -m src.field_failures
Needs results/field_per_species.csv (python -m src.field_analysis results/kaggle) and the DIMPSAR build
index (dimpsar_index.csv, written by src.crossdomain build). Writes paper/figures/tab_field_failures.tex
and prints the rank correlation between the strip-pooling gain and the leaf's share of the crop.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from src.data.taxonomy import CLASSES


def main(out: Path = Path("paper/figures/tab_field_failures.tex")):
    f = pd.read_csv("results/field_per_species.csv")
    acc = f.groupby(["species", "model"]).acc.mean().unstack() * 100
    idx = pd.read_csv(glob.glob("results/kaggle/maskhead/**/dimpsar_index.csv", recursive=True)[0])
    idx["species"] = [CLASSES[label] for label in idx.label]
    t = pd.DataFrame({"photos": idx.groupby("species").size(),
                      "excluded": idx.groupby("species").usable.apply(lambda s: int((~s).sum())),
                      "leaf": idx[idx.usable].groupby("species").mask_frac.median() * 100,
                      "plain": acc.plain, "strip": acc.strip, "sva": acc.sva}).sort_values("plain")
    rho = spearmanr(t.strip - t.plain, t.leaf)
    print(t.round(1).to_string())
    print(f"strip gain vs leaf share: Spearman rho={rho.statistic:.2f}, p={rho.pvalue:.2f}; "
          f"excluded {int(t.excluded.sum())}/{int(t.photos.sum())}")
    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
             "Species & Photos & Excluded & Leaf (\\%) & Plain & Strip & SVA \\\\", "\\midrule"]
    for s, r in t.iterrows():
        genus, *rest = s.split(" (")[0].split(" ")
        name = " ".join([genus] + [w.lower() for w in rest])
        lines.append(f"\\textit{{{name}}} & {int(r.photos)} & {int(r.excluded)} & {r.leaf:.0f} & {r.plain:.1f} & "
                     f"{r.strip:.1f} & {r.sva:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
