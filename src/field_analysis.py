"""Field-photo (DIMPSAR) analyses requested by reviewers.

  python -m src.field_analysis results/kaggle --summary results/kaggle/maskhead/dimpsar/dimpsar_summary.json
Writes paper/figures/tab_dimpsar_map.tex (folder -> species mapping with image counts) and
paper/figures/tab_field_species.tex (per-species 20-way raw accuracy, ConvNeXt-T, mean over runs), and
results/field_per_species.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.crossdomain import DIMPSAR_TO_OURS, OVERLAP
from src.data.taxonomy import CLASSES, TAXONOMY
from src.make_tables import collect

MODELS = ["plain", "global", "strip", "slabonly", "sva"]


def per_species(df: pd.DataFrame, backbone: str, split: str = "raw") -> pd.DataFrame:
    rows = []
    ov = np.array(OVERLAP)
    for _, r in df[(df.backbone == backbone) & (df.exp == "main") & (df.model.isin(MODELS))].iterrows():
        f = Path(r.run_dir) / f"preds_dimpsar_{split}.npz"
        if not f.exists():
            continue
        p = np.load(f)
        pred = ov[p["logits"][:, ov].argmax(1)]  # 20-way restricted, as in the paper
        for c in np.unique(p["label"]):
            sel = p["label"] == c
            rows.append(dict(model=r.model, fold=r.fold, seed=r.seed, species=CLASSES[c], n=int(sel.sum()),
                             acc=float((pred[sel] == c).mean())))
    return pd.DataFrame(rows)


def main(root: Path, summary: Path, out: Path):
    df = collect(root)
    counts = json.loads(summary.read_text())["per_class"]
    lines = ["\\begin{tabular}{lllr}", "\\toprule", "DIMPSAR folder & Species & Family & Images used \\\\", "\\midrule"]
    for k, v in sorted(DIMPSAR_TO_OURS.items(), key=lambda kv: TAXONOMY[kv[1]][3]):
        o, fam, g, s = TAXONOMY[v]
        lines.append(f"{k.replace('_', ' ')} & \\textit{{{s}}} & {fam} & {counts['sum'][k]} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "tab_dimpsar_map.tex").write_text("\n".join(lines), encoding="utf-8")

    ps = per_species(df, "convnext_tiny")
    ps.to_csv("results/field_per_species.csv", index=False)
    t = ps.groupby(["species", "model"]).acc.mean().unstack()[MODELS] * 100
    t["n"] = ps.groupby("species").n.first()
    t = t.sort_values("sva")
    lines = ["\\begin{tabular}{lr" + "c" * len(MODELS) + "}", "\\toprule",
             "Species & $n$ & Plain & Global & Strip & Slab-only & SVA \\\\", "\\midrule"]
    for sp, r in t.iterrows():
        cells = " & ".join(f"{r[m]:.0f}" for m in MODELS)
        lines.append(f"\\textit{{{TAXONOMY[sp][3]}}} & {int(r.n)} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "tab_field_species.tex").write_text("\n".join(lines), encoding="utf-8")
    better = (t["sva"] > t["plain"]).sum()
    print(t.round(1).to_string())
    print(f"SVA > plain on {better}/{len(t)} species; SVA > strip on {(t['sva'] > t['strip']).sum()}/{len(t)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--summary", default="results/kaggle/maskhead/dimpsar/dimpsar_summary.json")
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    main(Path(a.root), Path(a.summary), Path(a.out))
