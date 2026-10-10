"""Gate report for the re-pilot: merges pilot3 (fold0 plain / esa-euler) with pilot2r runs.

  python -m src.gate_report results/kaggle/pilot3 results/kaggle/pilot2r
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.make_tables import collect

COLS = {
    "test.acc": "clean",
    "rotation.mean_acc": "rot_mean",
    "rotation.min_acc": "rot_min",
    "corruptions.mean_corrupted_acc": "medleafC",
    "cueconflict.all.shape_bias": "shape_bias",
    "crossdomain.masked.acc20": "field_masked20",
    "crossdomain.raw.acc20": "field_raw20",
}


def topo_ratio(run_dirs: list[Path]) -> tuple[float, float]:
    """mean over runs of KL(holes)/KL(bites) and KL(sever)/KL(control)."""
    rb, rv = [], []
    for d in run_dirs:
        f = d / "eval_topodestroy_samples.csv"
        if f.exists():
            g = pd.read_csv(f).groupby("edit").kl.mean()
            rb.append(g["blade_holes"] / max(g["blade_bites"], 1e-9))
            rv.append(g["vein_sever"] / max(g["vein_control"], 1e-9))
    return (float(np.mean(rb)) if rb else np.nan, float(np.mean(rv)) if rv else np.nan)


def main(roots):
    frames = []
    for r in roots:
        df = collect(Path(r))
        df["root"] = r
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # pilot3: keep only plain and esa(euler) as the reused fold-0 runs
    p3 = df.root == roots[0]
    df = df[~p3 | df.variant.isin(["plain", "esa"])].copy()
    df.loc[p3 & (df.variant == "esa"), "tag"] = "eu"
    df["arm"] = np.where(df.tag.str.contains("norot"), "no-rot-aug", "rot-aug")
    df["model"] = df.variant + np.where(df.tag.str.replace("_norot", "").str.replace("norot", "") != "",
                                        "/" + df.tag.str.replace("_norot", "").str.replace("norot", ""), "")
    rows = []
    for (arm, model), g in df.groupby(["arm", "model"]):
        row = dict(arm=arm, model=model, n=len(g), folds=",".join(map(str, sorted(g.fold))))
        for c, n in COLS.items():
            if c in g:
                row[n] = g[c].mean()
        dirs = [Path(r) for r in []]
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["arm", "model"])
    # topology ratios from sample CSVs
    tb, tv = [], []
    for _, row in out.iterrows():
        sel = df[(df.arm == row.arm) & (df.model == row.model)]
        dirs = []
        for _, r in sel.iterrows():
            name = f"{r.backbone}_{r.variant}" + (f"_{r.tag}" if r.tag and r.root != roots[0] else "")
            hits = list(Path(r.root).rglob(f"{name}/fold{r.fold}_seed{r.seed}"))
            dirs += hits[:1]
        b, v = topo_ratio(dirs)
        tb.append(b)
        tv.append(v)
    out["topo_blade_ratio"], out["topo_vein_ratio"] = tb, tv
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(out.to_string(index=False))
    out.to_csv(Path(roots[-1]) / "gate_report.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
