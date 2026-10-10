"""Paired statistics between models (paper names: plain | global | strip | slabonly | sva | ...).

  python -m src.stats results/kaggle --exp main --a sva --b plain [--backbone convnext_tiny] [--norot]
- Wilcoxon signed-rank over matched (fold, seed) pairs for each metric
- McNemar (exact binomial) on pooled per-image test predictions
- Topology-destruction regression: log KL ~ chi_changing + visual distance, per model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from src.make_tables import collect

METRICS = ["test.acc", "test.bal_acc", "test.macro_f1", "test.ece", "rotation.mean_acc", "rotation.min_acc",
           "corruptions.mean_corrupted_acc", "cueconflict.all.shape_bias", "crossdomain.masked.acc20",
           "crossdomain.raw.acc20"]


def select(df, model, backbone, exp, rot_aug=True, protocol="grouped"):
    d = df[(df.model == model) & (df.backbone == backbone) & (df.exp == exp) & (df.rot_aug == rot_aug)
           & (df.protocol == protocol)]
    return d.drop_duplicates(["fold", "seed"], keep="last").set_index(["fold", "seed"])


def paired(df: pd.DataFrame, a: str, b: str, backbone: str, exp: str = "main", rot_aug: bool = True) -> dict:
    da, db = select(df, a, backbone, exp, rot_aug), select(df, b, backbone, exp, rot_aug)
    idx = da.index.intersection(db.index)
    out = dict(a=a, b=b, backbone=backbone, n_pairs=len(idx))
    for m in METRICS:
        if m in da and m in db:
            x, y = da.loc[idx, m].astype(float), db.loc[idx, m].astype(float)
            ok = x.notna() & y.notna()
            if ok.sum() >= 2:
                d = (x[ok] - y[ok]).to_numpy()
                p = wilcoxon(d).pvalue if np.any(d != 0) and ok.sum() >= 5 else float("nan")
                out[m] = dict(n=int(ok.sum()), mean_diff=float(d.mean()), wins=int((d > 0).sum()),
                              losses=int((d < 0).sum()), p=float(p))
    n01 = n10 = 0
    for key in idx:
        pa = np.load(Path(da.loc[key, "run_dir"]) / "preds_test.npz")
        pb = np.load(Path(db.loc[key, "run_dir"]) / "preds_test.npz")
        assert (pa["index"] == pb["index"]).all()
        ca, cb = pa["logits"].argmax(1) == pa["label"], pb["logits"].argmax(1) == pb["label"]
        n10 += int((ca & ~cb).sum())
        n01 += int((~ca & cb).sum())
    out["mcnemar"] = dict(a_only=n10, b_only=n01,
                          p=float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else float("nan"))
    return out


def topo_regression(df: pd.DataFrame, backbone: str, exp: str = "main") -> dict:
    """Per model and edit family: OLS of log KL on [1, chi_changing, visual distance], averaged over runs."""
    res = {}
    for _, r in df[(df.backbone == backbone) & (df.exp == exp) & (df.rot_aug == True)].iterrows():  # noqa: E712
        csv = Path(r.run_dir) / "eval_topodestroy_samples.csv"
        if not csv.exists():
            continue
        d = pd.read_csv(csv)
        vis = "lpips" if "lpips" in d and d["lpips"].notna().all() else "pixel_rmse"
        for fam, g in d.groupby("family" if "family" in d else "kind"):
            if g.chi_changing.nunique() < 2:
                continue
            X = np.column_stack([np.ones(len(g)), g.chi_changing.astype(float), g[vis].astype(float)])
            yv = np.log(g.kl.astype(float).clip(lower=0) + 1e-8)
            beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
            res.setdefault(r.model, {}).setdefault(fam, []).append(float(beta[1]))
    return {m: {f: dict(beta_chi_mean=float(np.mean(v)), beta_chi_sd=float(np.std(v)), n=len(v)) for f, v in fams.items()}
            for m, fams in res.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--exp", default="main")
    ap.add_argument("--a", default="sva")
    ap.add_argument("--b", nargs="+", default=["plain", "global", "strip", "slabonly"])
    ap.add_argument("--backbone", default="convnext_tiny")
    ap.add_argument("--norot", action="store_true")
    a = ap.parse_args()
    df = collect(Path(a.root))
    out = dict(paired=[paired(df, a.a, b, a.backbone, a.exp, not a.norot) for b in a.b],
               topo=topo_regression(df, a.backbone, a.exp))
    print(json.dumps(out, indent=1))