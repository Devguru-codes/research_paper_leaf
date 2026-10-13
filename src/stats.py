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


def corrected_ttest(d: np.ndarray, test_train_ratio: float = 0.3) -> dict:
    """Nadeau & Bengio (2003) corrected resampled t-test for paired differences from (repeated) k-fold CV.
    The variance is inflated by n_test/n_train to account for overlapping training sets; df = n - 1.
    Returns mean, corrected 95% CI and two-sided p."""
    from scipy.stats import t as tdist
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return dict(mean=float(d.mean()) if n else float("nan"), lo=float("nan"), hi=float("nan"), p=float("nan"))
    var = d.var(ddof=1)
    se = np.sqrt((1.0 / n + test_train_ratio) * var)
    if se == 0:
        return dict(mean=float(d.mean()), lo=float(d.mean()), hi=float(d.mean()), p=0.0 if d.mean() != 0 else 1.0)
    tval = d.mean() / se
    q = tdist.ppf(0.975, n - 1)
    return dict(mean=float(d.mean()), lo=float(d.mean() - q * se), hi=float(d.mean() + q * se),
                p=float(2 * tdist.sf(abs(tval), n - 1)))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaNs are passed through)."""
    idx = [i for i, p in enumerate(pvals) if p == p]
    order = sorted(idx, key=lambda i: pvals[i])
    adj, running = [float("nan")] * len(pvals), 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - r) * pvals[i]))
        adj[i] = running
    return adj


def select(df, model, backbone, exp, rot_aug=True, protocol="grouped", strong_aug=False):
    d = df[(df.model == model) & (df.backbone == backbone) & (df.exp == exp) & (df.rot_aug == rot_aug)
           & (df.protocol == protocol)]
    if "strong_aug" in d:
        d = d[d.strong_aug.fillna(False).astype(bool) == strong_aug]
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
                ct = corrected_ttest(d)
                out[m] = dict(n=int(ok.sum()), mean_diff=float(d.mean()), wins=int((d > 0).sum()),
                              losses=int((d < 0).sum()), p=float(p), p_corr_t=ct["p"], ci95=[ct["lo"], ct["hi"]])
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