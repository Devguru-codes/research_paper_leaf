"""Pre-specified contrasts of the study (ConvNeXt-T, 5 folds x 5 seeds) -> table + csv.

  python -m src.study_contrasts results/kaggle
Each contrast is a difference in 20-way field accuracy on matched (fold, seed) runs, for raw and
auto-masked DIMPSAR photographs:
  location      maskpool - plain      knowing where the leaf is
  localisation  strip / slab-only / SVA - maskpool
  geometry      SVA - slab-only (measured valuations), SVA - strip (object-centred sweep vs image axes)
  global        global - plain        the same valuations fused globally
Uncertainty is reported two ways: the Nadeau-Bengio corrected resampled t-test over runs (training
variability) with Holm correction over the whole family, and a two-way bootstrap that resamples runs
and field species (clusters of images) jointly, which also covers test-set sampling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.crossdomain import OVERLAP
from src.make_tables import collect
from src.stats import corrected_ttest, holm, select

CONTRASTS = [("location", "maskpool", "plain"), ("localisation", "strip", "maskpool"),
             ("localisation", "slabonly", "maskpool"), ("localisation", "sva", "maskpool"),
             ("geometry", "sva", "slabonly"), ("geometry", "sva", "strip"), ("global fusion", "global", "plain")]
LABEL = {"plain": "plain", "maskpool": "mask pooling", "strip": "strip", "slabonly": "slab-only", "sva": "SVA",
         "global": "global fusion"}


def correctness(run_dir: str, mode: str):
    z = np.load(Path(run_dir) / f"preds_dimpsar_{mode}.npz")
    lg = np.full_like(z["logits"], -1e9)
    lg[:, OVERLAP] = z["logits"][:, OVERLAP]
    return (lg.argmax(1) == z["label"]).astype(float), z["label"]


def two_way_bootstrap(D: np.ndarray, species: np.ndarray, B: int = 4000, seed: int = 0):
    """D (runs x images) paired differences; resample runs and species clusters jointly."""
    rng = np.random.default_rng(seed)
    sp = np.unique(species)
    S = np.stack([D[:, species == s].sum(1) for s in sp], 1)  # runs x species
    N = np.array([(species == s).sum() for s in sp], float)
    stats = np.empty(B)
    for b in range(B):
        r = rng.integers(0, D.shape[0], D.shape[0])
        s = rng.integers(0, len(sp), len(sp))
        stats[b] = S[np.ix_(r, s)].sum() / (len(r) * N[s].sum())
    return np.percentile(stats, [2.5, 97.5])


def main(root: Path, backbone: str = "convnext_tiny"):
    df = collect(root)
    rows = []
    for mode in ("raw", "masked"):
        for fam, a, b in CONTRASTS:
            da, db = select(df, a, backbone, "main"), select(df, b, backbone, "main")
            idx = da.index.intersection(db.index)
            Ca, Cb, lab = [], [], None
            for k in idx:
                ca, la = correctness(da.loc[k, "run_dir"], mode)
                cb, lb = correctness(db.loc[k, "run_dir"], mode)
                assert (la == lb).all() and (lab is None or (la == lab).all())
                lab = la
                Ca.append(ca)
                Cb.append(cb)
            D = 100 * (np.array(Ca) - np.array(Cb))
            per_run = D.mean(1)
            ct = corrected_ttest(per_run)
            lo, hi = two_way_bootstrap(D, lab)
            rows.append(dict(photos=mode, family=fam, a=a, b=b, n_runs=len(idx), diff=per_run.mean(),
                             t_lo=ct["lo"], t_hi=ct["hi"], p=ct["p"], wins=int((per_run > 0).sum()),
                             boot_lo=lo, boot_hi=hi))
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res.p.tolist())
    Path("results").mkdir(exist_ok=True)
    res.to_csv("results/study_contrasts.csv", index=False)
    print(res.round(4).to_string())
    write_table(res, Path("paper/figures/tab_contrasts.tex"))


def fmt_p(p):
    return "$<10^{-4}$" if p < 1e-4 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def write_table(res: pd.DataFrame, path: Path):
    lines = ["\\begin{tabular}{llcccc}", "\\toprule",
             "Photos & Contrast & $\\Delta$ (points) & 95\\% CI, runs & 95\\% CI, runs+species & Holm $p$ \\\\",
             "\\midrule"]
    for mode, g in res.groupby("photos", sort=False):
        for j, (_, r) in enumerate(g.iterrows()):
            lines.append(f"{mode if j == 0 else ''} & {LABEL[r.a]} $-$ {LABEL[r.b]} & {r['diff']:+.1f} "
                         f"({r.wins}/{r.n_runs}) & [{r.t_lo:+.1f}, {r.t_hi:+.1f}] & [{r.boot_lo:+.1f}, {r.boot_hi:+.1f}] & "
                         f"{fmt_p(r.p_holm)} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/kaggle"))
