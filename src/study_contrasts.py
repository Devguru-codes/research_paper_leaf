"""Contrasts of the study (ConvNeXt-T, matched fold-seed runs) -> table + csv.

  python -m src.study_contrasts results/kaggle [leafsnap]
Each contrast is a difference in 20-way field accuracy on matched (fold, seed) runs, for raw and
auto-masked DIMPSAR photographs. Two families, Holm-corrected separately:
  planned (fixed before any run):
    total         strip / slab-only / SVA - plain
    geometry      SVA - slab-only (measured valuations), SVA - strip (object-centred sweep vs image axes)
    global        global - plain        the same valuations fused globally
  post hoc (controls added after a first analysis to test a suspected confound):
    features      patchpool - plain     an extra branch on mid-level patch features (no mask)
    location      maskpool - patchpool  the same branch pooled over the leaf mask
    localisation  strip / slab-only / SVA - maskpool
Uncertainty: Nadeau-Bengio corrected resampled t-test over runs (training variability), a two-way
bootstrap over runs and field species (test-set sampling), and for the geometry / global contrasts a
two one-sided test (TOST) of equivalence within +-MARGIN points on the corrected t distribution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

from src.make_tables import collect
from src.stats import corrected_ttest, holm, select

PLANNED = [("total", "strip", "plain"), ("total", "slabonly", "plain"), ("total", "sva", "plain"),
           ("geometry", "sva", "slabonly"), ("geometry", "sva", "strip"), ("global fusion", "global", "plain")]
POSTHOC = [("features", "patchpool", "plain"), ("location", "maskpool", "patchpool"),
           ("localisation", "strip", "maskpool"), ("localisation", "slabonly", "maskpool"),
           ("localisation", "sva", "maskpool")]
# LeafSnap30 replication: the contrasts fixed in paper/preregistration_leafsnap.md, one Holm family
LEAFSNAP = [("total", "strip", "plain"), ("total", "sva", "plain"), ("location", "maskpool", "patchpool"),
            ("localisation", "strip", "maskpool"), ("geometry", "sva", "strip")]
EQUIV = {"geometry", "global fusion"}
MARGIN = 3.0  # accuracy points
LABEL = {"plain": "plain", "patchpool": "patch pooling", "maskpool": "mask pooling", "strip": "strip",
         "slabonly": "slab-only", "sva": "SVA", "global": "global fusion"}


def correctness(run_dir: str, mode: str):
    z = np.load(Path(run_dir) / f"preds_dimpsar_{mode}.npz")
    lg = np.full_like(z["logits"], -1e9)
    seen = np.unique(z["label"])  # the classes the field set contains (DIMPSAR: OVERLAP; LeafSnap: all 30)
    lg[:, seen] = z["logits"][:, seen]
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


def tost(d: np.ndarray, margin: float, test_train_ratio: float = 0.3) -> float:
    """Equivalence p-value: max of the two one-sided corrected t-tests against -margin and +margin."""
    n = len(d)
    se = np.sqrt((1.0 / n + test_train_ratio) * d.var(ddof=1))
    lo = tdist.sf((d.mean() + margin) / se, n - 1)
    hi = tdist.sf((margin - d.mean()) / se, n - 1)
    return float(max(lo, hi))


def contrast(df, backbone, mode, fam, a, b, exp="main"):
    da, db = select(df, a, backbone, exp), select(df, b, backbone, exp)
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
    return dict(photos=mode, family=fam, a=a, b=b, n_runs=len(idx), diff=per_run.mean(), t_lo=ct["lo"],
                t_hi=ct["hi"], p=ct["p"], wins=int((per_run > 0).sum()), boot_lo=lo, boot_hi=hi,
                p_equiv=tost(per_run, MARGIN) if fam in EQUIV else float("nan"))


def main(root: Path, exp: str = "main", backbone: str = "convnext_tiny"):
    """exp: main | leafsnap (the pre-registered seeds 0-1) | leafsnap_all (seeds 0-4, the later extension)."""
    df = collect(root)
    if exp == "leafsnap":
        df = df[df.seed < 2]
    data_exp = "leafsnap" if exp.startswith("leafsnap") else exp
    parts = []
    sets = ((("pre-registered" if exp == "leafsnap" else "extension"), LEAFSNAP),) if exp.startswith("leafsnap") \
        else (("planned", PLANNED), ("post hoc", POSTHOC))
    for name, fam_list in sets:
        rows = [dict(contrast_set=name, **contrast(df, backbone, mode, fam, a, b, data_exp))
                for mode in ("raw", "masked") for fam, a, b in fam_list]
        part = pd.DataFrame(rows)
        part["p_holm"] = holm(part.p.tolist())
        parts.append(part)
    res = pd.concat(parts, ignore_index=True)
    Path("results").mkdir(exist_ok=True)
    sfx = "" if exp == "main" else f"_{exp}"
    res.to_csv(f"results/study_contrasts{sfx}.csv", index=False)
    print(res.drop(columns=["contrast_set"]).round(4).to_string())
    write_table(res, Path(f"paper/figures/tab_contrasts{sfx}.tex"))


def fmt_p(p):
    if p != p:
        return "--"
    return "$<10^{-4}$" if p < 1e-4 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def write_table(res: pd.DataFrame, path: Path):
    lines = ["\\begin{tabular}{lllcccc}", "\\toprule",
             "Set & Photos & Contrast & $\\Delta$ (runs won) & 95\\% CI, runs & 95\\% CI, runs+species & "
             "Holm $p$ \\\\", "\\midrule"]
    for (cs, mode), g in res.groupby(["contrast_set", "photos"], sort=False):
        for j, (_, r) in enumerate(g.iterrows()):
            eq = f"$^\\dagger$" if r.p_equiv == r.p_equiv and r.p_equiv < 0.05 else ""
            lines.append(f"{cs if j == 0 and mode == 'raw' else ''} & {mode if j == 0 else ''} & "
                         f"{LABEL[r.a]} $-$ {LABEL[r.b]} & {r['diff']:+.1f}{eq} ({r.wins}/{r.n_runs}) & "
                         f"[{r.t_lo:+.1f}, {r.t_hi:+.1f}] & [{r.boot_lo:+.1f}, {r.boot_hi:+.1f}] & {fmt_p(r.p_holm)} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/kaggle"), *sys.argv[2:3])
