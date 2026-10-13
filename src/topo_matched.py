"""Topology-destruction probe with perceptual matching (reviewer request).

  python -m src.topo_matched results/kaggle
For each model, χ-changing edits (blade holes) are compared with χ-preserving edits (margin bites) only
within LPIPS strata that contain both kinds (10 quantile bins of the pooled LPIPS); the response is the
KL divergence between clean and edited predictions. Reports the per-stratum-weighted mean log-ratio
log(KL_holes / KL_bites) with a bootstrap 95% CI, plus the LPIPS balance after matching. Vein edits are
reported as unmatched when their LPIPS ranges do not overlap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.make_tables import collect


def matched_logratio(d: pd.DataFrame, bins: int = 10, boot: int = 1000, seed: int = 0):
    d = d.copy()
    d["stratum"] = pd.qcut(d.lpips, bins, labels=False, duplicates="drop")
    rows = []
    for s, g in d.groupby("stratum"):
        a, b = g[g.chi_changing], g[~g.chi_changing]
        if len(a) >= 5 and len(b) >= 5:
            rows.append((s, a, b))
    if not rows:
        return None
    w = np.array([min(len(a), len(b)) for _, a, b in rows], float)
    w /= w.sum()
    eps = 1e-8

    def stat(sample_fn):
        return float(sum(wi * (np.log(sample_fn(a).kl.mean() + eps) - np.log(sample_fn(b).kl.mean() + eps))
                         for wi, (_, a, b) in zip(w, rows)))

    rng = np.random.default_rng(seed)
    est = stat(lambda x: x)
    bs = [stat(lambda x: x.sample(len(x), replace=True, random_state=int(rng.integers(1 << 31)))) for _ in range(boot)]
    a_all = pd.concat([a for _, a, _ in rows])
    b_all = pd.concat([b for _, _, b in rows])
    return dict(log_ratio=est, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                strata=len(rows), lpips_holes=float(a_all.lpips.median()), lpips_bites=float(b_all.lpips.median()),
                lpips_balance_p=float(mannwhitneyu(a_all.lpips, b_all.lpips).pvalue))


def write_table(res: pd.DataFrame, path: Path):
    names = {"convnext_tiny": "ConvNeXt-T", "vit_small": "ViT-S/16", "bioclip2": "BioCLIP 2"}
    lab = {"plain": "Plain", "global": "Global fusion", "strip": "Strip pooling", "slabonly": "Slab-only",
           "sva": "SVA", "eu": "Euler token (ablation)"}
    lines = ["\\begin{tabular}{llc}", "\\toprule", "Backbone & Model & log-ratio [95\\% CI] \\\\", "\\midrule"]
    for bb in names:
        g = res[res.backbone == bb].set_index("model")
        ms = [m for m in lab if m in g.index]
        for j, m in enumerate(ms):
            x = g.loc[m]
            lines.append(f"{names[bb] if j == 0 else ''} & {lab[m]} & {x.log_ratio:+.2f} [{x.lo:+.2f}, {x.hi:+.2f}] \\\\")
        if ms:
            lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(root: Path):
    df = collect(root)
    out = []
    for name, sel in [("main", (df.exp == "main") & (df.seed == 0)), ("ablations", df.exp == "ablations")]:
        for _, r in df[sel].iterrows():
            f = Path(r.run_dir) / "eval_topodestroy_samples.csv"
            if f.exists():
                d = pd.read_csv(f)
                d["model"] = r.model if name == "main" else r.tag
                d["backbone"] = r.backbone
                out.append(d)
    d = pd.concat(out)
    d["chi_changing"] = d.chi_changing.astype(bool)
    res = []
    for (bb, m), g in d[d.kind == "blade"].groupby(["backbone", "model"]):
        r = matched_logratio(g)
        if r:
            res.append(dict(backbone=bb, model=m, **r))
    res = pd.DataFrame(res)
    res.to_csv("results/topo_matched.csv", index=False)
    print(res.round(3).to_string())
    write_table(res, Path("paper/figures/tab_topo.tex"))
    v = d[d.kind == "vein"]
    a, b = v[v.chi_changing].lpips, v[~v.chi_changing].lpips
    print(f"vein edits: LPIPS sever median {a.median():.4f} vs control {b.median():.4f}; "
          f"overlap fraction of sever edits within control range: {(a <= b.max()).mean():.2f}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/kaggle"))
