"""Paper figures and LaTeX tables, generated only from run outputs.

  python -m src.make_figures results/kaggle --out paper/figures
Writes:
  fig_crossdomain.pdf   field-photo accuracy (masked / raw), per backbone and model
  fig_rotation.pdf      accuracy vs rotation angle, with and without rotation augmentation
  fig_corruptions.pdf   MedLeaf-C accuracy per corruption family (severity-averaged)
  tab_main.tex          main table (mean +- sd over runs) per backbone and model
  tab_ablations.tex     ablation table (ConvNeXt-T)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.make_tables import collect  # noqa: E402

MODELS = ["plain", "global", "strip", "slabonly", "sva"]
LABEL = {"plain": "Plain", "global": "Global fusion", "strip": "Strip pooling", "slabonly": "Slab-only",
         "sva": "SVA (ours)", "euler": "Euler token", "hadwiger": "Hadwiger token"}
COLOR = {"plain": "#9e9e9e", "global": "#7aa6c2", "strip": "#c2a77a", "slabonly": "#9fc27a", "sva": "#c0392b"}
BACKBONES = [("convnext_tiny", "ConvNeXt-T"), ("vit_small", "ViT-S/16"), ("bioclip2", "BioCLIP 2")]
FAMILIES = ["gaussian_noise", "gaussian_blur", "jpeg", "low_resolution", "darken", "contrast", "hue_shift",
            "desaturate", "occlusion", "background"]


def main_runs(df):
    return df[(df.exp == "main") & (df.protocol == "grouped") & (df.rot_aug == True)]  # noqa: E712


def fig_crossdomain(df, out):
    d = main_runs(df)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, mode in zip(axes, ["masked", "raw"]):
        col = f"crossdomain.{mode}.acc20"
        x = np.arange(len(BACKBONES))
        w = 0.16
        for i, m in enumerate(MODELS):
            vals = [d[(d.backbone == b) & (d.model == m)][col].astype(float) if col in d else pd.Series(dtype=float)
                    for b, _ in BACKBONES]
            mu = [100 * v.mean() if len(v) else np.nan for v in vals]
            sd = [100 * v.std() if len(v) > 1 else 0 for v in vals]
            ax.bar(x + (i - 2) * w, mu, w, yerr=sd, color=COLOR[m], label=LABEL[m], capsize=2)
        ax.set_xticks(x, [n for _, n in BACKBONES])
        ax.set_title(f"Field photographs ({mode})")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("20-way accuracy (%)")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_crossdomain.pdf")
    plt.close(fig)


def per_angle(run_dir):
    p = Path(run_dir) / "eval_rotation.json"
    if not p.exists():
        return None
    pa = json.loads(p.read_text())["per_angle"]
    return pd.Series({int(k): v["acc"] for k, v in pa.items()}).sort_index()


def fig_rotation(df, out):
    d = df[(df.backbone == "convnext_tiny") & (df.exp.isin(["main", "norot"])) & (df.protocol == "grouped")
           & (df.seed == 0)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, rot in zip(axes, [True, False]):
        for m in MODELS:
            curves = [c for c in (per_angle(r) for r in d[(d.model == m) & (d.rot_aug == rot)].run_dir) if c is not None]
            if not curves:
                continue
            c = pd.concat(curves, axis=1).mean(1) * 100
            ax.plot(c.index, c.values, marker="o", ms=3, color=COLOR[m], label=LABEL[m])
        ax.set_title("trained with rotation aug." if rot else "trained without rotation aug.")
        ax.set_xlabel("rotation (degrees)")
        ax.set_xticks(range(0, 360, 60))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy (%)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_rotation.pdf")
    plt.close(fig)


def fig_corruptions(df, out):
    d = main_runs(df)
    d = d[d.backbone == "convnext_tiny"]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    x = np.arange(len(FAMILIES))
    w = 0.16
    for i, m in enumerate(MODELS):
        g = d[d.model == m]
        mu = []
        for f in FAMILIES:
            cols = [f"corruptions.{f}/{s}" for s in (1, 2, 3) if f"corruptions.{f}/{s}" in g]
            mu.append(100 * g[cols].astype(float).mean(1).mean() if cols and len(g) else np.nan)
        ax.bar(x + (i - 2) * w, mu, w, color=COLOR[m], label=LABEL[m])
    ax.set_xticks(x, [f.replace("_", " ") for f in FAMILIES], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(30, 101)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "fig_corruptions.pdf")
    plt.close(fig)


def fmt(v, scale=100, nd=1):
    v = v.dropna().astype(float)
    if not len(v):
        return "--"
    s = f"{scale * v.mean():.{nd}f}"
    return s + (f"\\,$\\pm$\\,{scale * v.std():.{nd}f}" if len(v) > 1 else "")


MAIN_COLS = [("test.acc", "Clean"), ("rotation.min_acc", "Rot.\\ min"), ("corruptions.mean_corrupted_acc", "MedLeaf-C"),
             ("cueconflict.all.shape_bias", "Shape bias"), ("crossdomain.masked.acc20", "Field (mask)"),
             ("crossdomain.raw.acc20", "Field (raw)")]


def tab_main(df, out):
    d = main_runs(df)
    lines = ["\\begin{tabular}{ll" + "c" * len(MAIN_COLS) + "}", "\\toprule",
             "Backbone & Model & " + " & ".join(n for _, n in MAIN_COLS) + " \\\\", "\\midrule"]
    for b, bn in BACKBONES:
        rows = [m for m in MODELS if len(d[(d.backbone == b) & (d.model == m)])]
        for j, m in enumerate(rows):
            g = d[(d.backbone == b) & (d.model == m)]
            cells = [fmt(g[c]) if c in g else "--" for c, _ in MAIN_COLS]
            lines.append((bn if j == 0 else "") + f" & {LABEL[m]} & " + " & ".join(cells) + " \\\\")
        if rows:
            lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    (out / "tab_main.tex").write_text("\n".join(lines))


ABL_COLS = [("test.acc", "Clean"), ("rotation.min_acc", "Rot.\\ min"), ("crossdomain.masked.acc20", "Field (mask)"),
            ("crossdomain.raw.acc20", "Field (raw)")]


def tab_ablations(df, out):
    d = df[(df.backbone == "convnext_tiny") & (df.protocol == "grouped") & (df.seed == 0) & (df.fold < 3)
           & (df.rot_aug == True)]  # noqa: E712
    ref = d[(d.exp == "main") & (d.model == "sva")]
    abl = d[d.exp == "ablations"]
    names = {"ap_globalattn": "attention over all leaf patches", "ap_kappa1": "softer relaxation ($\\kappa{=}1$)",
             "ap_kappa16": "sharper relaxation ($\\kappa{=}16$)", "ap_permw": "weights permuted within the leaf",
             "ap_veinonly": "venation field only", "ap_bladeonly": "blade only", "ap_K16T8": "coarse sweep ($K{=}16,T{=}8$)",
             "eu": "Euler token", "had": "Hadwiger token (area, perimeter, Euler)", "ap_unitw": "unit cell weights"}
    groups = [("SVA (full)", ref)] + sorted(((names.get(t, t), g) for t, g in abl.groupby("tag")),
                                            key=lambda x: -x[1]["crossdomain.raw.acc20"].mean())
    lines = ["\\begin{tabular}{l" + "c" * len(ABL_COLS) + "}", "\\toprule",
             "Variant & " + " & ".join(n for _, n in ABL_COLS) + " \\\\", "\\midrule"]
    for name, g in groups:
        lines.append(name + " & " + " & ".join(fmt(g[c]) if c in g else "--" for c, _ in ABL_COLS)
                     + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "tab_ablations.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = collect(Path(a.root))
    for f in (fig_crossdomain, fig_rotation, fig_corruptions, tab_main, tab_ablations):
        try:
            f(df, out)
            print("ok", f.__name__)
        except Exception as e:  # keep going so partial results still produce the other outputs
            print("skip", f.__name__, repr(e))
