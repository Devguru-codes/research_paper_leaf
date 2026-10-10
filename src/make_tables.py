"""Collect all run outputs under a results root into one flat table (+ grouped mean/std).

  python -m src.make_tables results/kaggle/pilot            -> prints summary, writes summary.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (int, float, bool, np.floating)) or v is None:
            out[key] = v
    return out


def model_name(cfg: dict) -> str:
    """Paper model name from (variant, token): plain | global | strip | slabonly | sva | euler | hadwiger | ..."""
    v, tok = cfg["variant"], cfg.get("token")
    if v == "slabstat" and tok == "ap":
        return "sva"
    if v == "esa":
        return {"none": "slabonly", None: "euler"}.get(tok, tok)
    return v


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for mj in sorted(root.rglob("metrics.json")):
        run = mj.parent
        m = json.loads(mj.read_text())
        cfg = m["cfg"]
        row = dict(run_dir=str(run), exp=cfg["exp"], backbone=cfg["backbone"], model=model_name(cfg), variant=cfg["variant"],
                   tag=cfg.get("tag", ""), rot_aug=cfg.get("rot_aug", True),
                   protocol=cfg["protocol"], fold=cfg["fold"], seed=cfg["seed"], train_min=m["train_time_s"] / 60,
                   params_fusion=m.get("params_fusion", 0), leak_test_in_train=m["leakage"]["test_in_train_leaf_group"])
        row.update(flatten(m["test"], "test."))
        for ej in run.glob("eval_*.json"):
            e = json.loads(ej.read_text())
            name = ej.stem.replace("eval_", "")
            if name == "rotation":
                e = {k: v for k, v in e.items() if k != "per_angle"}
            row.update(flatten(e, name + "."))
        rows.append(row)
    return pd.DataFrame(rows)


KEY = ["test.acc", "test.bal_acc", "test.macro_f1", "test.ece", "rotation.mean_acc", "rotation.min_acc",
       "rotation.consistency", "corruptions.mean_corrupted_acc", "corruptions.desaturate/3",
       "corruptions.background/3", "cueconflict.all.shape_bias", "cueconflict.moderate_stretch.shape_bias",
       "topodestroy.blade_holes.mean_kl", "topodestroy.blade_bites.mean_kl",
       "topodestroy.vein_sever.mean_kl", "topodestroy.vein_control.mean_kl",
       "crossdomain.masked.acc20", "crossdomain.raw.acc20", "train_min"]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in KEY if c in df]
    g = df.groupby(["exp", "protocol", "backbone", "model", "tag"])[cols]
    mean, std, n = g.mean(), g.std(), g.size().rename("n")
    out = mean.copy()
    for c in cols:
        out[c] = [f"{m:.4f}" + (f"±{s:.4f}" if np.isfinite(s) else "") for m, s in zip(mean[c], std[c])]
    return out.join(n)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    df = collect(root)
    df.to_csv(root / "runs.csv", index=False)
    s = summarize(df)
    s.to_csv(root / "summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(s.T)
