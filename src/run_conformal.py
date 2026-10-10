"""Offline conformal analysis (P8) over saved predictions of every main run.

  python -m src.run_conformal results/kaggle --out results/conformal.csv
For each run: calibrate on its held-out calibration split (preds_cal.npz), then evaluate sets on
the in-domain test fold and, without any guarantee, on the DIMPSAR field photos (masked, raw).
Two calibrations: 'standard' (one alpha=0.10 for all classes, still Mondrian by tier so it is
comparable) and 'hazard' (alpha 0.10 / 0.05 / 0.01 for tiers 0 / 1 / 2).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.conformal import calibrate, evaluate_sets, predict_sets
from src.make_tables import collect

SCHEMES = {"standard": {0: 0.10, 1: 0.10, 2: 0.10}, "hazard": {0: 0.10, 1: 0.05, 2: 0.01}}


def main(root: Path, out: Path):
    df = collect(root)
    df = df[(df.exp == "main") & (df.protocol == "grouped") & (df.seed == 0)]
    rows = []
    for _, r in df.iterrows():
        d = Path(r.run_dir)
        cal = np.load(d / "preds_cal.npz")
        for scheme, alphas in SCHEMES.items():
            q = calibrate(cal["logits"], cal["label"], alphas, seed=int(r.fold))
            for split in ("test", "dimpsar_masked", "dimpsar_raw"):
                f = d / f"preds_{split}.npz"
                if not f.exists():
                    continue
                p = np.load(f)
                ev = evaluate_sets(predict_sets(p["logits"], q, seed=int(r.fold) + 7), p["label"])
                rows.append(dict(backbone=r.backbone, model=r.model, fold=r.fold, scheme=scheme, split=split,
                                 n_cal=len(cal["label"]), **ev))
    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)
    cols = ["coverage", "mean_size", "singleton", "mean_lca_depth", "coverage_tier0", "coverage_tier1",
            "coverage_tier2", "toxic_flag_rate"]
    cols = [c for c in cols if c in res]
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(res.groupby(["split", "scheme", "backbone", "model"])[cols].mean().round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="results/conformal.csv")
    a = ap.parse_args()
    main(Path(a.root), Path(a.out))
