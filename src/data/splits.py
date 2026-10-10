"""Cross-validation splits.

protocol='grouped'      (ours)  exact duplicates removed; re-shot leaf groups never split;
                                stratified by species. 5 outer folds; inside each training
                                part one group-aware stratified holdout (~1/6) is the
                                calibration/monitoring set (no model selection on it).
protocol='random_dups'  (P1)    all 1,835 files, plain stratified 5-fold (as in prior work).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def make_split(df: pd.DataFrame, fold: int, protocol: str = "grouped", n_folds: int = 5,
               seed: int = 0, n_inner: int = 6):
    """Return dict of index arrays (into df) for train / cal / test."""
    if protocol == "grouped":
        d = df[df.keep].copy()
        outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        tr, te = list(outer.split(d, d.label, d.leaf_group))[fold]
        dtr = d.iloc[tr]
        inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed + 100 + fold)
        itr, ical = next(inner.split(dtr, dtr.label, dtr.leaf_group))
        return dict(train=dtr.index[itr].to_numpy(), cal=dtr.index[ical].to_numpy(), test=d.index[te].to_numpy())
    if protocol == "random_dups":
        outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        tr, te = list(outer.split(df, df.label))[fold]
        dtr = df.iloc[tr]
        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + 100 + fold)
        itr, ical = next(inner.split(dtr, dtr.label))
        return dict(train=dtr.index[itr].to_numpy(), cal=dtr.index[ical].to_numpy(), test=df.index[te].to_numpy())
    raise ValueError(protocol)


def leakage_report(df: pd.DataFrame, split: dict) -> dict:
    """Count test images whose leaf group / exact-duplicate group also appears in train."""
    tr, te = df.loc[split["train"]], df.loc[split["test"]]
    return dict(test_in_train_leaf_group=int(te.leaf_group.isin(tr.leaf_group).sum()),
                test_in_train_exact=int(te.exact_group.isin(tr.exact_group).sum()))
