"""Hazard-aware hierarchical conformal prediction (secondary deployment layer).

Split-conformal with APS non-conformity scores, Mondrian over hazard tiers of the *candidate*
class: class y enters the set iff s(x, y) <= q_{tier(y)}, where q_g is the finite-sample
(1-alpha_g) quantile of true-label scores among calibration points whose label is in tier g.
Under exchangeability of calibration and test points this gives, for every tier g,
P(Y in C(X) | Y in g) >= 1 - alpha_g. With few calibration points in a tier the quantile
level exceeds 1 and q_g = +inf, i.e. that tier's classes are always included -- the honest,
conservative consequence of small n (reported, not hidden). No guarantee is claimed under
distribution shift; there coverage is reported empirically.

Sets are summarised taxonomically: the deepest node of the APG IV tree that contains all
members (species < genus < family < order < root).
"""
from __future__ import annotations

import math

import numpy as np

from src.data.hazards import TIERS
from src.data.taxonomy import CLASSES, TAXONOMY
from src.metrics import softmax

TIER = np.array(TIERS)


def aps_scores(probs: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
    """APS score for every (sample, class): mass of classes ranked at or above the class.
    With u ~ U(0,1) the randomised version subtracts u * p_y."""
    order = np.argsort(-probs, 1)
    sorted_p = np.take_along_axis(probs, order, 1)
    cum = np.cumsum(sorted_p, 1)
    scores = np.empty_like(probs)
    np.put_along_axis(scores, order, cum, 1)
    if u is not None:
        scores = scores - u[:, None] * probs
    return scores


def calibrate(cal_logits, cal_y, alphas: dict[int, float], seed: int = 0) -> np.ndarray:
    """Per-class thresholds (C,) from tier-wise quantiles."""
    rng = np.random.default_rng(seed)
    p = softmax(cal_logits)
    s = aps_scores(p, rng.random(len(p)))[np.arange(len(p)), cal_y]
    q = np.full(len(CLASSES), np.inf)
    for g, a in alphas.items():
        sel = TIER[cal_y] == g
        n = int(sel.sum())
        level = math.ceil((n + 1) * (1 - a)) / max(n, 1)
        qg = np.inf if (n == 0 or level > 1) else float(np.quantile(s[sel], level, method="higher"))
        q[TIER == g] = qg
    return q


def predict_sets(logits, q, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = softmax(logits)
    s = aps_scores(p, rng.random(len(p)))
    sets = s <= q[None, :]
    sets[np.arange(len(p)), p.argmax(1)] = True  # never return an empty set
    return sets


def lca_depth(members: np.ndarray) -> int:
    """0 species, 1 genus, 2 family, 3 order, 4 root."""
    tax = [TAXONOMY[CLASSES[i]] for i in np.nonzero(members)[0]]
    if len(tax) == 1:
        return 0
    for depth, li in ((1, 2), (2, 1), (3, 0)):
        if len({t[li] for t in tax}) == 1:
            return depth
    return 4


def evaluate_sets(sets: np.ndarray, y: np.ndarray) -> dict:
    cov = sets[np.arange(len(y)), y]
    out = dict(coverage=float(cov.mean()), mean_size=float(sets.sum(1).mean()),
               singleton=float((sets.sum(1) == 1).mean()),
               mean_lca_depth=float(np.mean([lca_depth(r) for r in sets])))
    for g in sorted(set(TIER.tolist())):
        sel = TIER[y] == g
        if sel.any():
            out[f"coverage_tier{g}"] = float(cov[sel].mean())
            out[f"n_tier{g}"] = int(sel.sum())
    # how often a high-hazard species is silently excluded when it is the truth / when predicted
    hi = TIER == 2
    out["toxic_missed"] = int(((TIER[y] == 2) & ~cov).sum())
    out["toxic_flag_rate"] = float(sets[:, hi].any(1).mean())
    return out
