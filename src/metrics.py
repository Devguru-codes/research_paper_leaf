from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.data.taxonomy import CLASSES, tree_distance

_TD = np.array([[tree_distance(i, j) for j in range(len(CLASSES))] for i in range(len(CLASSES))])


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            out += sel.mean() * abs((pred[sel] == y[sel]).mean() - conf[sel].mean())
    return float(out)


def classification_metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    p = softmax(logits)
    pred = p.argmax(1)
    wrong = pred != y
    return dict(
        acc=float((pred == y).mean()),
        bal_acc=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average="macro")),
        nll=float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean()),
        ece=float(ece(p, y)),
        mistake_severity=float(_TD[y[wrong], pred[wrong]].mean()) if wrong.any() else 0.0,
        n=int(len(y)),
    )
