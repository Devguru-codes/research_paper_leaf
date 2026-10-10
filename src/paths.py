"""Resolve dataset / cache locations for both local runs and Kaggle kernels."""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

_LOCAL_RAW = PROJECT / "data" / "Segmented Medicinal Leaf Images-20261003T153133Z-1-001" / "Segmented Medicinal Leaf Images"


def raw_root() -> Path:
    env = os.environ.get("MEDLEAF_RAW")
    if env:
        return Path(env)
    return _LOCAL_RAW


def cache_root() -> Path:
    """Preprocessed images/masks/vein maps + index.csv."""
    env = os.environ.get("MEDLEAF_CACHE")
    if env:
        return Path(env)
    return PROJECT / "cache"


def results_root() -> Path:
    env = os.environ.get("MEDLEAF_RESULTS")
    p = Path(env) if env else PROJECT / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p
