import numpy as np
import pandas as pd
import pytest

from src.data.splits import leakage_report, make_split
from src.paths import cache_root

IDX = cache_root() / "index.csv"


@pytest.mark.skipif(not IDX.exists(), reason="run src.data.audit first")
@pytest.mark.parametrize("fold", range(5))
def test_grouped_protocol_has_no_leakage(fold):
    df = pd.read_csv(IDX)
    sp = make_split(df, fold, "grouped")
    assert leakage_report(df, sp) == dict(test_in_train_leaf_group=0, test_in_train_exact=0)
    allidx = np.concatenate([sp["train"], sp["cal"], sp["test"]])
    assert len(set(allidx)) == len(allidx)
    assert set(df.loc[sp["test"]].label) == set(range(30))
    assert df.loc[allidx].keep.all()


@pytest.mark.skipif(not IDX.exists(), reason="run src.data.audit first")
def test_grouped_folds_cover_dataset_once():
    df = pd.read_csv(IDX)
    tests = np.concatenate([make_split(df, f, "grouped")["test"] for f in range(5)])
    assert len(tests) == df.keep.sum() == len(set(tests))


@pytest.mark.skipif(not IDX.exists(), reason="run src.data.audit first")
def test_random_dups_protocol_leaks():
    df = pd.read_csv(IDX)
    leaks = sum(leakage_report(df, make_split(df, f, "random_dups"))["test_in_train_leaf_group"] for f in range(5))
    assert leaks > 0
