import numpy as np

from src.conformal import TIER, calibrate, evaluate_sets, lca_depth, predict_sets


def _synthetic(n, rng, C=30, temp=1.5):
    y = rng.integers(C, size=n)
    logits = rng.normal(size=(n, C))
    logits[np.arange(n), y] += rng.normal(1.5, 1.0, n)  # noisy but informative classifier
    return logits / temp, y


def test_marginal_coverage_exchangeable():
    rng = np.random.default_rng(0)
    covs = []
    for rep in range(30):
        lc, yc = _synthetic(3000, rng)
        lt, yt = _synthetic(3000, rng)
        q = calibrate(lc, yc, {0: 0.1, 1: 0.1, 2: 0.1}, seed=rep)
        covs.append(evaluate_sets(predict_sets(lt, q, seed=rep + 99), yt)["coverage"])
    assert np.mean(covs) >= 0.9 - 0.005


def test_small_tier_is_always_included():
    rng = np.random.default_rng(1)
    lc, yc = _synthetic(300, rng)
    q = calibrate(lc, yc, {0: 0.1, 1: 0.1, 2: 0.001})
    assert np.isinf(q[TIER == 2]).all()
    lt, yt = _synthetic(200, rng)
    sets = predict_sets(lt, q)
    assert sets[:, TIER == 2].all()


def test_lca_depth():
    m = np.zeros(30, bool)
    m[[28, 29]] = False
    from src.data.taxonomy import CLASSES
    i = CLASSES.index("Syzygium Cumini (Jamun)")
    j = CLASSES.index("Syzygium Jambos (Rose Apple)")
    k = CLASSES.index("Psidium Guajava (Guava)")
    l = CLASSES.index("Punica Granatum (Pomegranate)")
    a = np.zeros(30, bool); a[i] = True
    assert lca_depth(a) == 0
    a[j] = True
    assert lca_depth(a) == 1
    a[k] = True
    assert lca_depth(a) == 2
    a[l] = True
    assert lca_depth(a) == 3
