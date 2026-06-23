import numpy as np

from qfs.algorithms.shor import shor_factor


def test_factor_fifteen():
    f = shor_factor(15, t=4, rng=np.random.default_rng(0))
    assert f is not None
    assert sorted(f) == [3, 5]
    assert f[0] * f[1] == 15


def test_factor_fifteen_seed_robust():
    for seed in range(3):
        f = shor_factor(15, t=4, rng=np.random.default_rng(seed))
        assert sorted(f) == [3, 5]


def test_even_number_shortcut():
    # classical fast path only; the quantum path is exercised by test_factor_fifteen
    assert sorted(shor_factor(14, t=4, rng=np.random.default_rng(0))) == [2, 7]
