import numpy as np

from qfs.algorithms.shor import shor_order


def test_order_of_7_mod_15():
    assert shor_order(7, 15, t=4, rng=np.random.default_rng(0)) == 4


def test_order_of_2_mod_15():
    assert shor_order(2, 15, t=4, rng=np.random.default_rng(0)) == 4


def test_order_is_seed_robust():
    # order finding recovers r = 4 across several seeds (verified)
    for seed in range(4):
        assert shor_order(7, 15, t=4, rng=np.random.default_rng(seed)) == 4
