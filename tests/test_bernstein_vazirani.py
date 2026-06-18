import numpy as np

from qfs.algorithms.bernstein_vazirani import bernstein_vazirani


def test_bv_recovers_hidden_string():
    for s in ([1, 0, 1], [0, 0, 0], [1, 1, 1], [0, 1, 1, 0]):
        assert bernstein_vazirani(s, rng=np.random.default_rng(0)) == s


def test_bv_is_seed_independent():
    s = [1, 0, 1, 1]
    a = bernstein_vazirani(s, rng=np.random.default_rng(1))
    b = bernstein_vazirani(s, rng=np.random.default_rng(99))
    assert a == b == s
