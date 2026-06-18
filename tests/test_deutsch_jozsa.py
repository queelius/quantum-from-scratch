import numpy as np

from qfs.algorithms.oracles import bit_oracle, phase_oracle
from qfs.algorithms.deutsch_jozsa import deutsch_jozsa


def test_bit_oracle_is_permutation():
    op = bit_oracle(lambda x: x & 1, 2)  # f(x) = low bit of x
    assert np.allclose(op @ op, np.eye(op.shape[0]))  # self-inverse permutation


def test_phase_oracle_flips_one_amplitude():
    op = phase_oracle(3, 2)
    assert op[3, 3] == -1.0
    assert op[0, 0] == 1.0


def test_dj_constant_zero():
    assert deutsch_jozsa(lambda x: 0, 3, rng=np.random.default_rng(0)) == "constant"


def test_dj_constant_one():
    assert deutsch_jozsa(lambda x: 1, 3, rng=np.random.default_rng(0)) == "constant"


def test_dj_balanced_parity():
    # parity of the 3 input bits is balanced
    f = lambda x: (bin(x).count("1")) & 1
    assert deutsch_jozsa(f, 3, rng=np.random.default_rng(0)) == "balanced"


def test_dj_balanced_first_bit():
    f = lambda x: (x >> 2) & 1  # MSB of a 3-bit input
    assert deutsch_jozsa(f, 3, rng=np.random.default_rng(0)) == "balanced"
