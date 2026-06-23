import numpy as np

from qfs.algorithms.shor import modmul_unitary


def test_modmul_is_a_permutation_unitary():
    U = modmul_unitary(7, 15, 4)
    assert np.allclose(U @ U.conj().T, np.eye(16))
    # every column is a basis vector (a permutation matrix)
    assert np.allclose(np.sort(U.sum(axis=0)), np.ones(16))


def test_modmul_maps_correctly():
    U = modmul_unitary(7, 15, 4)
    e = np.eye(16)
    assert np.allclose(U @ e[:, 1], e[:, 7])     # 7*1 mod 15 = 7
    assert np.allclose(U @ e[:, 7], e[:, 4])     # 7*7 mod 15 = 49 mod 15 = 4
    assert np.allclose(U @ e[:, 2], e[:, 14])    # 7*2 mod 15 = 14


def test_modmul_identity_above_N():
    U = modmul_unitary(7, 15, 4)
    e = np.eye(16)
    assert np.allclose(U @ e[:, 15], e[:, 15])   # 15 >= N, fixed
