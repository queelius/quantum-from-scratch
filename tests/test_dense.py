import numpy as np

from qfs import gates
from qfs.dense import embed, kron_embed


def test_kron_embed_single_qubit():
    # H on qubit 0 of a 1-qubit system is just H.
    assert np.allclose(kron_embed(gates.H, 0, 1), gates.H)
    # X on qubit 1 of a 2-qubit system is I kron X.
    assert np.allclose(kron_embed(gates.X, 1, 2), np.kron(gates.I, gates.X))


def test_embed_matches_kron_when_no_controls():
    for target in range(3):
        assert np.allclose(embed(gates.H, target, 3), kron_embed(gates.H, target, 3))


def test_embed_cnot_truth_table():
    # control = qubit 0, target = qubit 1, big-endian basis |q0 q1>.
    cnot = embed(gates.X, 1, 2, controls=(0,))
    e = np.eye(4)
    # |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>
    assert np.allclose(cnot @ e[:, 0], e[:, 0])
    assert np.allclose(cnot @ e[:, 1], e[:, 1])
    assert np.allclose(cnot @ e[:, 2], e[:, 3])
    assert np.allclose(cnot @ e[:, 3], e[:, 2])


def test_embed_is_unitary():
    op = embed(gates.X, 2, 3, controls=(0, 1))  # Toffoli
    assert np.allclose(op @ op.conj().T, np.eye(8))
