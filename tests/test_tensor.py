import numpy as np

from qfs import gates
from qfs.dense import embed
from qfs.tensor import apply_gate
from qfs.statevector import StateVector


def _rand_state(n, rng):
    v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return v / np.linalg.norm(v)


def test_single_qubit_matches_dense_embed():
    rng = np.random.default_rng(0)
    for n in range(1, 6):
        for t in range(n):
            for G in (gates.H, gates.X, gates.Y, gates.Z, gates.S, gates.T):
                v = _rand_state(n, rng)
                assert np.allclose(apply_gate(v, G, [t], n), embed(G, t, n) @ v)


def test_controlled_gate_matches_dense():
    rng = np.random.default_rng(1)
    cnot2 = embed(gates.X, 1, 2, controls=(0,))  # 4x4 CNOT (control=0, target=1)
    for n in range(2, 6):
        for c in range(n):
            for t in range(n):
                if c == t:
                    continue
                v = _rand_state(n, rng)
                a = apply_gate(v, cnot2, [c, t], n)
                b = embed(gates.X, t, n, controls=(c,)) @ v
                assert np.allclose(a, b)


def test_swap_permutes_basis_state():
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    v = np.zeros(8, dtype=complex)
    v[0b100] = 1  # |100>
    out = apply_gate(v, swap, [0, 2], 3)  # swap qubit 0 and qubit 2
    assert np.allclose(out, np.eye(8)[:, 0b001])  # |001>


def test_general_two_qubit_unitary_preserves_norm():
    rng = np.random.default_rng(2)
    M = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    Q, _ = np.linalg.qr(M)  # a random 2-qubit unitary
    v = _rand_state(4, rng)
    assert np.isclose(np.linalg.norm(apply_gate(v, Q, [1, 3], 4)), 1.0)


def test_apply_tensor_on_statevector_matches_dense_apply():
    rng = np.random.default_rng(3)
    cnot2 = embed(gates.X, 1, 2, controls=(0,))
    for _ in range(20):
        n = int(rng.integers(2, 5))
        c, t = rng.choice(n, size=2, replace=False)
        sv_t = StateVector(n).apply(gates.H, 0).apply_tensor(cnot2, [int(c), int(t)])
        sv_d = StateVector(n).apply(gates.H, 0).apply(gates.X, int(t), controls=(int(c),))
        assert np.allclose(sv_t.amps, sv_d.amps)


def test_scales_beyond_dense():
    # einsum applies a gate on 16 qubits in milliseconds; the dense 2^16 x 2^16
    # operator (~34 GB) is infeasible. Confirm it runs and stays normalized.
    n = 16
    v = np.zeros(2 ** n, dtype=complex)
    v[0] = 1.0
    for q in range(n):
        v = apply_gate(v, gates.H, [q], n)  # H on all 16 qubits
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert np.allclose(np.abs(v), 1 / np.sqrt(2 ** n))  # uniform superposition
