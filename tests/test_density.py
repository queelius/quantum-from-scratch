import numpy as np
import pytest

from qfs import gates
from qfs.statevector import StateVector
from qfs.density import DensityMatrix


def test_apply_non_hermitian_gate_matches_statevector():
    # H and X are Hermitian, so a wrong-dagger bug (op.T instead of op.conj().T)
    # would be invisible; non-Hermitian gates through apply catch it.
    for U in (gates.S, gates.T, gates.Ry(0.7)):
        sv = StateVector(2).apply(gates.H, 0).apply(U, 1)
        dm = DensityMatrix(2).apply(gates.H, 0).apply(U, 1)
        assert np.allclose(dm.rho, np.outer(sv.amps, sv.amps.conj()))


def test_partial_trace_distinguishes_qubits():
    # |1> tensor |0> is asymmetric: keep=[0] -> |1><1|, keep=[1] -> |0><0|.
    # A bug that swaps qubit indices would fail this (the symmetric Bell tests cannot).
    dm = DensityMatrix.from_statevector(np.kron([0, 1], [1, 0]))
    assert np.allclose(dm.partial_trace(keep=[0]).rho, np.outer([0, 1], [0, 1]))
    assert np.allclose(dm.partial_trace(keep=[1]).rho, np.outer([1, 0], [1, 0]))


def test_pure_state_has_unit_purity():
    # purity Tr(rho^2) is 1 for a pure state, < 1 for a mixed one; this distinguishes
    # the correct |bell><bell| from a structurally-valid-but-wrong mixed matrix.
    bell = DensityMatrix.from_statevector([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
    assert np.isclose(np.trace(bell.rho @ bell.rho).real, 1.0)
    reduced = bell.partial_trace(keep=[0])
    assert np.isclose(np.trace(reduced.rho @ reduced.rho).real, 0.5)


def test_measure_targets_the_right_qubit():
    # |0> tensor |1> (n=2): measuring qubit 0 deterministically gives 0, qubit 1 gives 1.
    state = np.kron([1, 0], [0, 1])
    assert DensityMatrix.from_statevector(state, rng=np.random.default_rng(0)).measure(0) == 0
    assert DensityMatrix.from_statevector(state, rng=np.random.default_rng(0)).measure(1) == 1


def test_from_statevector_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        DensityMatrix.from_statevector([1, 0, 0])


def test_ground_state():
    dm = DensityMatrix(2)
    assert dm.rho.shape == (4, 4)
    assert dm.rho[0, 0] == 1.0
    assert np.isclose(np.trace(dm.rho).real, 1.0)


def test_diagonal_is_probabilities():
    dm = DensityMatrix(1).apply(gates.H, 0)
    assert np.allclose(dm.probabilities(), [0.5, 0.5])


def test_matches_statevector_on_noiseless_circuit():
    ops = [(gates.H, 0, ()), (gates.X, 1, (0,)), (gates.H, 2, ()), (gates.X, 2, (1,))]
    sv = StateVector(3)
    dm = DensityMatrix(3)
    for U, tgt, ctrl in ops:
        sv.apply(U, tgt, ctrl)
        dm.apply(U, tgt, ctrl)
    assert np.allclose(dm.probabilities(), sv.probabilities())
    assert np.allclose(dm.rho, np.outer(sv.amps, sv.amps.conj()))


def test_partial_trace_of_bell_is_maximally_mixed():
    bell = DensityMatrix.from_statevector([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
    assert np.allclose(bell.partial_trace(keep=[0]).rho, np.eye(2) / 2)


def test_two_faces_of_rho():
    bell = DensityMatrix.from_statevector([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
    reduced = bell.partial_trace(keep=[0]).rho
    mixture = 0.5 * np.outer([1, 0], [1, 0]) + 0.5 * np.outer([0, 1], [0, 1])
    assert np.allclose(reduced, mixture)


def test_partial_trace_of_product_state():
    plus = [1 / np.sqrt(2), 1 / np.sqrt(2)]
    prod = np.kron([1, 0], plus)
    dm = DensityMatrix.from_statevector(prod)
    assert np.allclose(dm.partial_trace(keep=[1]).rho, np.outer(plus, plus))


def test_invariants():
    dm = DensityMatrix.from_statevector([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
    assert np.isclose(np.trace(dm.rho).real, 1.0)
    assert np.allclose(dm.rho, dm.rho.conj().T)
    assert np.all(np.linalg.eigvalsh(dm.rho) > -1e-12)


def test_expectation_matches_statevector():
    sv = StateVector(1).apply(gates.Ry(0.9), 0)
    dm = DensityMatrix.from_statevector(sv.amps)
    for O in (gates.X, gates.Y, gates.Z):
        assert np.isclose(dm.expectation(O), sv.expectation(O))


def test_measure_statistics_and_collapse():
    outcomes = [
        DensityMatrix(1, rng=np.random.default_rng(s)).apply(gates.H, 0).measure(0)
        for s in range(2000)
    ]
    assert 0.45 < sum(outcomes) / len(outcomes) < 0.55
    dm = DensityMatrix(1, rng=np.random.default_rng(3)).apply(gates.H, 0)
    b = dm.measure(0)
    assert np.isclose(dm.probabilities()[b], 1.0)
