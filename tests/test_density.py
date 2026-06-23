import numpy as np

from qfs import gates
from qfs.statevector import StateVector
from qfs.density import DensityMatrix


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
