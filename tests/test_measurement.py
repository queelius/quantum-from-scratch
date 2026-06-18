import numpy as np

from qfs import gates
from qfs.statevector import StateVector


def test_measure_basis_state_is_deterministic():
    sv = StateVector(1, rng=np.random.default_rng(0)).apply(gates.X, 0)
    assert sv.measure(0) == 1
    # state has collapsed to |1>, still normalized
    assert np.isclose(np.linalg.norm(sv.amps), 1.0)


def test_measure_plus_state_statistics():
    sv = StateVector(1, rng=np.random.default_rng(42)).apply(gates.H, 0)
    outcomes = [
        StateVector(1, rng=np.random.default_rng(s)).apply(gates.H, 0).measure(0)
        for s in range(2000)
    ]
    frac_one = sum(outcomes) / len(outcomes)
    assert 0.45 < frac_one < 0.55


def test_measure_collapses_state():
    sv = StateVector(1, rng=np.random.default_rng(1)).apply(gates.H, 0)
    b = sv.measure(0)
    # after collapse the state is a basis state consistent with b
    assert np.isclose(sv.probabilities()[b], 1.0)


def test_sample_counts_sum_to_shots():
    sv = StateVector(2, rng=np.random.default_rng(7)).apply(gates.H, 0).apply(gates.H, 1)
    counts = sv.sample(1000)
    assert sum(counts.values()) == 1000
    assert set(counts) <= {"00", "01", "10", "11"}


def test_expectation_z_on_basis_states():
    assert np.isclose(StateVector(1).expectation(gates.Z), 1.0)        # <0|Z|0> = +1
    assert np.isclose(StateVector(1).apply(gates.X, 0).expectation(gates.Z), -1.0)
    assert np.isclose(StateVector(1).apply(gates.H, 0).expectation(gates.X), 1.0)
