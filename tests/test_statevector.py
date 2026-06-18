import numpy as np
import pytest

from qfs import gates
from qfs.statevector import StateVector


def test_ground_state():
    sv = StateVector(3)
    assert sv.n == 3
    assert sv.amps.shape == (8,)
    assert sv.amps[0] == 1.0
    assert np.isclose(sv.amps.sum(), 1.0)


def test_hadamard_makes_plus_state():
    sv = StateVector(1).apply(gates.H, 0)
    assert np.allclose(sv.amps, [1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_x_flips_qubit():
    sv = StateVector(1).apply(gates.X, 0)
    assert np.allclose(sv.amps, [0, 1])


def test_apply_is_chainable_and_normalized():
    sv = StateVector(2).apply(gates.H, 0).apply(gates.H, 1)
    assert np.isclose(np.linalg.norm(sv.amps), 1.0)
    assert np.allclose(sv.probabilities(), [0.25, 0.25, 0.25, 0.25])


def test_from_amplitudes():
    sv = StateVector.from_amplitudes([0, 1, 0, 0])
    assert sv.n == 2
    assert sv.amps[1] == 1.0


def test_from_amplitudes_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        StateVector.from_amplitudes([1, 0, 0])
