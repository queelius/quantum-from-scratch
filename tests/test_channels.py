import numpy as np
import pytest

from qfs import gates
from qfs.density import DensityMatrix
from qfs.channels import (
    depolarizing,
    amplitude_damping,
    phase_damping,
    bit_flip,
    phase_flip,
    apply_channel,
)

ALL = [depolarizing(0.3), amplitude_damping(0.4), phase_damping(0.5), bit_flip(0.2), phase_flip(0.2)]


def test_channels_are_trace_preserving():
    for kraus in ALL:
        s = sum(K.conj().T @ K for K in kraus)
        assert np.allclose(s, np.eye(2))


def test_apply_channel_keeps_trace_one():
    for kraus in ALL:
        dm = DensityMatrix.from_statevector([0.6, 0, 0.8, 0])
        apply_channel(dm, kraus, 0)
        assert np.isclose(np.trace(dm.rho).real, 1.0)


def test_amplitude_damping_decays_population():
    gamma = 0.4
    dm = DensityMatrix.from_statevector([0, 1])
    apply_channel(dm, amplitude_damping(gamma), 0)
    assert np.isclose(dm.probabilities()[1], 1 - gamma)
    assert np.isclose(dm.probabilities()[0], gamma)


def test_phase_damping_kills_coherence_keeps_population():
    lam = 0.5
    dm = DensityMatrix(1).apply(gates.H, 0)  # |+>, coherence 0.5
    apply_channel(dm, phase_damping(lam), 0)
    assert np.isclose(abs(dm.rho[0, 1]), 0.5 * np.sqrt(1 - lam))
    assert np.allclose(dm.probabilities(), [0.5, 0.5])


def test_channel_acts_only_on_target():
    dm = DensityMatrix.from_statevector(np.kron([0, 1], [0, 1]))  # |11>
    apply_channel(dm, amplitude_damping(0.5), 1)
    q0_one = dm.probabilities()[0b10] + dm.probabilities()[0b11]
    q1_one = dm.probabilities()[0b01] + dm.probabilities()[0b11]
    assert np.isclose(q0_one, 1.0)        # untargeted qubit 0 untouched
    assert np.isclose(q1_one, 0.5)        # targeted qubit 1 DID decay (rules out identity)


def test_flip_channels_apply_the_pauli_at_p_one():
    # bit_flip(1.0) is a deterministic X; phase_flip(1.0) is a deterministic Z
    dm = DensityMatrix.from_statevector([1, 0])              # |0>
    apply_channel(dm, bit_flip(1.0), 0)
    assert np.allclose(dm.probabilities(), [0, 1])           # -> |1>
    minus = [1 / np.sqrt(2), -1 / np.sqrt(2)]
    dm = DensityMatrix(1).apply(gates.H, 0)                  # |+>
    apply_channel(dm, phase_flip(1.0), 0)
    assert np.allclose(dm.rho, np.outer(minus, minus))       # -> |->


def test_amplitude_damping_saturates_to_ground():
    for amps in ([0, 1], [1 / np.sqrt(2), 1 / np.sqrt(2)]):
        dm = DensityMatrix.from_statevector(amps)
        apply_channel(dm, amplitude_damping(1.0), 0)
        assert np.allclose(dm.rho, np.outer([1, 0], [1, 0]))  # everything decays to |0><0|


def test_constructors_reject_out_of_range():
    for constructor in (depolarizing, amplitude_damping, phase_damping, bit_flip, phase_flip):
        with pytest.raises(ValueError):
            constructor(1.5)


def test_repeated_phase_damping_decoheres():
    dm = DensityMatrix.from_statevector([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
    coherence0 = abs(dm.rho[0, 3])
    for _ in range(15):
        apply_channel(dm, phase_damping(0.3), 0)
    assert abs(dm.rho[0, 3]) < 0.1 * coherence0
    assert np.isclose(np.trace(dm.rho).real, 1.0)
    assert np.allclose(np.diag(dm.rho).real, [0.5, 0, 0, 0.5])


def test_depolarizing_fully_mixes():
    dm = DensityMatrix.from_statevector([1, 0])
    apply_channel(dm, depolarizing(1.0), 0)
    assert np.allclose(dm.rho, np.eye(2) / 2)


def test_no_op_limits():
    # p=0 / gamma=0 / lam=0 leave the state untouched
    for kraus in (depolarizing(0.0), amplitude_damping(0.0), phase_damping(0.0), bit_flip(0.0), phase_flip(0.0)):
        dm = DensityMatrix(1).apply(gates.H, 0)
        before = dm.rho.copy()
        apply_channel(dm, kraus, 0)
        assert np.allclose(dm.rho, before)
