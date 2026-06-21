import numpy as np

from qfs.algorithms.phase_estimation import phase_estimation


def phase_gate(phi):
    return np.array([[1, 0], [0, np.exp(2j * np.pi * phi)]], dtype=complex)


def test_dyadic_phases_exact():
    # |1> is the eigenstate with eigenvalue exp(2j*pi*phi); 4 counting qubits
    # resolve any multiple of 1/16 exactly.
    for phi in (0.0, 0.25, 0.5, 0.125, 0.375, 0.0625):
        est = phase_estimation(phase_gate(phi), [0, 1], t=4, rng=np.random.default_rng(0))
        assert np.isclose(est, phi), f"phi={phi} est={est}"


def test_t_gate_eighth():
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)  # phase 1/8
    assert np.isclose(phase_estimation(T, [0, 1], t=3, rng=np.random.default_rng(0)), 0.125)


def test_eigenvalue_one_gives_zero_phase():
    # |0> is the eigenstate of the phase gate with eigenvalue 1 (phi = 0)
    est = phase_estimation(phase_gate(0.5), [1, 0], t=4, rng=np.random.default_rng(0))
    assert np.isclose(est, 0.0)


def test_phase_is_seed_independent_for_dyadic():
    # exact dyadic phases collapse the counting register deterministically
    a = phase_estimation(phase_gate(0.375), [0, 1], t=4, rng=np.random.default_rng(1))
    b = phase_estimation(phase_gate(0.375), [0, 1], t=4, rng=np.random.default_rng(99))
    assert a == b == 0.375
