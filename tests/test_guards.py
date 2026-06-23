import numpy as np
import pytest

from qfs import gates
from qfs.dense import controlled_operator
from qfs.algorithms.phase_estimation import phase_estimation


def test_controlled_operator_rejects_control_in_targets():
    with pytest.raises(ValueError):
        controlled_operator(gates.X, control=1, targets=[1], n=2)


def test_phase_estimation_rejects_unnormalized_eigenstate():
    U = np.array([[1, 0], [0, 1j]], dtype=complex)
    with pytest.raises(ValueError):
        phase_estimation(U, [1, 1], t=3)            # norm sqrt(2), not 1


def test_phase_estimation_accepts_normalized_eigenstate():
    U = np.array([[1, 0], [0, 1j]], dtype=complex)  # phase 1/4 on |1>
    assert np.isclose(phase_estimation(U, [0, 1], t=3, rng=np.random.default_rng(0)), 0.25)
