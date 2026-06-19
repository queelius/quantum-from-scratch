import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator

from qfs.algorithms.qft import qft_matrix


def _equal_up_to_phase(a, b):
    return np.isclose(abs(np.vdot(a.flatten(), b.flatten())) /
                      (np.linalg.norm(a) * np.linalg.norm(b)), 1.0, atol=1e-8)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_qft_matches_qiskit(n):
    # QFTGate is the current (non-deprecated) Qiskit QFT. Its operator equals our
    # big-endian qft_matrix up to a global phase (verified for n=1..4, qiskit 2.4).
    ours = qft_matrix(n)
    theirs = Operator(QFTGate(n)).data
    assert _equal_up_to_phase(ours, theirs)
