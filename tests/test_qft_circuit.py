import numpy as np

from qfs.algorithms.qft import qft_matrix, qft_circuit
from qfs.statevector import StateVector


def _circuit_unitary(circ, n):
    # assemble the circuit's operator by running it on each basis state
    cols = []
    for k in range(2 ** n):
        sv = StateVector.from_amplitudes(np.eye(2 ** n)[:, k])
        cols.append(circ.run(state=sv).amps)
    return np.column_stack(cols)


def test_qft_circuit_matches_matrix_exactly():
    for n in (1, 2, 3, 4):
        U = _circuit_unitary(qft_circuit(n), n)
        assert np.allclose(U, qft_matrix(n))


def test_qft_circuit_of_zero_is_uniform():
    sv = qft_circuit(3).run(state=StateVector(3))
    assert np.allclose(sv.amps, np.ones(8) / np.sqrt(8))
