"""Quantum phase estimation: read an eigenvalue's phase off a counting register."""

import numpy as np

from .. import gates
from ..statevector import StateVector
from ..dense import controlled_operator
from .qft import qft_matrix


def phase_estimation(U, eigenstate, t, rng=None):
    eigenstate = np.asarray(eigenstate, dtype=np.complex128)
    if not np.isclose(np.linalg.norm(eigenstate), 1.0):
        raise ValueError("eigenstate must be normalized (L2 norm 1)")
    m = int(round(np.log2(len(eigenstate))))
    n = t + m
    targets = list(range(t, n))

    # counting register |0...0> (high t bits), target register = eigenstate (low m bits)
    amps = np.zeros(2 ** n, dtype=np.complex128)
    amps[: len(eigenstate)] = eigenstate
    sv = StateVector.from_amplitudes(amps, rng=rng)

    for j in range(t):
        sv.apply(gates.H, j)                       # uniform superposition on counting qubits

    for j in range(t):                             # controlled-U^(2^(t-1-j)) ladder
        u_power = np.linalg.matrix_power(U, 2 ** (t - 1 - j))
        sv.amps = controlled_operator(u_power, j, targets, n) @ sv.amps

    # inverse QFT on the counting register (it is the high block, so kron with identity)
    inverse_qft = np.kron(qft_matrix(t).conj().T, np.eye(2 ** m, dtype=np.complex128))
    sv.amps = inverse_qft @ sv.amps

    bits = [sv.measure(j) for j in range(t)]        # read the counting register
    value = sum(b << (t - 1 - i) for i, b in enumerate(bits))
    return value / 2 ** t
