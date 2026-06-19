"""Quantum Fourier Transform: direct matrix and gate circuit."""

import numpy as np

from ..circuit import Circuit
from .. import gates


def qft_matrix(n):
    """The n-qubit QFT unitary: F[j, k] = exp(2j*pi*j*k / 2**n) / sqrt(2**n)."""
    N = 2 ** n
    js = np.arange(N)
    return np.exp(2j * np.pi * np.outer(js, js) / N) / np.sqrt(N)


def qft_circuit(n):
    """The QFT as a gate circuit: Hadamards, a controlled-phase ladder, and a
    bit-reversal swap network. Equals qft_matrix(n) exactly."""
    qc = Circuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.apply(gates.phase(2 * np.pi / 2 ** (k - j + 1)), j, controls=(k,))
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc
