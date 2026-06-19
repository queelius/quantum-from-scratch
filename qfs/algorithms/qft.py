"""Quantum Fourier Transform: direct matrix and gate circuit."""

import numpy as np


def qft_matrix(n):
    """The n-qubit QFT unitary: F[j, k] = exp(2j*pi*j*k / 2**n) / sqrt(2**n)."""
    N = 2 ** n
    js = np.arange(N)
    return np.exp(2j * np.pi * np.outer(js, js) / N) / np.sqrt(N)
