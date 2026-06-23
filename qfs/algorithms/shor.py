"""Shor's algorithm: order finding by phase estimation, plus the classical reduction."""

import math
from fractions import Fraction

import numpy as np

from .phase_estimation import phase_estimation


def modmul_unitary(a, N, n_target):
    """Permutation |x> -> |(a*x) mod N> for x < N, identity for x >= N."""
    dim = 2 ** n_target
    op = np.zeros((dim, dim), dtype=np.complex128)
    for x in range(dim):
        op[(a * x) % N if x < N else x, x] = 1.0
    return op
