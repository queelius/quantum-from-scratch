"""Grover's search: amplitude amplification of a marked basis state."""

import numpy as np

from .. import gates
from ..statevector import StateVector
from .oracles import phase_oracle


def diffusion(n):
    dim = 2 ** n
    s = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    return 2 * np.outer(s, s.conj()) - np.eye(dim, dtype=np.complex128)


def optimal_iterations(n, num_marked=1):
    return int(np.floor((np.pi / 4) * np.sqrt((2 ** n) / num_marked)))


def grover_search(marked, n, iterations=None, rng=None):
    if iterations is None:
        iterations = optimal_iterations(n)
    sv = StateVector(n, rng=rng)
    for q in range(n):
        sv.apply(gates.H, q)
    oracle = phase_oracle(marked, n)
    diff = diffusion(n)
    for _ in range(iterations):
        sv.amps = oracle @ sv.amps
        sv.amps = diff @ sv.amps
    return sv
