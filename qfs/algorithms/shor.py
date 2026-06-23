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


def order_from_phase(phi, N):
    """Candidate order: denominator of the best rational approximation of phi
    with denominator at most N (the continued-fraction convergent)."""
    return Fraction(phi).limit_denominator(N).denominator


def shor_order(a, N, t, trials=16, rng=None):
    """Find the order of a modulo N using phase estimation.

    Runs phase estimation on modmul_unitary(a, N, ceil(log2 N)) with the
    eigenstate-superposition |1> up to `trials` times. For each measured
    phase, extracts a candidate order and keeps it if a**r % N == 1.
    Returns the smallest verified order, or None if no valid order found.

    Args:
        a: Base.
        N: Modulus.
        t: Number of precision qubits for phase estimation.
        trials: Number of phase estimation runs (default 16).
        rng: NumPy random number generator (default: new instance).

    Returns:
        Smallest verified order, or None.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_target = int(np.ceil(np.log2(N)))
    U = modmul_unitary(a, N, n_target)
    one = np.zeros(2 ** n_target, dtype=np.complex128)
    one[1] = 1.0
    candidates = set()
    for _ in range(trials):
        phi = phase_estimation(U, one, t, rng=rng)
        r = order_from_phase(phi, N)
        if r and pow(a, r, N) == 1:
            candidates.add(r)
    return min(candidates) if candidates else None


def shor_factor(N, t, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if N % 2 == 0:
        return (2, N // 2)
    for a in range(2, N):
        g = math.gcd(a, N)
        if g != 1:
            return (g, N // g)
        r = shor_order(a, N, t, rng=rng)
        if r and r % 2 == 0:
            x = pow(a, r // 2, N)
            if x != N - 1:
                for f in (math.gcd(x - 1, N), math.gcd(x + 1, N)):
                    if 1 < f < N:
                        return (f, N // f)
    return None
