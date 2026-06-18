"""Bernstein-Vazirani: recover a hidden string s where f(x) = s . x (mod 2)."""

from .. import gates
from ..statevector import StateVector
from .oracles import bit_oracle


def bernstein_vazirani(s_bits, rng=None):
    n_in = len(s_bits)

    def f(x):
        bits = [(x >> (n_in - 1 - i)) & 1 for i in range(n_in)]
        return sum(si * bi for si, bi in zip(s_bits, bits)) & 1

    n = n_in + 1
    sv = StateVector(n, rng=rng)
    sv.apply(gates.X, n - 1)
    for q in range(n):
        sv.apply(gates.H, q)
    sv.amps = bit_oracle(f, n_in) @ sv.amps
    for q in range(n_in):
        sv.apply(gates.H, q)
    return [sv.measure(q) for q in range(n_in)]
