"""Deutsch-Jozsa: decide if f is constant or balanced with one oracle call."""

from .. import gates
from ..statevector import StateVector
from .oracles import bit_oracle


def deutsch_jozsa(f, n_in, rng=None):
    n = n_in + 1
    sv = StateVector(n, rng=rng)
    sv.apply(gates.X, n - 1)              # ancilla -> |1>
    for q in range(n):                   # Hadamard everywhere
        sv.apply(gates.H, q)
    sv.amps = bit_oracle(f, n_in) @ sv.amps
    for q in range(n_in):                # Hadamard on inputs
        sv.apply(gates.H, q)
    outcomes = [sv.measure(q) for q in range(n_in)]
    return "constant" if all(b == 0 for b in outcomes) else "balanced"
