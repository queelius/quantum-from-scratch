"""Circuit: a recorded sequence of gate operations runnable on a StateVector."""

from . import gates
from .statevector import StateVector


class Circuit:
    def __init__(self, n):
        self.n = n
        self.ops = []  # list of (U, target, controls, label)

    def apply(self, U, target, controls=(), label=None):
        self.ops.append((U, target, tuple(controls), label))
        return self

    def h(self, t):
        return self.apply(gates.H, t, label="H")

    def x(self, t):
        return self.apply(gates.X, t, label="X")

    def y(self, t):
        return self.apply(gates.Y, t, label="Y")

    def z(self, t):
        return self.apply(gates.Z, t, label="Z")

    def s(self, t):
        return self.apply(gates.S, t, label="S")

    def t_gate(self, t):
        return self.apply(gates.T, t, label="T")

    def cnot(self, c, t):
        return self.apply(gates.X, t, controls=(c,), label="CNOT")

    def cz(self, c, t):
        return self.apply(gates.Z, t, controls=(c,), label="CZ")

    def run(self, state=None, rng=None):
        if state is None:
            state = StateVector(self.n, rng=rng)
        for U, target, controls, _label in self.ops:
            state.apply(U, target, controls)
        return state

    def __len__(self):
        return len(self.ops)
