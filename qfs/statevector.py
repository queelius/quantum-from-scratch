"""StateVector engine: pure quantum states as length-2^n amplitude vectors."""

import numpy as np

from .dense import embed


class StateVector:
    def __init__(self, n, rng=None):
        self.n = n
        self.amps = np.zeros(2 ** n, dtype=np.complex128)
        self.amps[0] = 1.0
        self.rng = rng if rng is not None else np.random.default_rng()

    @classmethod
    def from_amplitudes(cls, amps, rng=None):
        amps = np.asarray(amps, dtype=np.complex128)
        n = int(round(np.log2(len(amps))))
        if len(amps) != 2 ** n:
            raise ValueError(
                f"amplitude vector length {len(amps)} is not a power of two"
            )
        sv = cls(n, rng)
        sv.amps = amps.copy()
        return sv

    def apply(self, U, target, controls=()):
        self.amps = embed(U, target, self.n, controls) @ self.amps
        return self

    def probabilities(self):
        return np.abs(self.amps) ** 2
