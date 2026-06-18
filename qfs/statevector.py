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

    def measure(self, qubit):
        pos = self.n - 1 - qubit
        idx = np.arange(2 ** self.n)
        is_one = ((idx >> pos) & 1).astype(bool)
        p_one = float(np.sum(np.abs(self.amps[is_one]) ** 2))
        outcome = 1 if self.rng.random() < p_one else 0
        keep = is_one if outcome == 1 else ~is_one
        self.amps = np.where(keep, self.amps, 0.0)
        self.amps = self.amps / np.linalg.norm(self.amps)
        return outcome

    def measure_all(self):
        probs = self.probabilities()
        i = int(self.rng.choice(len(probs), p=probs))
        bits = [(i >> (self.n - 1 - q)) & 1 for q in range(self.n)]
        self.amps = np.zeros_like(self.amps)
        self.amps[i] = 1.0
        return bits

    def sample(self, shots):
        probs = self.probabilities()
        draws = self.rng.choice(len(probs), size=shots, p=probs)
        counts = {}
        for d in draws:
            key = format(int(d), f"0{self.n}b")
            counts[key] = counts.get(key, 0) + 1
        return counts

    def expectation(self, observable):
        return float(np.real(np.conj(self.amps) @ observable @ self.amps))
