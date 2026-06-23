"""DensityMatrix engine: mixed states as 2^n x 2^n operators (post 8).

A density matrix rho describes a state that may be a classical mixture of pure
states, or the reduced state of one part of an entangled whole. It is the object
the pure statevector cannot represent.
"""

import numpy as np

from .dense import embed


class DensityMatrix:
    """A mixed quantum state: a 2^n x 2^n positive, trace-1, Hermitian operator."""

    def __init__(self, n, rng=None):
        """Create an n-qubit density matrix in the ground state |0...0><0...0|."""
        self.n = n
        dim = 2 ** n
        self.rho = np.zeros((dim, dim), dtype=np.complex128)
        self.rho[0, 0] = 1.0
        self.rng = rng if rng is not None else np.random.default_rng()

    @classmethod
    def from_statevector(cls, amps, rng=None):
        """Build the density matrix |psi><psi| of a pure state."""
        amps = np.asarray(amps, dtype=np.complex128)
        n = int(round(np.log2(len(amps))))
        if len(amps) != 2 ** n:
            raise ValueError(
                f"amplitude vector length {len(amps)} is not a power of two"
            )
        dm = cls(n, rng)
        dm.rho = np.outer(amps, amps.conj())
        return dm

    def apply(self, U, target, controls=()):
        """Unitary evolution rho -> op rho op-dagger (op is the embedded gate). Chainable."""
        op = embed(U, target, self.n, controls)
        self.rho = op @ self.rho @ op.conj().T
        return self

    def probabilities(self):
        """Measurement probabilities: the real diagonal of rho."""
        return np.real(np.diag(self.rho))

    def expectation(self, observable):
        """Expectation value Tr(observable @ rho)."""
        return float(np.real(np.trace(observable @ self.rho)))

    def partial_trace(self, keep):
        """Trace out the qubits not in `keep`, returning the reduced DensityMatrix.

        Tracing out qubit B of an entangled pure state on AB gives a mixed state on
        A: this is one of the two ways a density matrix becomes mixed. `keep` is
        sorted, so the reduced matrix is laid out in ascending kept-qubit order
        regardless of the order it is passed in.
        """
        keep = sorted(keep)
        n = self.n
        t = self.rho.reshape([2] * (2 * n))
        letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        sub_row = [""] * n
        sub_col = [""] * n
        idx = 0
        for q in range(n):
            if q in keep:
                sub_row[q] = letters[idx]
                idx += 1
                sub_col[q] = letters[idx]
                idx += 1
            else:
                shared = letters[idx]
                idx += 1
                sub_row[q] = shared
                sub_col[q] = shared
        subs = "".join(sub_row) + "".join(sub_col)
        out = "".join(sub_row[q] for q in keep) + "".join(sub_col[q] for q in keep)
        reduced = np.einsum(subs + "->" + out, t).reshape(2 ** len(keep), 2 ** len(keep))
        result = DensityMatrix(len(keep), self.rng)
        result.rho = reduced
        return result

    def measure(self, qubit):
        """Measure one qubit (Born rule), collapse rho onto the outcome, return the bit."""
        pos = self.n - 1 - qubit
        idx = np.arange(2 ** self.n)
        is_one = ((idx >> pos) & 1).astype(bool)
        p_one = float(np.sum(self.probabilities()[is_one]))
        outcome = 1 if self.rng.random() < p_one else 0
        keep = is_one if outcome == 1 else ~is_one
        proj = np.diag(keep.astype(float)).astype(np.complex128)
        self.rho = proj @ self.rho @ proj
        self.rho = self.rho / np.trace(self.rho).real
        return outcome
