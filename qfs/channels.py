"""Noise channels: Kraus operators acting on a DensityMatrix (post 9).

A channel is a list of Kraus operators K_k with sum_k K_k-dagger K_k = I. It
evolves a density matrix as rho -> sum_k K_k rho K_k-dagger. Unlike unitary
evolution, a channel can turn a pure state into a mixed one: this is how noise
and decoherence enter the model.
"""

import numpy as np

from .dense import embed

_I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def depolarizing(p):
    """With probability p, replace the qubit with the maximally mixed state."""
    if not 0 <= p <= 1:
        raise ValueError("depolarizing probability p must be in [0, 1]")
    return [np.sqrt(1 - 3 * p / 4) * _I, np.sqrt(p / 4) * _X,
            np.sqrt(p / 4) * _Y, np.sqrt(p / 4) * _Z]


def amplitude_damping(gamma):
    """Energy loss: |1> decays to |0> with probability gamma (models T1 relaxation)."""
    if not 0 <= gamma <= 1:
        raise ValueError("amplitude damping gamma must be in [0, 1]")
    return [np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128),
            np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)]


def phase_damping(lam):
    """Loss of coherence without energy loss: shrinks off-diagonals (models T2 dephasing).

    lam is the probability per application that the environment learns the qubit's
    phase; it scales the off-diagonal coherence by sqrt(1 - lam).
    """
    if not 0 <= lam <= 1:
        raise ValueError("phase damping lam must be in [0, 1]")
    return [np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=np.complex128),
            np.array([[0, 0], [0, np.sqrt(lam)]], dtype=np.complex128)]


def bit_flip(p):
    """Apply X with probability p."""
    if not 0 <= p <= 1:
        raise ValueError("bit flip probability p must be in [0, 1]")
    return [np.sqrt(1 - p) * _I, np.sqrt(p) * _X]


def phase_flip(p):
    """Apply Z with probability p."""
    if not 0 <= p <= 1:
        raise ValueError("phase flip probability p must be in [0, 1]")
    return [np.sqrt(1 - p) * _I, np.sqrt(p) * _Z]


def apply_channel(dm, kraus, target):
    """Apply a single-qubit channel (list of 2x2 Kraus operators) to `target` of dm.

    Evolves dm in place: rho -> sum_k E_k rho E_k-dagger, where each E_k is the
    Kraus operator embedded on qubit `target`. Returns dm.
    """
    embedded = [embed(K, target, dm.n) for K in kraus]
    dm.rho = sum(E @ dm.rho @ E.conj().T for E in embedded)
    return dm
