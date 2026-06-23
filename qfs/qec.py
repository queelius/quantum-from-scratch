"""Quantum error correction: the 3-qubit repetition codes (post 10).

A logical qubit is spread across three physical qubits so that a single bit-flip
(or, in the X basis, phase-flip) error can be detected from a parity syndrome and
undone without ever measuring the protected logical state.
"""

import numpy as np

from . import gates
from .dense import embed

_Z0Z1 = embed(gates.Z, 0, 3) @ embed(gates.Z, 1, 3)
_Z1Z2 = embed(gates.Z, 1, 3) @ embed(gates.Z, 2, 3)
_BITFLIP_SYNDROME = {(1, 1): None, (-1, 1): 0, (-1, -1): 1, (1, -1): 2}
_Hall = embed(gates.H, 0, 3) @ embed(gates.H, 1, 3) @ embed(gates.H, 2, 3)


def encode_bitflip(a, b):
    """Encode logical a|0>+b|1> as a|000>+b|111> via two CNOTs."""
    psi = np.zeros(8, dtype=np.complex128)
    psi[0] = a
    psi[0b100] = b
    psi = embed(gates.X, 1, 3, controls=(0,)) @ psi
    psi = embed(gates.X, 2, 3, controls=(0,)) @ psi
    return psi


def bitflip_syndrome(psi):
    """The parity syndrome (sign of <Z0 Z1>, sign of <Z1 Z2>) of an encoded state.

    The two parity operators have definite eigenvalues on a code state with at most
    one bit-flip, so reading their expectation extracts the syndrome without
    disturbing the logical information.
    """
    s1 = int(round(np.real(np.conj(psi) @ _Z0Z1 @ psi)))
    s2 = int(round(np.real(np.conj(psi) @ _Z1Z2 @ psi)))
    return (s1, s2)


def correct_bitflip(psi):
    """Detect a single bit-flip from the syndrome and undo it. Returns the corrected state."""
    flipped = _BITFLIP_SYNDROME[bitflip_syndrome(psi)]
    if flipped is not None:
        psi = embed(gates.X, flipped, 3) @ psi
    return psi


def decode_bitflip(psi):
    """Reverse the encoding, returning the logical state to qubit 0."""
    psi = embed(gates.X, 2, 3, controls=(0,)) @ psi
    psi = embed(gates.X, 1, 3, controls=(0,)) @ psi
    return psi


def run_bitflip_code(a, b, error_qubit=None):
    """Encode, optionally flip one qubit, correct, decode; return the recovered (a, b).

    The result matches the input logical amplitudes whenever at most one qubit flipped.
    """
    psi = encode_bitflip(a, b)
    if error_qubit is not None:
        psi = embed(gates.X, error_qubit, 3) @ psi
    psi = decode_bitflip(correct_bitflip(psi))
    return complex(psi[0]), complex(psi[0b100])


def run_phaseflip_code(a, b, error_qubit=None):
    """Like run_bitflip_code but protects against a single Z (phase) error.

    It is the bit-flip code conjugated by Hadamards: a phase error in the
    computational basis is a bit-flip error in the X basis, where the same parity
    machinery catches it.
    """
    psi = _Hall @ encode_bitflip(a, b)
    if error_qubit is not None:
        psi = embed(gates.Z, error_qubit, 3) @ psi
    psi = _Hall @ psi
    psi = decode_bitflip(correct_bitflip(psi))
    return complex(psi[0]), complex(psi[0b100])
