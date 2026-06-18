"""Oracle constructions for algorithm modules."""

import numpy as np


def bit_oracle(f, n_in):
    """Reversible oracle |x>|y> -> |x>|y XOR f(x)>; ancilla is the last qubit.

    Big-endian: the ancilla is bit 0 (least significant); the input integer x is
    the remaining high bits (input qubit 0 = MSB of x).
    """
    n = n_in + 1
    dim = 2 ** n
    op = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        x = i >> 1
        y = i & 1
        j = (x << 1) | (y ^ (int(f(x)) & 1))
        op[j, i] = 1.0
    return op


def phase_oracle(marked, n):
    """Diagonal operator that flips the sign of basis index `marked`."""
    op = np.eye(2 ** n, dtype=np.complex128)
    op[marked, marked] = -1.0
    return op
