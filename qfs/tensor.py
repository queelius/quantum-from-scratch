"""Tensor-contraction gate application: the O(2^n) engine (post 7).

Applies a k-qubit gate by reshaping the statevector to a rank-n tensor and
contracting the gate over the target axes, instead of building a dense 2^n
operator. Big-endian: tensor axis q corresponds to qubit q.
"""

import numpy as np


def apply_gate(amps, U, targets, n):
    """Apply k-qubit gate U to `targets` of an n-qubit statevector by contraction.

    U is 2^k x 2^k (k = len(targets)). The order of `targets` is part of the
    contract: gate wire i (the i-th pair of row/column indices of U, big-endian)
    acts on qubit targets[i], so targets=[c, t] and targets=[t, c] with the same U
    are different operations. Returns the new length-2^n amplitude vector. Cost is
    O(2^n), versus O(4^n) to build the dense operator.
    """
    k = len(targets)
    psi = amps.reshape([2] * n)
    Ut = U.reshape([2] * (2 * k))
    psi = np.tensordot(Ut, psi, axes=(list(range(k, 2 * k)), list(targets)))
    psi = np.moveaxis(psi, list(range(k)), list(targets))
    return psi.reshape(2 ** n)
