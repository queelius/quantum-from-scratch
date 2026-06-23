"""Dense operator construction: embed small gates into full 2^n operators."""

import numpy as np


def kron_embed(U, target, n):
    """Embed single-qubit U on `target` via Kronecker products (big-endian)."""
    op = np.array([[1.0 + 0j]])
    for q in range(n):
        op = np.kron(op, U if q == target else np.eye(2, dtype=np.complex128))
    return op


def embed(U, target, n, controls=()):
    """Full 2^n operator: apply single-qubit U to `target` when all `controls` are 1.

    Big-endian: qubit q lives at bit position (n - 1 - q). With controls=() this
    is the plain single-qubit embedding; with controls it is a controlled-U.
    """
    dim = 2 ** n
    op = np.zeros((dim, dim), dtype=np.complex128)
    tpos = n - 1 - target
    cpos = [n - 1 - c for c in controls]
    for col in range(dim):
        if all((col >> p) & 1 for p in cpos):
            tbit = (col >> tpos) & 1
            for out in (0, 1):
                amp = U[out, tbit]
                if amp == 0:
                    continue
                row = (col & ~(1 << tpos)) | (out << tpos)
                op[row, col] += amp
        else:
            op[col, col] = 1.0
    return op


def controlled_operator(U, control, targets, n):
    """Full 2^n operator applying register-operator U to `targets` when `control` is 1.

    U is 2^len(targets) x 2^len(targets). Big-endian: qubit q is at bit position
    (n - 1 - q). On the control-is-0 block this is the identity.
    """
    if control in targets:
        raise ValueError(f"control qubit {control} must not be among targets {targets}")
    dim = 2 ** n
    op = np.zeros((dim, dim), dtype=np.complex128)
    cpos = n - 1 - control
    m = len(targets)
    tpos = [n - 1 - t for t in targets]
    for col in range(dim):
        if not ((col >> cpos) & 1):
            op[col, col] = 1.0
            continue
        r = 0
        for tp in tpos:
            r = (r << 1) | ((col >> tp) & 1)
        for rp in range(2 ** m):
            amp = U[rp, r]
            if amp == 0:
                continue
            row = col
            for b, tp in enumerate(tpos):
                bit = (rp >> (m - 1 - b)) & 1
                row = (row & ~(1 << tp)) | (bit << tp)
            op[row, col] += amp
    return op
