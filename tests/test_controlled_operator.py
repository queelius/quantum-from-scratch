import numpy as np

from qfs import gates
from qfs.dense import controlled_operator, embed


def test_control_zero_is_identity_block():
    # X on target qubit 1, controlled by qubit 0; on |0xx> states it does nothing.
    op = controlled_operator(gates.X, control=0, targets=[1], n=2)
    e = np.eye(4)
    assert np.allclose(op @ e[:, 0], e[:, 0])   # |00> -> |00>
    assert np.allclose(op @ e[:, 1], e[:, 1])   # |01> -> |01>


def test_single_qubit_matches_embed():
    # controlled_operator with a 1-qubit U and one target equals embed's controlled gate
    for n in (2, 3):
        for control in range(n):
            for target in range(n):
                if target == control:
                    continue
                a = controlled_operator(gates.X, control, [target], n)
                b = embed(gates.X, target, n, controls=(control,))
                assert np.allclose(a, b)


def test_two_qubit_target_block():
    # A 2-qubit U (here the 4x4 identity-with-swap) applied to targets [1,2] when qubit 0 is 1
    swap = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]], dtype=complex)
    op = controlled_operator(swap, control=0, targets=[1, 2], n=3)
    e = np.eye(8)
    # |1 01> (index 5) should become |1 10> (index 6): control on, swap targets
    assert np.allclose(op @ e[:, 5], e[:, 6])
    # control off leaves |0 01> (index 1) alone
    assert np.allclose(op @ e[:, 1], e[:, 1])


def test_controlled_operator_is_unitary():
    op = controlled_operator(gates.H, control=0, targets=[1], n=2)
    assert np.allclose(op @ op.conj().T, np.eye(4))
