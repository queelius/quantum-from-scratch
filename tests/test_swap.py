import numpy as np

from qfs import gates
from qfs.circuit import Circuit
from qfs.statevector import StateVector


def test_swap_exchanges_basis_state():
    # |10> should become |01> after swapping qubit 0 and qubit 1
    sv = StateVector(2).apply(gates.X, 0)        # |10>
    out = Circuit(2).swap(0, 1).run(state=sv)
    assert np.allclose(out.amps, [0, 1, 0, 0])    # |01>


def test_swap_is_three_cnots():
    swapped = Circuit(3).swap(0, 2).run().amps
    manual = (
        StateVector(3)
        .apply(gates.X, 2, controls=(0,))
        .apply(gates.X, 0, controls=(2,))
        .apply(gates.X, 2, controls=(0,))
        .amps
    )
    assert np.allclose(swapped, manual)


def test_swap_on_superposition():
    # H on qubit 0 then swap(0,1) puts the superposition on qubit 1
    out = Circuit(2).h(0).swap(0, 1).run().amps
    expected = StateVector(2).apply(gates.H, 1).amps
    assert np.allclose(out, expected)
