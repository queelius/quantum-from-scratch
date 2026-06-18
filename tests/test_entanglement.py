import numpy as np

from qfs import gates
from qfs.statevector import StateVector


def bell():
    # |Phi+> = (|00> + |11>)/sqrt(2): H on qubit 0, then CNOT(control=0, target=1).
    return StateVector(2).apply(gates.H, 0).apply(gates.X, 1, controls=(0,))


def test_bell_amplitudes():
    sv = bell()
    assert np.allclose(sv.amps, [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])


def test_bell_is_not_a_product_state():
    # A product state (a|0>+b|1>) x (c|0>+d|1>) has amps [ac, ad, bc, bd],
    # so amps[0]*amps[3] == amps[1]*amps[2]. Entanglement breaks this.
    a = bell().amps
    assert not np.isclose(a[0] * a[3], a[1] * a[2])


def test_bell_measurements_are_correlated():
    rng = np.random.default_rng(0)
    for _ in range(50):
        sv = StateVector(2, rng=rng).apply(gates.H, 0).apply(gates.X, 1, controls=(0,))
        b0 = sv.measure(0)
        b1 = sv.measure(1)
        assert b0 == b1  # the two qubits always agree
