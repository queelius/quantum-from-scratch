import numpy as np

from qfs import gates
from qfs.circuit import Circuit
from qfs.statevector import StateVector


def test_circuit_reproduces_manual_application():
    qc = Circuit(2).h(0).cnot(0, 1)
    via_circuit = qc.run().amps
    manual = StateVector(2).apply(gates.H, 0).apply(gates.X, 1, controls=(0,)).amps
    assert np.allclose(via_circuit, manual)


def test_circuit_length_and_chaining():
    qc = Circuit(3).h(0).x(1).cz(0, 2)
    assert len(qc) == 3


def test_circuit_runs_on_supplied_state():
    qc = Circuit(1).x(0)
    sv = StateVector(1)
    out = qc.run(state=sv)
    assert out is sv
    assert np.allclose(sv.amps, [0, 1])


def test_bell_via_circuit():
    sv = Circuit(2).h(0).cnot(0, 1).run()
    assert np.allclose(sv.amps, [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
