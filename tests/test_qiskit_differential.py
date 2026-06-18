import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qfs import gates
from qfs.circuit import Circuit
from qfs.statevector import StateVector
from qfs.algorithms.oracles import bit_oracle

SINGLE = {
    "h": gates.H, "x": gates.X, "y": gates.Y,
    "z": gates.Z, "s": gates.S, "t": gates.T,
}


def _random_program(n, depth, rng):
    """Return a list of ('gate', name, target) / ('ctrl', name, c, t) ops."""
    prog = []
    names = list(SINGLE)
    for _ in range(depth):
        if n >= 2 and rng.random() < 0.4:
            c, t = rng.choice(n, size=2, replace=False)
            prog.append(("ctrl", rng.choice(["cnot", "cz"]), int(c), int(t)))
        else:
            prog.append(("gate", rng.choice(names), int(rng.integers(n))))
    return prog


def _build_ours(n, prog):
    qc = Circuit(n)
    for op in prog:
        if op[0] == "gate":
            qc.apply(SINGLE[op[1]], op[2])
        elif op[1] == "cnot":
            qc.cnot(op[2], op[3])
        else:
            qc.cz(op[2], op[3])
    return qc.run().amps


def _build_qiskit(n, prog):
    # Map our qubit q to Qiskit qubit (n-1-q) so index orderings line up.
    qc = QuantumCircuit(n)
    m = lambda q: n - 1 - q
    for op in prog:
        if op[0] == "gate":
            getattr(qc, op[1])(m(op[2]))
        elif op[1] == "cnot":
            qc.cx(m(op[2]), m(op[3]))
        else:
            qc.cz(m(op[2]), m(op[3]))
    return Statevector(qc).data


def _equal_up_to_phase(a, b):
    return np.isclose(abs(np.vdot(a, b)), 1.0, atol=1e-8)


@pytest.mark.parametrize("seed", range(25))
def test_random_circuits_match_qiskit(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 5))      # 1 to 4 qubits
    depth = int(rng.integers(1, 12))
    prog = _random_program(n, depth, rng)
    ours = _build_ours(n, prog)
    theirs = _build_qiskit(n, prog)
    assert _equal_up_to_phase(ours, theirs), f"mismatch n={n} prog={prog}"


def test_bit_oracle_matches_qiskit():
    # Cross-check bit_oracle against Qiskit at the algorithm-primitive level.
    # f(x) = the MSB of the input register (= input qubit 0), so the oracle is a
    # single CX from input qubit 0 to the ancilla. Inputs go through H but the
    # ancilla stays |0>, so the oracle's permutation is observable (non-uniform
    # state) and the choice of f is order-sensitive: this pins down both the
    # ancilla placement and the big-endian input-bit ordering.
    n_in = 3
    n = n_in + 1
    f = lambda x: (x >> (n_in - 1)) & 1

    sv = StateVector(n)
    for q in range(n_in):
        sv.apply(gates.H, q)
    sv.amps = bit_oracle(f, n_in) @ sv.amps
    ours = sv.amps

    m = lambda q: n - 1 - q
    qc = QuantumCircuit(n)
    for q in range(n_in):
        qc.h(m(q))
    qc.cx(m(0), m(n_in))
    theirs = Statevector(qc).data

    assert _equal_up_to_phase(ours, theirs)
