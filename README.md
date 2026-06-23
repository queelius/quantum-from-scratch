# Quantum Computing From Scratch

A quantum computer simulator built from nothing, in pure Python on top of NumPy, to
understand how the thing actually works. No quantum-computing framework, no black
boxes. A qubit is a unit vector in C^2, a gate is a matrix, measurement is sampling,
and everything that gets called spooky is something you can watch happen inside an
array.

I built this to learn quantum computing properly, by constructing it rather than
driving a library that hides the linear algebra. Each module is one increment of a
blog series on [metafunctor.com](https://metafunctor.com); the `notebooks/` are the
companion notebooks for those posts.

## The series

| # | Post | What gets built |
|---|------|-----------------|
| 0 | What is a qubit | statevector, gates, measurement, the Bloch sphere |
| 1 | Many qubits and entanglement | n-qubit register, CNOT, Bell states |
| 2 | Circuits and interference | the Circuit type, Deutsch-Jozsa, Bernstein-Vazirani |
| 3 | Grover's search | oracles, the diffusion operator, amplitude amplification |
| 4 | The Quantum Fourier Transform | `qft_matrix`, `qft_circuit`, SWAP |
| 5 | Phase estimation | the controlled-operator primitive, QPE |
| 6 | Shor's algorithm | modular multiplication, order finding, factoring 15 |
| 7 | Scaling the simulator | the O(2^n) tensor-contraction engine |
| 8 | Mixed states and the density matrix | the DensityMatrix engine, partial trace |
| 9 | Noise and decoherence | Kraus channels: damping, depolarizing, flips |
| 10 | Quantum error correction | the 3-qubit bit-flip and phase-flip codes |

Posts 0 to 6 are the pure-state arc; 7 to 10 are the realism arc (scaling, mixed
states, noise, and correcting it).

## Library layout

```
qfs/
  gates.py         single-qubit gates (I,X,Y,Z,H,S,T) and rotations
  dense.py         embedding small gates into full 2^n operators
  statevector.py   StateVector: apply, measure, sample, expectation
  circuit.py       Circuit: a recorded, runnable sequence of gates
  tensor.py        apply_gate: the O(2^n) contraction engine (post 7)
  density.py       DensityMatrix: mixed states, partial trace (post 8)
  channels.py      noise channels as Kraus operators (post 9)
  qec.py           the 3-qubit repetition codes (post 10)
  viz.py           Bloch sphere, probability and density-matrix plots
  algorithms/      Deutsch-Jozsa, Bernstein-Vazirani, Grover, QFT, QPE, Shor
```

## Running it

The project uses [uv](https://docs.astral.sh/uv/).

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest            # the full suite (152 tests)
```

The companion notebooks are jupytext-paired `.py` files. To execute one and render it
to markdown plus figures:

```bash
scripts/render-post notebooks/00_what_is_a_qubit.py
```

## On correctness

The whole library is checked two ways. Canonical results are tested analytically (Bell
probabilities, Grover success probability, the QFT against the classical FFT, exact
eigenphases, factor(15) = 3 x 5). And the simulator is differentially tested against
[Qiskit](https://www.ibm.com/quantum/qiskit) as an independent oracle: random circuits
run through both must agree up to global phase. Qiskit is the verifier here, never the
implementation.

## What is not here

This is a teaching simulator, not a competitive one. It does exact statevector
simulation, so a laptop tops out around 30 qubits (the dense operators in the early
modules are heavier and cap lower). Shor's algorithm runs on N = 15; larger numbers
need the tensor engine and more patience. There is no tensor-network or
approximate-simulation path, and no real-hardware backend. Those are deliberate
omissions: the point was to see the machine, not to outrun Qiskit.

## License

MIT. Use it however you like.
