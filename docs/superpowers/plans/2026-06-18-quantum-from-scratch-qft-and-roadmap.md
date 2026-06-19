# Quantum From Scratch (Increment 2: QFT, post 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Quantum Fourier Transform to the qfs simulator, both as a direct matrix and as a gate-by-gate circuit, validated to agree exactly and to match Qiskit.

**Architecture:** A new `qfs/algorithms/qft.py` provides `qft_matrix(n)` (direct construction) and `qft_circuit(n)` (Hadamard + controlled-phase ladder + bit-reversal swaps, built on the existing `Circuit`). A `swap` is added to `Circuit` (three CNOTs). The two QFT forms are cross-checked against each other, against the classical FFT, and against Qiskit. This closes the carry-forward from increment 1: QFT's bit-reversal is the classic endianness trap, so it gets the differential oracle the foundations review asked for.

**Tech Stack:** Python 3.11+, NumPy, pytest, qiskit (dev-only oracle). Same `uv` workflow as increment 1.

**Scope note:** This plan covers **post 4 (QFT) only**, with full TDD detail. Posts 5-10 (phase estimation, Shor, the einsum engine, density matrices, noise, error correction) are a documented roadmap at the end of this file, to be turned into their own plans one chunk at a time, exactly as posts 0-3 were planned separately from the start. This keeps each plan executable and its code verified, rather than shipping hand-waved code for the hard parts (Shor and QEC especially).

## Global Constraints

- **Qubit ordering: big-endian.** Qubit 0 is the most significant bit; qubit `t` is at bit position `n - 1 - t`; basis index `sum(b_t * 2**(n-1-t))`. (Same as increment 1.)
- **`qft_matrix(n)`** returns the unitary `F` with `F[j, k] = exp(2j*pi*j*k / 2**n) / sqrt(2**n)` as `complex128`. This satisfies `F @ v == sqrt(N) * numpy.fft.ifft(v)` (verified for n=1..3).
- **`qft_circuit(n)`** must produce a unitary equal to `qft_matrix(n)` **exactly** (not merely up to global phase): the final bit-reversal swap network is required, verified for n=1..4.
- **Amplitudes / dtype / rng:** as in increment 1 (complex128, `self.rng`).
- **No em-dashes or en-dashes anywhere** (commit hook enforced). Use commas, colons, parentheses, periods, plain hyphens.
- **Dependencies:** runtime numpy only; qiskit is dev-only.
- **Commits** end with the two trailer lines:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01B8Jvwtmjkqe7B9iEPH1oaK`.

---

### Task 1: The QFT as a direct matrix

**Files:**
- Create: `qfs/algorithms/qft.py`
- Test: `tests/test_qft.py`

**Interfaces:**
- Consumes: numpy.
- Produces: `qft_matrix(n) -> np.ndarray` (2^n x 2^n complex128), the QFT unitary.

- [ ] **Step 1: Write the failing tests**

`tests/test_qft.py`:
```python
import numpy as np

from qfs.algorithms.qft import qft_matrix


def test_qft_is_unitary():
    for n in (1, 2, 3):
        F = qft_matrix(n)
        assert np.allclose(F @ F.conj().T, np.eye(2 ** n))


def test_qft_matches_scaled_inverse_fft():
    rng = np.random.default_rng(0)
    for n in (1, 2, 3):
        N = 2 ** n
        v = rng.normal(size=N) + 1j * rng.normal(size=N)
        assert np.allclose(qft_matrix(n) @ v, np.sqrt(N) * np.fft.ifft(v))


def test_qft_of_zero_is_uniform():
    N = 8
    F = qft_matrix(3)
    col0 = F @ np.eye(N)[:, 0]            # QFT of |000>
    assert np.allclose(col0, np.ones(N) / np.sqrt(N))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_qft.py -v`
Expected: FAIL (cannot import qfs.algorithms.qft).

- [ ] **Step 3: Implement `qfs/algorithms/qft.py`**

```python
"""Quantum Fourier Transform: direct matrix and gate circuit."""

import numpy as np


def qft_matrix(n):
    """The n-qubit QFT unitary: F[j, k] = exp(2j*pi*j*k / 2**n) / sqrt(2**n)."""
    N = 2 ** n
    js = np.arange(N)
    return np.exp(2j * np.pi * np.outer(js, js) / N) / np.sqrt(N)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_qft.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qfs/algorithms/qft.py tests/test_qft.py
git commit -m "feat: QFT as a direct matrix"
```

---

### Task 2: SWAP on the Circuit

**Files:**
- Modify: `qfs/circuit.py` (add a `swap` method to `Circuit`)
- Test: `tests/test_swap.py`

**Interfaces:**
- Consumes: existing `Circuit` (Task 8 of increment 1) and `embed`/CNOT semantics.
- Produces: `Circuit.swap(self, a, b) -> Circuit`: appends three CNOTs (a->b, b->a, a->b), which exchange qubits a and b. Chainable.

- [ ] **Step 1: Write the failing tests**

`tests/test_swap.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_swap.py -v`
Expected: FAIL (Circuit has no attribute 'swap').

- [ ] **Step 3: Add `swap` to `qfs/circuit.py`**

Add this method inside the `Circuit` class:
```python
    def swap(self, a, b):
        return self.cnot(a, b).cnot(b, a).cnot(a, b)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_swap.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qfs/circuit.py tests/test_swap.py
git commit -m "feat: SWAP on Circuit (three CNOTs)"
```

---

### Task 3: The QFT as a gate circuit

**Files:**
- Modify: `qfs/algorithms/qft.py` (add `qft_circuit`)
- Test: `tests/test_qft_circuit.py`
- Test: `tests/test_qft_qiskit.py` (Qiskit differential, dev-only)

**Interfaces:**
- Consumes: `qfs.gates`, `qfs.circuit.Circuit`, `qfs.algorithms.qft.qft_matrix`.
- Produces: `qft_circuit(n) -> Circuit`: a `Circuit` of `n` qubits that, when run from |0..0> via `.run(state)`, applies the QFT. The construction is Hadamard on each qubit `j`, then controlled-phase rotations of angle `2*pi / 2**(k-j+1)` controlled by each later qubit `k`, then a bit-reversal swap network. The resulting unitary equals `qft_matrix(n)` exactly.

- [ ] **Step 1: Write the failing tests**

`tests/test_qft_circuit.py`:
```python
import numpy as np

from qfs.algorithms.qft import qft_matrix, qft_circuit
from qfs.statevector import StateVector


def _circuit_unitary(circ, n):
    # assemble the circuit's operator by running it on each basis state
    cols = []
    for k in range(2 ** n):
        sv = StateVector.from_amplitudes(np.eye(2 ** n)[:, k])
        cols.append(circ.run(state=sv).amps)
    return np.column_stack(cols)


def test_qft_circuit_matches_matrix_exactly():
    for n in (1, 2, 3, 4):
        U = _circuit_unitary(qft_circuit(n), n)
        assert np.allclose(U, qft_matrix(n))


def test_qft_circuit_of_zero_is_uniform():
    sv = qft_circuit(3).run(state=StateVector(3))
    assert np.allclose(sv.amps, np.ones(8) / np.sqrt(8))
```

`tests/test_qft_qiskit.py`:
```python
import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator

from qfs.algorithms.qft import qft_matrix


def _equal_up_to_phase(a, b):
    return np.isclose(abs(np.vdot(a.flatten(), b.flatten())) /
                      (np.linalg.norm(a) * np.linalg.norm(b)), 1.0, atol=1e-8)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_qft_matches_qiskit(n):
    # QFTGate is the current (non-deprecated) Qiskit QFT. Its operator equals our
    # big-endian qft_matrix up to a global phase (verified for n=1..4, qiskit 2.4).
    ours = qft_matrix(n)
    theirs = Operator(QFTGate(n)).data
    assert _equal_up_to_phase(ours, theirs)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_qft_circuit.py tests/test_qft_qiskit.py -v`
Expected: FAIL (qft_circuit not defined).

- [ ] **Step 3: Add `qft_circuit` to `qfs/algorithms/qft.py`**

```python
from ..circuit import Circuit
from .. import gates


def qft_circuit(n):
    """The QFT as a gate circuit: Hadamards, a controlled-phase ladder, and a
    bit-reversal swap network. Equals qft_matrix(n) exactly."""
    qc = Circuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.apply(gates.phase(2 * np.pi / 2 ** (k - j + 1)), j, controls=(k,))
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_qft_circuit.py tests/test_qft_qiskit.py -v`
Expected: PASS. The `qft_circuit == qft_matrix` exact-match test is the primary correctness gate; the `QFTGate` comparison (verified to match for n=1..4 on qiskit 2.4) is the independent Qiskit cross-check. If only the Qiskit test fails, it is a qiskit-version API drift in `QFTGate`, not a bug in our circuit.

- [ ] **Step 5: Run the full suite and commit**

Run: `uv run pytest -q`
Expected: all increment-1 tests plus the new QFT tests PASS.

```bash
git add qfs/algorithms/qft.py tests/test_qft_circuit.py tests/test_qft_qiskit.py
git commit -m "feat: QFT as a gate circuit, validated against the matrix and Qiskit"
```

> **Notebook checkpoint (post 4):** after Task 3, author `notebooks/04_the_quantum_fourier_transform.py` (the QFT as the change of basis from computational to frequency, the controlled-phase ladder, why the swaps, and the FFT connection), then render with `scripts/render-post`. Writing pass in the soul voice, not a TDD task.

## Self-Review

- **Spec coverage:** post 4 (QFT) from the design spec section 8 is implemented as both matrix and circuit, with the FFT cross-check and the Qiskit differential the foundations review asked for. Covered.
- **Placeholder scan:** every code step contains complete, verified code (the matrix, the circuit, and the swap were all checked against `qft_matrix` for n=1..4 and against `sqrt(N)*ifft` before this plan was written). No TODOs.
- **Type consistency:** `qft_matrix(n) -> ndarray`, `qft_circuit(n) -> Circuit`, `Circuit.swap(a, b) -> Circuit` are consistent across tasks. `qft_circuit` uses `Circuit.apply(U, target, controls)` and `Circuit.swap`, both defined before use.

---

## Roadmap: posts 5-10 (each its own future plan)

These are sequenced with the new simulator capability each one forces. Turn each into its own plan when the prior chunk is solid, the same way posts 0-3 and post 4 were planned separately.

- **Post 5: Phase estimation.** New capability needed: a *controlled full-register operator*, applying a controlled `U**(2**j)` (U a register unitary) conditioned on one counting qubit. Add `controlled_operator(U, control, targets, n)` (block structure: identity on the control=0 block, U on the control=1 block). Then QPE is: counting register in uniform superposition, the controlled-power ladder, the inverse QFT (`qft_circuit(n)` run in reverse, or `qft_matrix(n).conj().T`), measure. Test against unitaries with known eigenphases (a phase gate, a Z rotation).

- **Post 6: Shor's algorithm.** The capstone of the pure-state arc, and the hardest. New capability: a modular-multiplication unitary `|x> -> |a*x mod M>` built as a permutation matrix, plus its controlled powers (reuse post 5's controlled_operator). Then order-finding is QPE on that unitary, and the classical post-processing is continued fractions on the measured phase. Likely two posts (the quantum order-finding, then the classical reduction from factoring to order-finding). Test on small M (15, 21) with known factors.

- **Post 7: Scaling the simulator (the einsum engine).** New capability: an O(2**n) gate application that reshapes the statevector to a rank-n tensor and contracts the gate over the target axes (`np.einsum` / `np.tensordot`), replacing the O(4**n) dense `embed` for large n. This is the deferred "engine B" from the foundations spec. Build it as an alternate `apply` path, validate it against the dense path for random circuits (they must agree), and show the qubit count it unlocks (the "obvious simulator dies at 12 qubits" post). Also the place to add general k-qubit gates and a native SWAP.

- **Post 8: Mixed states and density matrices.** The big structural addition: a `DensityMatrix` engine (2**n x 2**n), `apply` as `U @ rho @ U.conj().T`, `measure`, `expectation`, and `partial_trace(keep)`. The pedagogical core is showing the two faces of rho (a classical mixture vs the partial trace of an entangled pure state) produce the same matrix. Cross-validate against `StateVector` on noiseless circuits (must agree), per the spec invariant.

- **Post 9: Noise and decoherence.** New capability: `Channel` via Kraus operators applied to a `DensityMatrix` (`sum_k K_k rho K_k.conj().T`): depolarizing, amplitude damping, phase damping, bit-flip and phase-flip. The visual payoff is the density-matrix heatmap with the off-diagonals decaying. Validate channels against analytic decay rates.

- **Post 10: A taste of error correction.** The 3-qubit bit-flip code (encode, syndrome measurement, correction), then the Shor 9-qubit code as a stretch. QEC framed as actively fighting the decoherence from post 9. Density-matrix memory caps the qubit count, which is fine: these toy codes use about 9 qubits or fewer.

**Carry-forwards already logged (from the increment-1 final review):**
- The differential oracle now reaches algorithm code paths (post 4 adds the QFT-vs-Qiskit check; extend per post).
- Unify the two collapse idioms in `StateVector` when `measure(qubits)` (multi-qubit measurement) lands, likely in post 8.
- Consider a `from_amplitudes` normalization option when the density-matrix arc needs trace-1 maintenance.
