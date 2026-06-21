# Quantum From Scratch (Increment 3: Phase Estimation, post 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add quantum phase estimation to the qfs simulator: read the phase of a unitary's eigenvalue off a counting register using the QFT.

**Architecture:** One new dense-operator primitive, `controlled_operator(U, control, targets, n)`, applies a full register operator `U` to a set of target qubits conditioned on a single control qubit (the capability the QFT increment did not need). On top of it, `qfs/algorithms/phase_estimation.py` implements the standard circuit: a counting register in uniform superposition, a ladder of controlled `U**(2**k)` rotations, an inverse QFT on the counting register (the conjugate transpose of `qft_matrix`), and a measurement that reads the phase. Both pieces were verified empirically before this plan was written (dyadic phases recovered exactly for the phase gate and the T gate).

**Tech Stack:** Python 3.11+, NumPy, pytest. Same `uv` workflow as earlier increments. (No new Qiskit comparison is required here; the analytic eigenphase tests are exact.)

**Scope note:** This plan covers **post 5 (phase estimation) only**. Post 6 (Shor) and the realism arc (posts 7-10) remain the roadmap in the QFT plan, to be turned into their own plans. Shor builds directly on this increment's `controlled_operator` (for controlled modular multiplication) plus continued fractions, so phase estimation is the right stopping point.

## Global Constraints

- **Qubit ordering: big-endian.** Qubit 0 is the most significant bit; qubit `t` is at bit position `n - 1 - t`. (Same as earlier increments.)
- **Register layout for phase estimation:** the `t` counting qubits are qubits `0..t-1` (the high bits); the `m` eigenstate qubits are `t..t+m-1` (the low bits), where `m = log2(len(eigenstate))` and `n = t + m`.
- **`controlled_operator(U, control, targets, n)`** returns a 2^n x 2^n complex128 operator: identity on the `control`-is-0 block, and `U` applied to `targets` on the `control`-is-1 block. With a single-qubit `U` and one target it must equal `embed(U, target, n, controls=(control,))`.
- **`phase_estimation(U, eigenstate, t, rng=None)`** returns the estimated phase as a float in [0, 1): the measured counting register read as a big-endian binary fraction divided by 2**t.
- **Amplitudes / dtype / rng:** complex128, randomness from `self.rng` as before.
- **No em-dashes or en-dashes anywhere** (commit hook enforced). Plain hyphens, commas, colons, parentheses, periods.
- **Dependencies:** runtime numpy only.
- **Commits** end with the two trailer lines:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01B8Jvwtmjkqe7B9iEPH1oaK`.

---

### Task 1: The controlled full-register operator

**Files:**
- Modify: `qfs/dense.py` (add `controlled_operator`)
- Test: `tests/test_controlled_operator.py`

**Interfaces:**
- Consumes: numpy; cross-checks against the existing `embed`.
- Produces: `controlled_operator(U, control, targets, n) -> np.ndarray` (2^n x 2^n complex128). `U` is a 2^len(targets) x 2^len(targets) operator. The result applies `U` to the qubits in `targets` (in the order given, big-endian) when qubit `control` is 1, and is the identity when `control` is 0.

- [ ] **Step 1: Write the failing tests**

`tests/test_controlled_operator.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_controlled_operator.py -v`
Expected: FAIL (cannot import controlled_operator).

- [ ] **Step 3: Add `controlled_operator` to `qfs/dense.py`**

```python
def controlled_operator(U, control, targets, n):
    """Full 2^n operator applying register-operator U to `targets` when `control` is 1.

    U is 2^len(targets) x 2^len(targets). Big-endian: qubit q is at bit position
    (n - 1 - q). On the control-is-0 block this is the identity.
    """
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_controlled_operator.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qfs/dense.py tests/test_controlled_operator.py
git commit -m "feat: controlled full-register operator"
```

---

### Task 2: Phase estimation

**Files:**
- Create: `qfs/algorithms/phase_estimation.py`
- Test: `tests/test_phase_estimation.py`

**Interfaces:**
- Consumes: numpy; `qfs.gates`; `qfs.statevector.StateVector`; `qfs.dense.controlled_operator`; `qfs.algorithms.qft.qft_matrix`.
- Produces: `phase_estimation(U, eigenstate, t, rng=None) -> float`. `U` is a 2^m x 2^m unitary, `eigenstate` a length-2^m eigenvector of `U` with eigenvalue `exp(2j*pi*phi)`. `t` is the number of counting qubits. Returns the estimate of `phi` in [0, 1).

- [ ] **Step 1: Write the failing tests**

`tests/test_phase_estimation.py`:
```python
import numpy as np

from qfs.algorithms.phase_estimation import phase_estimation


def phase_gate(phi):
    return np.array([[1, 0], [0, np.exp(2j * np.pi * phi)]], dtype=complex)


def test_dyadic_phases_exact():
    # |1> is the eigenstate with eigenvalue exp(2j*pi*phi); 4 counting qubits
    # resolve any multiple of 1/16 exactly.
    for phi in (0.0, 0.25, 0.5, 0.125, 0.375, 0.0625):
        est = phase_estimation(phase_gate(phi), [0, 1], t=4, rng=np.random.default_rng(0))
        assert np.isclose(est, phi), f"phi={phi} est={est}"


def test_t_gate_eighth():
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)  # phase 1/8
    assert np.isclose(phase_estimation(T, [0, 1], t=3, rng=np.random.default_rng(0)), 0.125)


def test_eigenvalue_one_gives_zero_phase():
    # |0> is the eigenstate of the phase gate with eigenvalue 1 (phi = 0)
    est = phase_estimation(phase_gate(0.5), [1, 0], t=4, rng=np.random.default_rng(0))
    assert np.isclose(est, 0.0)


def test_phase_is_seed_independent_for_dyadic():
    # exact dyadic phases collapse the counting register deterministically
    a = phase_estimation(phase_gate(0.375), [0, 1], t=4, rng=np.random.default_rng(1))
    b = phase_estimation(phase_gate(0.375), [0, 1], t=4, rng=np.random.default_rng(99))
    assert a == b == 0.375
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_phase_estimation.py -v`
Expected: FAIL (cannot import phase_estimation).

- [ ] **Step 3: Implement `qfs/algorithms/phase_estimation.py`**

```python
"""Quantum phase estimation: read an eigenvalue's phase off a counting register."""

import numpy as np

from .. import gates
from ..statevector import StateVector
from ..dense import controlled_operator
from .qft import qft_matrix


def phase_estimation(U, eigenstate, t, rng=None):
    eigenstate = np.asarray(eigenstate, dtype=np.complex128)
    m = int(round(np.log2(len(eigenstate))))
    n = t + m
    targets = list(range(t, n))

    # counting register |0...0> (high t bits), target register = eigenstate (low m bits)
    amps = np.zeros(2 ** n, dtype=np.complex128)
    amps[: len(eigenstate)] = eigenstate
    sv = StateVector.from_amplitudes(amps, rng=rng)

    for j in range(t):
        sv.apply(gates.H, j)                       # uniform superposition on counting qubits

    for j in range(t):                             # controlled-U^(2^(t-1-j)) ladder
        u_power = np.linalg.matrix_power(U, 2 ** (t - 1 - j))
        sv.amps = controlled_operator(u_power, j, targets, n) @ sv.amps

    # inverse QFT on the counting register (it is the high block, so kron with identity)
    inverse_qft = np.kron(qft_matrix(t).conj().T, np.eye(2 ** m, dtype=np.complex128))
    sv.amps = inverse_qft @ sv.amps

    bits = [sv.measure(j) for j in range(t)]        # read the counting register
    value = sum(b << (t - 1 - i) for i, b in enumerate(bits))
    return value / 2 ** t
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_phase_estimation.py -v`
Expected: PASS. (Exact dyadic phases collapse the counting register deterministically, so the dyadic tests are seed-independent.)

- [ ] **Step 5: Run the full suite and commit**

Run: `uv run pytest -q`
Expected: all earlier tests plus the new phase-estimation tests PASS.

```bash
git add qfs/algorithms/phase_estimation.py tests/test_phase_estimation.py
git commit -m "feat: quantum phase estimation"
```

> **Notebook checkpoint (post 5):** after Task 2, author `notebooks/05_phase_estimation.py` (the counting register as a ruler for the phase, the controlled-power ladder, the inverse QFT as the readout, exact dyadic phases vs the more honest non-dyadic case where the estimate is a distribution that peaks at the nearest bin), then render with `scripts/render-post`. Consider showing a non-dyadic phase (e.g. 1/3) where more counting qubits sharpen the estimate. Writing pass in the soul voice, not a TDD task.

## Self-Review

- **Spec coverage:** post 5 (phase estimation) from the roadmap is implemented with the new `controlled_operator` primitive it requires, plus the QPE pipeline (counting register, controlled-power ladder, inverse QFT, measurement). Covered. The roadmap noted "test against unitaries with known eigenphases (a phase gate, a Z rotation)": the phase gate and T gate tests do exactly this.
- **Placeholder scan:** every code step contains complete, verified code (`controlled_operator` and `phase_estimation` were both run and checked: dyadic phases for the phase gate and the T gate's 1/8 were recovered exactly before this plan was written). No TODOs.
- **Type consistency:** `controlled_operator(U, control, targets, n) -> ndarray` is consistent between dense.py (Task 1) and its use in phase_estimation (Task 2). `phase_estimation(U, eigenstate, t, rng=None) -> float`. The inverse QFT uses `qft_matrix(t).conj().T` from the QFT increment. The cross-check against `embed` reuses the existing signature.

## Next (post 6 and beyond)

- **Post 6 (Shor):** order finding is phase estimation on the modular-multiplication unitary `|x> -> |a*x mod M>`, whose controlled powers reuse `controlled_operator`. The new piece is building that permutation unitary (and its powers efficiently), plus the classical continued-fractions step that turns the measured phase into the period. Then factoring reduces to order finding. Likely two posts.
- **Carry-forward (still open):** extract a `circuit_unitary` test helper to `tests/helpers.py` (the QFT increment defined it inline); phase estimation did not need it (its tests check the returned float directly), but Shor's order-finding tests may.
