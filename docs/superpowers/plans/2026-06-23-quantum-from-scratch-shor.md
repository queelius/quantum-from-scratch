# Quantum From Scratch (Increment 4: Shor's Algorithm, post 6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Factor an integer with Shor's algorithm: order finding via phase estimation on modular multiplication, plus the classical continued-fraction reduction, demonstrated end to end on N = 15.

**Architecture:** Two carry-forward hardening guards land first (Task 1). Then `qfs/algorithms/shor.py` adds `modmul_unitary(a, N, n_target)` (the permutation `|x> -> |a*x mod N>`), `order_from_phase` (continued fractions, via `fractions.Fraction.limit_denominator`), `shor_order` (run phase estimation on the eigenstate-superposition `|1>`, extract the period, verify), and `shor_factor` (the classical reduction from factoring to order finding). Every piece was verified empirically before this plan was written: order finding recovers r for N = 15 across bases and seeds, and `shor_factor(15)` returns `(3, 5)` deterministically.

**Tech Stack:** Python 3.11+, NumPy, pytest. Same `uv` workflow as earlier increments.

**Scope and a real limit:** This is post 6, the capstone of the pure-state arc. The dense simulator caps Shor near N = 15: its register is `t + ceil(log2 N)` qubits and the dense operators are O(4^qubits), so N = 21 (with a genuinely non-dyadic order like 6) needs the einsum engine from post 7. N = 15 is the canonical textbook example and fits comfortably. The continued-fraction half is N-independent and is unit-tested on non-dyadic phases directly. Posts 7-10 (einsum engine, density matrices, noise, error correction) remain the roadmap.

## Global Constraints

- **Qubit ordering: big-endian** (as in every prior increment). Qubit `q` is at bit position `n - 1 - q`.
- **`modmul_unitary(a, N, n_target)`** returns a 2^n_target x 2^n_target complex128 permutation: column `x` maps to `(a*x) mod N` for `x < N`, and to `x` for `x >= N` (so it stays unitary on the full register). Requires `gcd(a, N) == 1`.
- **`order_from_phase(phi, N) -> int`** returns `Fraction(phi).limit_denominator(N).denominator` (the continued-fraction convergent denominator, a candidate order).
- **`shor_order(a, N, t, trials=16, rng=None) -> int | None`** returns the smallest verified period `r` (with `a**r % N == 1`) found across `trials` phase-estimation runs, or None.
- **`shor_factor(N, t, rng=None) -> tuple | None`** returns a nontrivial factor pair of `N`.
- **Amplitudes / dtype / rng:** complex128, randomness from the passed `rng`.
- **No em-dashes or en-dashes anywhere** (commit hook enforced). Plain hyphens, commas, colons, parentheses, periods.
- **Commits** end with the two trailer lines:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01B8Jvwtmjkqe7B9iEPH1oaK`.

---

### Task 1: Hardening guards (carry-forwards from the QPE review)

**Files:**
- Modify: `qfs/dense.py` (guard in `controlled_operator`)
- Modify: `qfs/algorithms/phase_estimation.py` (guard in `phase_estimation`)
- Test: `tests/test_guards.py`

**Interfaces:**
- Consumes: the existing `controlled_operator` and `phase_estimation`.
- Produces: `controlled_operator` raises `ValueError` if `control` is in `targets`. `phase_estimation` raises `ValueError` if `eigenstate` is not normalized (L2 norm not close to 1).

- [ ] **Step 1: Write the failing tests**

`tests/test_guards.py`:
```python
import numpy as np
import pytest

from qfs import gates
from qfs.dense import controlled_operator
from qfs.algorithms.phase_estimation import phase_estimation


def test_controlled_operator_rejects_control_in_targets():
    with pytest.raises(ValueError):
        controlled_operator(gates.X, control=1, targets=[1], n=2)


def test_phase_estimation_rejects_unnormalized_eigenstate():
    U = np.array([[1, 0], [0, 1j]], dtype=complex)
    with pytest.raises(ValueError):
        phase_estimation(U, [1, 1], t=3)            # norm sqrt(2), not 1


def test_phase_estimation_accepts_normalized_eigenstate():
    U = np.array([[1, 0], [0, 1j]], dtype=complex)  # phase 1/4 on |1>
    assert np.isclose(phase_estimation(U, [0, 1], t=3, rng=np.random.default_rng(0)), 0.25)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_guards.py -v`
Expected: the two `raises` tests FAIL (no guard yet); the third passes.

- [ ] **Step 3: Add the guards**

In `qfs/dense.py`, at the top of `controlled_operator` (before building `op`):
```python
    if control in targets:
        raise ValueError(f"control qubit {control} must not be among targets {targets}")
```

In `qfs/algorithms/phase_estimation.py`, immediately after `eigenstate = np.asarray(eigenstate, dtype=np.complex128)`:
```python
    if not np.isclose(np.linalg.norm(eigenstate), 1.0):
        raise ValueError("eigenstate must be normalized (L2 norm 1)")
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_guards.py -v`
Expected: PASS. Then `uv run pytest -q` (no regressions: prior phase-estimation tests pass normalized states).

- [ ] **Step 5: Commit**

```bash
git add qfs/dense.py qfs/algorithms/phase_estimation.py tests/test_guards.py
git commit -m "feat: input guards on controlled_operator and phase_estimation"
```

---

### Task 2: The modular-multiplication unitary

**Files:**
- Create: `qfs/algorithms/shor.py`
- Test: `tests/test_shor_modmul.py`

**Interfaces:**
- Consumes: numpy.
- Produces: `modmul_unitary(a, N, n_target) -> np.ndarray` (2^n_target square, complex128). Column `x` is `|(a*x) mod N>` for `x < N`, else `|x>`.

- [ ] **Step 1: Write the failing tests**

`tests/test_shor_modmul.py`:
```python
import numpy as np

from qfs.algorithms.shor import modmul_unitary


def test_modmul_is_a_permutation_unitary():
    U = modmul_unitary(7, 15, 4)
    assert np.allclose(U @ U.conj().T, np.eye(16))
    # every column is a basis vector (a permutation matrix)
    assert np.allclose(np.sort(U.sum(axis=0)), np.ones(16))


def test_modmul_maps_correctly():
    U = modmul_unitary(7, 15, 4)
    e = np.eye(16)
    assert np.allclose(U @ e[:, 1], e[:, 7])     # 7*1 mod 15 = 7
    assert np.allclose(U @ e[:, 7], e[:, 4])     # 7*7 mod 15 = 49 mod 15 = 4
    assert np.allclose(U @ e[:, 2], e[:, 14])    # 7*2 mod 15 = 14


def test_modmul_identity_above_N():
    U = modmul_unitary(7, 15, 4)
    e = np.eye(16)
    assert np.allclose(U @ e[:, 15], e[:, 15])   # 15 >= N, fixed
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_shor_modmul.py -v`
Expected: FAIL (cannot import qfs.algorithms.shor).

- [ ] **Step 3: Implement `modmul_unitary` in `qfs/algorithms/shor.py`**

```python
"""Shor's algorithm: order finding by phase estimation, plus the classical reduction."""

import math
from fractions import Fraction

import numpy as np

from .phase_estimation import phase_estimation


def modmul_unitary(a, N, n_target):
    """Permutation |x> -> |(a*x) mod N> for x < N, identity for x >= N."""
    dim = 2 ** n_target
    op = np.zeros((dim, dim), dtype=np.complex128)
    for x in range(dim):
        op[(a * x) % N if x < N else x, x] = 1.0
    return op
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_shor_modmul.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qfs/algorithms/shor.py tests/test_shor_modmul.py
git commit -m "feat: modular-multiplication unitary for Shor"
```

---

### Task 3: Continued-fraction order extraction

**Files:**
- Modify: `qfs/algorithms/shor.py` (add `order_from_phase`)
- Test: `tests/test_shor_continued_fraction.py`

**Interfaces:**
- Consumes: `fractions.Fraction`.
- Produces: `order_from_phase(phi, N) -> int`, the denominator of the best rational approximation of `phi` with denominator at most `N`.

- [ ] **Step 1: Write the failing tests**

`tests/test_shor_continued_fraction.py`:
```python
from qfs.algorithms.shor import order_from_phase


def test_dyadic_phases():
    assert order_from_phase(0.75, 15) == 4    # 3/4
    assert order_from_phase(0.25, 15) == 4    # 1/4
    assert order_from_phase(0.5, 15) == 2     # 1/2
    assert order_from_phase(0.0, 15) == 1     # 0/1


def test_non_dyadic_phases():
    # the continued-fraction half is N-independent and handles approximate phases
    assert order_from_phase(0.833, 21) == 6   # approx 5/6
    assert order_from_phase(1 / 3, 21) == 3   # 1/3
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_shor_continued_fraction.py -v`
Expected: FAIL (order_from_phase not defined).

- [ ] **Step 3: Add `order_from_phase` to `qfs/algorithms/shor.py`**

```python
def order_from_phase(phi, N):
    """Candidate order: denominator of the best rational approximation of phi
    with denominator at most N (the continued-fraction convergent)."""
    return Fraction(phi).limit_denominator(N).denominator
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_shor_continued_fraction.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qfs/algorithms/shor.py tests/test_shor_continued_fraction.py
git commit -m "feat: continued-fraction order extraction"
```

---

### Task 4: Order finding

**Files:**
- Modify: `qfs/algorithms/shor.py` (add `shor_order`)
- Test: `tests/test_shor_order.py`

**Interfaces:**
- Consumes: numpy; `modmul_unitary`, `order_from_phase`, `phase_estimation`.
- Produces: `shor_order(a, N, t, trials=16, rng=None) -> int | None`. Runs phase estimation on `modmul_unitary(a, N, ceil(log2 N))` with the eigenstate-superposition `|1>` up to `trials` times; for each measured phase, extracts a candidate order and keeps it if `a**r % N == 1`; returns the smallest verified order (or None).

- [ ] **Step 1: Write the failing tests**

`tests/test_shor_order.py`:
```python
import numpy as np

from qfs.algorithms.shor import shor_order


def test_order_of_7_mod_15():
    assert shor_order(7, 15, t=4, rng=np.random.default_rng(0)) == 4


def test_order_of_2_mod_15():
    assert shor_order(2, 15, t=4, rng=np.random.default_rng(0)) == 4


def test_order_is_seed_robust():
    # order finding recovers r = 4 across several seeds (verified)
    for seed in range(4):
        assert shor_order(7, 15, t=4, rng=np.random.default_rng(seed)) == 4
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_shor_order.py -v`
Expected: FAIL (shor_order not defined).

- [ ] **Step 3: Add `shor_order` to `qfs/algorithms/shor.py`**

```python
def shor_order(a, N, t, trials=16, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n_target = int(np.ceil(np.log2(N)))
    U = modmul_unitary(a, N, n_target)
    one = np.zeros(2 ** n_target, dtype=np.complex128)
    one[1] = 1.0                                   # |1> is a uniform superposition of all eigenstates
    candidates = set()
    for _ in range(trials):
        phi = phase_estimation(U, one, t, rng=rng)
        r = order_from_phase(phi, N)
        if r and pow(a, r, N) == 1:
            candidates.add(r)
    return min(candidates) if candidates else None
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_shor_order.py -v`
Expected: PASS. (Verified deterministic for these seeds: 16 trials reliably surface a phase of the form s/4 with gcd(s, 4) = 1, giving r = 4.)

- [ ] **Step 5: Run the full suite and commit**

Run: `uv run pytest -q`
Expected: all prior tests plus the new order-finding tests PASS.

```bash
git add qfs/algorithms/shor.py tests/test_shor_order.py
git commit -m "feat: Shor order finding via phase estimation"
```

---

### Task 5: Factoring

**Files:**
- Modify: `qfs/algorithms/shor.py` (add `shor_factor`)
- Test: `tests/test_shor_factor.py`

**Interfaces:**
- Consumes: `math`, numpy; `shor_order`.
- Produces: `shor_factor(N, t, rng=None) -> tuple | None`. Returns a nontrivial factor pair of `N`: handle even `N` directly; otherwise for each base `a` with `gcd(a, N) > 1` return that factor, else find the order `r`, and if `r` is even and `a**(r/2) != -1 mod N`, return `gcd(a**(r/2) +/- 1, N)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_shor_factor.py`:
```python
import numpy as np

from qfs.algorithms.shor import shor_factor


def test_factor_fifteen():
    f = shor_factor(15, t=4, rng=np.random.default_rng(0))
    assert f is not None
    assert sorted(f) == [3, 5]
    assert f[0] * f[1] == 15


def test_factor_fifteen_seed_robust():
    for seed in range(3):
        f = shor_factor(15, t=4, rng=np.random.default_rng(seed))
        assert sorted(f) == [3, 5]


def test_even_number_shortcut():
    assert sorted(shor_factor(14, t=4, rng=np.random.default_rng(0))) == [2, 7]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_shor_factor.py -v`
Expected: FAIL (shor_factor not defined).

- [ ] **Step 3: Add `shor_factor` to `qfs/algorithms/shor.py`**

```python
def shor_factor(N, t, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if N % 2 == 0:
        return (2, N // 2)
    for a in range(2, N):
        g = math.gcd(a, N)
        if g != 1:
            return (g, N // g)
        r = shor_order(a, N, t, rng=rng)
        if r and r % 2 == 0:
            x = pow(a, r // 2, N)
            if x != N - 1:
                for f in (math.gcd(x - 1, N), math.gcd(x + 1, N)):
                    if 1 < f < N:
                        return (f, N // f)
    return None
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_shor_factor.py -v`
Expected: PASS (factor(15) returns (3, 5) deterministically for these seeds).

- [ ] **Step 5: Run the full suite and commit**

Run: `uv run pytest -q`
Expected: all tests PASS.

```bash
git add qfs/algorithms/shor.py tests/test_shor_factor.py
git commit -m "feat: Shor factoring via order finding"
```

> **Notebook checkpoint (post 6):** after Task 5, author `notebooks/06_shors_algorithm.py` (factoring as period finding, the modular-multiplication operator and why its eigenphases are s/r, |1> as a superposition of all eigenstates so one QPE run samples a random s/r, continued fractions to recover r, and the classical reduction to factors; demonstrate factor(15) = 3 x 5; be honest that the dense simulator caps this near N = 15 and that scaling is the next post). Render with `scripts/render-post`. Writing pass in the soul voice, not a TDD task.

## Self-Review

- **Spec coverage:** post 6 (Shor) from the roadmap is implemented end to end: the modular-multiplication unitary, the continued-fraction reduction, order finding (QPE on the eigenstate-superposition |1>), and factoring. The two QPE-review carry-forwards (control-not-in-targets, eigenstate normalization) land in Task 1. The third carry-forward (amp == 0 exact equality) is NOT triggered here because `modmul_unitary` is a permutation with exact 0/1 entries; it stays an open note for any future non-permutation U.
- **Placeholder scan:** every code step contains complete, verified code (`modmul_unitary`, `order_from_phase`, `shor_order`, `shor_factor` were run and checked: order finding returns r = 4 for N = 15 across seeds, and `shor_factor(15)` returns (3, 5) deterministically for seeds 0 to 2). No TODOs.
- **Type consistency:** `modmul_unitary(a, N, n_target) -> ndarray`, `order_from_phase(phi, N) -> int`, `shor_order(a, N, t, trials, rng) -> int|None`, `shor_factor(N, t, rng) -> tuple|None`. `shor_order` calls `modmul_unitary`, `order_from_phase`, and `phase_estimation` with matching signatures; `shor_factor` calls `shor_order`. Consistent.

## Next: the realism arc (posts 7-10)

This completes the pure-state arc (a state is always a single vector with definite amplitudes). The remaining roadmap, each its own plan:
- **Post 7 (einsum engine):** the O(2^n) tensor-contraction `apply`, replacing the dense O(4^n) `embed` for large n. Unlocks larger Shor (N = 21 and up) and general k-qubit gates. Validate against the dense path on random circuits.
- **Post 8 (density matrices):** the `DensityMatrix` engine, `partial_trace`, and the two-faces-of-rho lesson. Also where multi-qubit `measure` and the unified collapse idiom land (an earlier carry-forward).
- **Post 9 (noise and decoherence):** Kraus channels on the density matrix, watching the off-diagonals decay.
- **Post 10 (error correction):** the 3-qubit and Shor 9-qubit codes, QEC as fighting the decoherence from post 9.
