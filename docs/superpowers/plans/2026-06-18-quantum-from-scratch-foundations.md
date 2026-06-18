# Quantum From Scratch (Foundations: posts 0-3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean, tested, from-scratch Python statevector quantum simulator that takes a reader from "what is a qubit" through Grover's search, and validate it against Qiskit.

**Architecture:** A small NumPy-backed library. Gate matrices live in `gates.py`; a dense operator builder in `dense.py` embeds small gates into full 2^n operators (handling single-qubit and controlled gates uniformly); `StateVector` holds amplitudes and exposes apply/measure/sample/expectation; `Circuit` records ops and runs them; algorithm modules (Deutsch-Jozsa, Bernstein-Vazirani, Grover) build on that core. Qiskit is a test-only oracle.

**Tech Stack:** Python 3.11+, NumPy, pytest, matplotlib (viz), jupytext + Jupyter (notebooks), qiskit (dev-only differential oracle), uv (env/runner).

## Global Constraints

- **Qubit ordering: big-endian.** Qubit 0 is the most significant bit. For n qubits, qubit `t` occupies bit position `n - 1 - t`. The basis state |b0 b1 ... b_{n-1}> has integer index `sum(b_t * 2**(n-1-t))`.
- **Amplitudes:** `StateVector.amps` is a 1-D `numpy.ndarray` of dtype `complex128`, length `2**n`. `amps[i]` is the amplitude of basis index `i`.
- **Randomness:** every stochastic method takes its randomness from a `numpy.random.Generator` stored on the state (`self.rng`). Tests construct states with `rng=np.random.default_rng(SEED)` for reproducibility.
- **Package name:** `qfs`. Layout: package dir `qfs/` at repo root; tests in `tests/`; notebooks in `notebooks/`.
- **No em-dashes or en-dashes anywhere** (repo house style is enforced by a commit hook). Use commas, colons, parentheses, periods, and plain hyphens.
- **Dependencies:** runtime depends only on `numpy`. `matplotlib`, `qiskit`, `jupytext`, `jupyter` are optional `dev` extras.
- **Commits:** every task ends with a commit. Commit message body ends with the two trailer lines:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01B8Jvwtmjkqe7B9iEPH1oaK`. (Trailers omitted from the per-step examples below for brevity; add them.)

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `qfs/__init__.py`
- Create: `notebooks/.gitkeep`
- Create: `.gitignore`
- Test: `tests/test_scaffolding.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an installable `qfs` package importable as `import qfs`, exposing `qfs.__version__` (str). The `uv` environment with `dev` extras (numpy, pytest, matplotlib, qiskit, jupytext) for all later tasks.

- [ ] **Step 1: Write the failing test**

`tests/test_scaffolding.py`:
```python
def test_package_imports():
    import qfs

    assert isinstance(qfs.__version__, str)
    assert qfs.__version__
```

- [ ] **Step 2: Create the package and project files**

`qfs/__init__.py`:
```python
"""Quantum From Scratch: a pedagogical quantum simulator."""

__version__ = "0.0.1"
```

`pyproject.toml`:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "qfs"
version = "0.0.1"
description = "Quantum computing from scratch: a pedagogical statevector and density-matrix simulator"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26"]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "matplotlib>=3.8",
    "qiskit>=1.0",
    "jupytext>=1.16",
    "jupyter",
]

[tool.hatch.build.targets.wheel]
packages = ["qfs"]
```

`.gitignore`:
```gitignore
__pycache__/
*.pyc
.venv/
.pytest_cache/
.ipynb_checkpoints/
notebooks/*.ipynb
```

Create the notebooks placeholder:
```bash
mkdir -p notebooks && touch notebooks/.gitkeep
```

- [ ] **Step 3: Create the environment and install**

Run:
```bash
uv venv
uv pip install -e ".[dev]"
```
Expected: resolves and installs numpy, pytest, matplotlib, qiskit, jupytext, jupyter; `qfs` installed editable.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_scaffolding.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml qfs/ notebooks/ .gitignore tests/test_scaffolding.py
git commit -m "chore: scaffold qfs package, tooling, and smoke test"
```

---

### Task 2: Gate library

**Files:**
- Create: `qfs/gates.py`
- Test: `tests/test_gates.py`

**Interfaces:**
- Consumes: numpy.
- Produces, all 2x2 `complex128` numpy arrays / functions returning them:
  - Constants: `I, X, Y, Z, H, S, T`.
  - Parametric: `Rx(theta)`, `Ry(theta)`, `Rz(theta)`, `phase(lam)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_gates.py`:
```python
import numpy as np
import pytest

from qfs import gates


def _is_unitary(U):
    return np.allclose(U @ U.conj().T, np.eye(U.shape[0]))


@pytest.mark.parametrize("U", [gates.I, gates.X, gates.Y, gates.Z, gates.H, gates.S, gates.T])
def test_constant_gates_are_unitary(U):
    assert U.shape == (2, 2)
    assert U.dtype == np.complex128
    assert _is_unitary(U)


def test_known_identities():
    assert np.allclose(gates.X @ gates.X, gates.I)
    assert np.allclose(gates.H @ gates.H, gates.I)
    assert np.allclose(gates.H @ gates.Z @ gates.H, gates.X)
    assert np.allclose(gates.S @ gates.S, gates.Z)
    assert np.allclose(gates.T @ gates.T, gates.S)


@pytest.mark.parametrize("g", [gates.Rx, gates.Ry, gates.Rz, gates.phase])
def test_parametric_gates_are_unitary(g):
    assert _is_unitary(g(0.7))


def test_rotation_values():
    assert np.allclose(gates.Rx(np.pi), -1j * gates.X)
    assert np.allclose(gates.Rz(np.pi), -1j * gates.Z)
    assert np.allclose(gates.phase(np.pi), gates.Z)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_gates.py -v`
Expected: FAIL (ModuleNotFoundError: qfs.gates).

- [ ] **Step 3: Implement `qfs/gates.py`**

```python
"""Single-qubit gate matrices (big-endian convention, complex128)."""

import numpy as np

I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)


def Rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def Ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def Rz(theta):
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=np.complex128,
    )


def phase(lam):
    return np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=np.complex128)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_gates.py -v`
Expected: PASS (all parametrized cases green).

- [ ] **Step 5: Commit**

```bash
git add qfs/gates.py tests/test_gates.py
git commit -m "feat: single-qubit gate library with unitarity and identity tests"
```

---

### Task 3: Dense operator embedding

**Files:**
- Create: `qfs/dense.py`
- Test: `tests/test_dense.py`

**Interfaces:**
- Consumes: numpy; `qfs.gates`.
- Produces:
  - `kron_embed(U, target, n) -> np.ndarray` (2^n x 2^n): single-qubit gate `U` on `target` via Kronecker products (pedagogical, used in post 0).
  - `embed(U, target, n, controls=()) -> np.ndarray` (2^n x 2^n): general dense operator applying single-qubit `U` to `target`, conditioned on every qubit in `controls` being 1. With `controls=()` this equals `kron_embed`. With one control it is a controlled-U (e.g. CNOT = `embed(X, t, n, controls=(c,))`).

- [ ] **Step 1: Write the failing tests**

`tests/test_dense.py`:
```python
import numpy as np

from qfs import gates
from qfs.dense import embed, kron_embed


def test_kron_embed_single_qubit():
    # H on qubit 0 of a 1-qubit system is just H.
    assert np.allclose(kron_embed(gates.H, 0, 1), gates.H)
    # X on qubit 1 of a 2-qubit system is I kron X.
    assert np.allclose(kron_embed(gates.X, 1, 2), np.kron(gates.I, gates.X))


def test_embed_matches_kron_when_no_controls():
    for target in range(3):
        assert np.allclose(embed(gates.H, target, 3), kron_embed(gates.H, target, 3))


def test_embed_cnot_truth_table():
    # control = qubit 0, target = qubit 1, big-endian basis |q0 q1>.
    cnot = embed(gates.X, 1, 2, controls=(0,))
    e = np.eye(4)
    # |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>
    assert np.allclose(cnot @ e[:, 0], e[:, 0])
    assert np.allclose(cnot @ e[:, 1], e[:, 1])
    assert np.allclose(cnot @ e[:, 2], e[:, 3])
    assert np.allclose(cnot @ e[:, 3], e[:, 2])


def test_embed_is_unitary():
    op = embed(gates.X, 2, 3, controls=(0, 1))  # Toffoli
    assert np.allclose(op @ op.conj().T, np.eye(8))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_dense.py -v`
Expected: FAIL (cannot import qfs.dense).

- [ ] **Step 3: Implement `qfs/dense.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_dense.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qfs/dense.py tests/test_dense.py
git commit -m "feat: dense gate embedding (kron + general controlled operator)"
```

---

### Task 4: StateVector core and gate application

**Files:**
- Create: `qfs/statevector.py`
- Test: `tests/test_statevector.py`

**Interfaces:**
- Consumes: numpy; `qfs.dense.embed`.
- Produces: class `StateVector` with:
  - `__init__(self, n, rng=None)`: ground state |0...0>; `rng` defaults to `np.random.default_rng()`.
  - `from_amplitudes(cls, amps, rng=None)` classmethod: build from an explicit length-2^n array (inferring n).
  - `apply(self, U, target, controls=()) -> StateVector`: applies `embed(U, target, n, controls)` to `amps` in place; returns self (chainable).
  - `probabilities(self) -> np.ndarray`: `abs(amps)**2`.
  - attribute `amps` (complex128 array) and `n` (int).

- [ ] **Step 1: Write the failing tests**

`tests/test_statevector.py`:
```python
import numpy as np

from qfs import gates
from qfs.statevector import StateVector


def test_ground_state():
    sv = StateVector(3)
    assert sv.n == 3
    assert sv.amps.shape == (8,)
    assert sv.amps[0] == 1.0
    assert np.isclose(sv.amps.sum(), 1.0)


def test_hadamard_makes_plus_state():
    sv = StateVector(1).apply(gates.H, 0)
    assert np.allclose(sv.amps, [1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_x_flips_qubit():
    sv = StateVector(1).apply(gates.X, 0)
    assert np.allclose(sv.amps, [0, 1])


def test_apply_is_chainable_and_normalized():
    sv = StateVector(2).apply(gates.H, 0).apply(gates.H, 1)
    assert np.isclose(np.linalg.norm(sv.amps), 1.0)
    assert np.allclose(sv.probabilities(), [0.25, 0.25, 0.25, 0.25])


def test_from_amplitudes():
    sv = StateVector.from_amplitudes([0, 1, 0, 0])
    assert sv.n == 2
    assert sv.amps[1] == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_statevector.py -v`
Expected: FAIL (cannot import qfs.statevector).

- [ ] **Step 3: Implement `qfs/statevector.py` (core only)**

```python
"""StateVector engine: pure quantum states as length-2^n amplitude vectors."""

import numpy as np

from .dense import embed


class StateVector:
    def __init__(self, n, rng=None):
        self.n = n
        self.amps = np.zeros(2 ** n, dtype=np.complex128)
        self.amps[0] = 1.0
        self.rng = rng if rng is not None else np.random.default_rng()

    @classmethod
    def from_amplitudes(cls, amps, rng=None):
        amps = np.asarray(amps, dtype=np.complex128)
        n = int(round(np.log2(len(amps))))
        sv = cls(n, rng)
        sv.amps = amps.copy()
        return sv

    def apply(self, U, target, controls=()):
        self.amps = embed(U, target, self.n, controls) @ self.amps
        return self

    def probabilities(self):
        return np.abs(self.amps) ** 2
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_statevector.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qfs/statevector.py tests/test_statevector.py
git commit -m "feat: StateVector core with chainable gate application"
```

---

### Task 5: Measurement, sampling, and expectation

**Files:**
- Modify: `qfs/statevector.py` (add methods to the existing class)
- Test: `tests/test_measurement.py`

**Interfaces:**
- Consumes: the `StateVector` from Task 4; `self.rng`.
- Produces, added to `StateVector`:
  - `measure(self, qubit) -> int`: sample one qubit's outcome (0/1) from the Born rule, collapse and renormalize the state, return the bit.
  - `measure_all(self) -> list[int]`: sample a full computational-basis outcome, collapse to it, return bits in qubit order (qubit 0 first).
  - `sample(self, shots) -> dict[str, int]`: non-collapsing; return counts keyed by big-endian bitstrings.
  - `expectation(self, observable) -> float`: real part of `<psi|observable|psi>` for a 2^n x 2^n Hermitian array.

- [ ] **Step 1: Write the failing tests**

`tests/test_measurement.py`:
```python
import numpy as np

from qfs import gates
from qfs.statevector import StateVector


def test_measure_basis_state_is_deterministic():
    sv = StateVector(1, rng=np.random.default_rng(0)).apply(gates.X, 0)
    assert sv.measure(0) == 1
    # state has collapsed to |1>, still normalized
    assert np.isclose(np.linalg.norm(sv.amps), 1.0)


def test_measure_plus_state_statistics():
    sv = StateVector(1, rng=np.random.default_rng(42)).apply(gates.H, 0)
    outcomes = [
        StateVector(1, rng=np.random.default_rng(s)).apply(gates.H, 0).measure(0)
        for s in range(2000)
    ]
    frac_one = sum(outcomes) / len(outcomes)
    assert 0.45 < frac_one < 0.55


def test_measure_collapses_state():
    sv = StateVector(1, rng=np.random.default_rng(1)).apply(gates.H, 0)
    b = sv.measure(0)
    # after collapse the state is a basis state consistent with b
    assert np.isclose(sv.probabilities()[b], 1.0)


def test_sample_counts_sum_to_shots():
    sv = StateVector(2, rng=np.random.default_rng(7)).apply(gates.H, 0).apply(gates.H, 1)
    counts = sv.sample(1000)
    assert sum(counts.values()) == 1000
    assert set(counts) <= {"00", "01", "10", "11"}


def test_expectation_z_on_basis_states():
    assert np.isclose(StateVector(1).expectation(gates.Z), 1.0)        # <0|Z|0> = +1
    assert np.isclose(StateVector(1).apply(gates.X, 0).expectation(gates.Z), -1.0)
    assert np.isclose(StateVector(1).apply(gates.H, 0).expectation(gates.X), 1.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_measurement.py -v`
Expected: FAIL (StateVector has no attribute 'measure').

- [ ] **Step 3: Add the methods to `qfs/statevector.py`**

Append inside the `StateVector` class:
```python
    def measure(self, qubit):
        pos = self.n - 1 - qubit
        idx = np.arange(2 ** self.n)
        is_one = ((idx >> pos) & 1).astype(bool)
        p_one = float(np.sum(np.abs(self.amps[is_one]) ** 2))
        outcome = 1 if self.rng.random() < p_one else 0
        keep = is_one if outcome == 1 else ~is_one
        self.amps = np.where(keep, self.amps, 0.0)
        self.amps = self.amps / np.linalg.norm(self.amps)
        return outcome

    def measure_all(self):
        probs = self.probabilities()
        i = int(self.rng.choice(len(probs), p=probs))
        bits = [(i >> (self.n - 1 - q)) & 1 for q in range(self.n)]
        self.amps = np.zeros_like(self.amps)
        self.amps[i] = 1.0
        return bits

    def sample(self, shots):
        probs = self.probabilities()
        draws = self.rng.choice(len(probs), size=shots, p=probs)
        counts = {}
        for d in draws:
            key = format(int(d), f"0{self.n}b")
            counts[key] = counts.get(key, 0) + 1
        return counts

    def expectation(self, observable):
        return float(np.real(np.conj(self.amps) @ observable @ self.amps))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_measurement.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qfs/statevector.py tests/test_measurement.py
git commit -m "feat: measurement, sampling, and expectation on StateVector"
```

---

### Task 6: Visualization (Bloch sphere and probability bars)

**Files:**
- Create: `qfs/viz.py`
- Test: `tests/test_viz.py`

**Interfaces:**
- Consumes: matplotlib; `qfs.gates`; `StateVector` (uses `expectation` and `probabilities`).
- Produces:
  - `bloch_vector(sv) -> np.ndarray` shape (3,): `(<X>, <Y>, <Z>)` for a 1-qubit state.
  - `plot_bloch(sv) -> matplotlib.figure.Figure`: 3-D Bloch sphere with the state arrow.
  - `plot_probabilities(sv) -> matplotlib.figure.Figure`: bar chart of basis-state probabilities.

- [ ] **Step 1: Write the failing tests**

`tests/test_viz.py`:
```python
import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import numpy as np
from matplotlib.figure import Figure

from qfs import gates
from qfs.statevector import StateVector
from qfs import viz


def test_bloch_vector_known_states():
    assert np.allclose(viz.bloch_vector(StateVector(1)), [0, 0, 1])              # |0> -> +z
    assert np.allclose(viz.bloch_vector(StateVector(1).apply(gates.X, 0)), [0, 0, -1])
    assert np.allclose(viz.bloch_vector(StateVector(1).apply(gates.H, 0)), [1, 0, 0])


def test_plot_bloch_returns_figure():
    fig = viz.plot_bloch(StateVector(1).apply(gates.H, 0))
    assert isinstance(fig, Figure)


def test_plot_probabilities_returns_figure():
    sv = StateVector(2).apply(gates.H, 0).apply(gates.X, 1)
    fig = viz.plot_probabilities(sv)
    assert isinstance(fig, Figure)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_viz.py -v`
Expected: FAIL (cannot import qfs.viz).

- [ ] **Step 3: Implement `qfs/viz.py`**

```python
"""Visualization helpers. Tests/notebooks pick the matplotlib backend."""

import numpy as np
import matplotlib.pyplot as plt

from . import gates


def bloch_vector(sv):
    if sv.n != 1:
        raise ValueError("bloch_vector requires a single-qubit state")
    return np.array(
        [sv.expectation(gates.X), sv.expectation(gates.Y), sv.expectation(gates.Z)]
    )


def plot_bloch(sv):
    v = bloch_vector(sv)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    u, w = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    ax.plot_wireframe(
        np.cos(u) * np.sin(w), np.sin(u) * np.sin(w), np.cos(w),
        color="lightgray", linewidth=0.5,
    )
    ax.quiver(0, 0, 0, v[0], v[1], v[2], color="crimson", linewidth=2)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return fig


def plot_probabilities(sv):
    probs = sv.probabilities()
    labels = [format(i, f"0{sv.n}b") for i in range(len(probs))]
    fig, ax = plt.subplots()
    ax.bar(labels, probs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("probability")
    ax.set_xlabel("basis state")
    return fig
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_viz.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qfs/viz.py tests/test_viz.py
git commit -m "feat: Bloch-sphere and probability visualization with smoke tests"
```

> **Notebook checkpoint (post 0):** after this task the post-0 notebook ("What is a qubit?") can be authored: build single-qubit states, apply gates, show `plot_bloch`, measure. Notebook prose is a separate writing pass (use the soul voice), not a TDD task.

---

### Task 7: Two-qubit gates and entanglement

**Files:**
- Test: `tests/test_entanglement.py`

**Interfaces:**
- Consumes: `StateVector.apply` with a `controls` argument (Task 4); `qfs.gates`.
- Produces: no new library code. This task proves entanglement works through the existing API and documents the Bell-state recipe (H on control, then CNOT). It is its own task because a reviewer validates the entanglement semantics independently of the core.

- [ ] **Step 1: Write the failing tests**

`tests/test_entanglement.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_entanglement.py -v`
Expected: FAIL (the test module does not exist yet; create it and confirm it runs and passes, since the underlying API already exists). If any assertion fails, the bug is in Task 4/5 code, fix there.

- [ ] **Step 3: Confirm behavior**

No new implementation expected. If `test_bell_measurements_are_correlated` fails, inspect `measure` collapse logic in `qfs/statevector.py`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_entanglement.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_entanglement.py
git commit -m "test: Bell-state entanglement (amplitudes, non-separability, correlation)"
```

> **Notebook checkpoint (post 1):** post-1 notebook ("Many qubits and entanglement") can now be authored.

---

### Task 8: Circuit abstraction

**Files:**
- Create: `qfs/circuit.py`
- Test: `tests/test_circuit.py`

**Interfaces:**
- Consumes: `qfs.gates`; `qfs.statevector.StateVector`.
- Produces: class `Circuit` with:
  - `__init__(self, n)`.
  - generic `apply(self, U, target, controls=(), label=None) -> Circuit` (records op, chainable).
  - convenience methods: `h(t)`, `x(t)`, `y(t)`, `z(t)`, `s(t)`, `t_gate(t)`, `cnot(c, t)`, `cz(c, t)`, each returning `self`.
  - `run(self, state=None, rng=None) -> StateVector`: if `state` is None, start from `StateVector(self.n, rng=rng)`; apply each recorded op; return the state.
  - `__len__`: number of ops.

- [ ] **Step 1: Write the failing tests**

`tests/test_circuit.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_circuit.py -v`
Expected: FAIL (cannot import qfs.circuit).

- [ ] **Step 3: Implement `qfs/circuit.py`**

```python
"""Circuit: a recorded sequence of gate operations runnable on a StateVector."""

from . import gates
from .statevector import StateVector


class Circuit:
    def __init__(self, n):
        self.n = n
        self.ops = []  # list of (U, target, controls, label)

    def apply(self, U, target, controls=(), label=None):
        self.ops.append((U, target, tuple(controls), label))
        return self

    def h(self, t):
        return self.apply(gates.H, t, label="H")

    def x(self, t):
        return self.apply(gates.X, t, label="X")

    def y(self, t):
        return self.apply(gates.Y, t, label="Y")

    def z(self, t):
        return self.apply(gates.Z, t, label="Z")

    def s(self, t):
        return self.apply(gates.S, t, label="S")

    def t_gate(self, t):
        return self.apply(gates.T, t, label="T")

    def cnot(self, c, t):
        return self.apply(gates.X, t, controls=(c,), label="CNOT")

    def cz(self, c, t):
        return self.apply(gates.Z, t, controls=(c,), label="CZ")

    def run(self, state=None, rng=None):
        if state is None:
            state = StateVector(self.n, rng=rng)
        for U, target, controls, _label in self.ops:
            state.apply(U, target, controls)
        return state

    def __len__(self):
        return len(self.ops)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_circuit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qfs/circuit.py tests/test_circuit.py
git commit -m "feat: Circuit abstraction with convenience gates and run()"
```

---

### Task 9: Oracles and Deutsch-Jozsa

**Files:**
- Create: `qfs/algorithms/__init__.py`
- Create: `qfs/algorithms/oracles.py`
- Create: `qfs/algorithms/deutsch_jozsa.py`
- Test: `tests/test_deutsch_jozsa.py`

**Interfaces:**
- Consumes: numpy; `qfs.gates`; `qfs.statevector.StateVector`.
- Produces:
  - `oracles.bit_oracle(f, n_in) -> np.ndarray` (2^(n_in+1) x ...): the reversible oracle |x>|y> -> |x>|y XOR f(x)>, where the ancilla is the last qubit (index n_in) and `x` is the integer formed by the input qubits (qubit 0 = MSB). `f` maps int -> 0/1.
  - `oracles.phase_oracle(marked, n) -> np.ndarray`: diagonal operator flipping the sign of basis index `marked`.
  - `deutsch_jozsa.deutsch_jozsa(f, n_in, rng=None) -> str`: returns `"constant"` or `"balanced"`.

- [ ] **Step 1: Write the failing tests**

`tests/test_deutsch_jozsa.py`:
```python
import numpy as np

from qfs.algorithms.oracles import bit_oracle, phase_oracle
from qfs.algorithms.deutsch_jozsa import deutsch_jozsa


def test_bit_oracle_is_permutation():
    op = bit_oracle(lambda x: x & 1, 2)  # f(x) = low bit of x
    assert np.allclose(op @ op, np.eye(op.shape[0]))  # self-inverse permutation


def test_phase_oracle_flips_one_amplitude():
    op = phase_oracle(3, 2)
    assert op[3, 3] == -1.0
    assert op[0, 0] == 1.0


def test_dj_constant_zero():
    assert deutsch_jozsa(lambda x: 0, 3, rng=np.random.default_rng(0)) == "constant"


def test_dj_constant_one():
    assert deutsch_jozsa(lambda x: 1, 3, rng=np.random.default_rng(0)) == "constant"


def test_dj_balanced_parity():
    # parity of the 3 input bits is balanced
    f = lambda x: (bin(x).count("1")) & 1
    assert deutsch_jozsa(f, 3, rng=np.random.default_rng(0)) == "balanced"


def test_dj_balanced_first_bit():
    f = lambda x: (x >> 2) & 1  # MSB of a 3-bit input
    assert deutsch_jozsa(f, 3, rng=np.random.default_rng(0)) == "balanced"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_deutsch_jozsa.py -v`
Expected: FAIL (cannot import qfs.algorithms).

- [ ] **Step 3: Implement the oracle and algorithm modules**

`qfs/algorithms/__init__.py`:
```python
"""Quantum algorithms built on the qfs core."""
```

`qfs/algorithms/oracles.py`:
```python
"""Oracle constructions for algorithm modules."""

import numpy as np


def bit_oracle(f, n_in):
    """Reversible oracle |x>|y> -> |x>|y XOR f(x)>; ancilla is the last qubit.

    Big-endian: the ancilla is bit 0 (least significant); the input integer x is
    the remaining high bits (input qubit 0 = MSB of x).
    """
    n = n_in + 1
    dim = 2 ** n
    op = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        x = i >> 1
        y = i & 1
        j = (x << 1) | (y ^ (int(f(x)) & 1))
        op[j, i] = 1.0
    return op


def phase_oracle(marked, n):
    """Diagonal operator that flips the sign of basis index `marked`."""
    op = np.eye(2 ** n, dtype=np.complex128)
    op[marked, marked] = -1.0
    return op
```

`qfs/algorithms/deutsch_jozsa.py`:
```python
"""Deutsch-Jozsa: decide if f is constant or balanced with one oracle call."""

from .. import gates
from ..statevector import StateVector
from .oracles import bit_oracle


def deutsch_jozsa(f, n_in, rng=None):
    n = n_in + 1
    sv = StateVector(n, rng=rng)
    sv.apply(gates.X, n - 1)              # ancilla -> |1>
    for q in range(n):                   # Hadamard everywhere
        sv.apply(gates.H, q)
    sv.amps = bit_oracle(f, n_in) @ sv.amps
    for q in range(n_in):                # Hadamard on inputs
        sv.apply(gates.H, q)
    outcomes = [sv.measure(q) for q in range(n_in)]
    return "constant" if all(b == 0 for b in outcomes) else "balanced"
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_deutsch_jozsa.py -v`
Expected: PASS. (Constant f leaves the input register in |0...0> deterministically; balanced f gives zero amplitude on |0...0>, so the result is seed-independent.)

- [ ] **Step 5: Commit**

```bash
git add qfs/algorithms/ tests/test_deutsch_jozsa.py
git commit -m "feat: oracles + Deutsch-Jozsa algorithm"
```

---

### Task 10: Bernstein-Vazirani

**Files:**
- Create: `qfs/algorithms/bernstein_vazirani.py`
- Test: `tests/test_bernstein_vazirani.py`

**Interfaces:**
- Consumes: `qfs.gates`; `qfs.statevector.StateVector`; `qfs.algorithms.oracles.bit_oracle`.
- Produces: `bernstein_vazirani(s_bits, rng=None) -> list[int]`. Input `s_bits` is the hidden string as a list of bits (index 0 = qubit 0 = MSB). Returns the recovered bits, which equal `s_bits`.

- [ ] **Step 1: Write the failing tests**

`tests/test_bernstein_vazirani.py`:
```python
import numpy as np

from qfs.algorithms.bernstein_vazirani import bernstein_vazirani


def test_bv_recovers_hidden_string():
    for s in ([1, 0, 1], [0, 0, 0], [1, 1, 1], [0, 1, 1, 0]):
        assert bernstein_vazirani(s, rng=np.random.default_rng(0)) == s


def test_bv_is_seed_independent():
    s = [1, 0, 1, 1]
    a = bernstein_vazirani(s, rng=np.random.default_rng(1))
    b = bernstein_vazirani(s, rng=np.random.default_rng(99))
    assert a == b == s
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_bernstein_vazirani.py -v`
Expected: FAIL (cannot import the module).

- [ ] **Step 3: Implement `qfs/algorithms/bernstein_vazirani.py`**

```python
"""Bernstein-Vazirani: recover a hidden string s where f(x) = s . x (mod 2)."""

from .. import gates
from ..statevector import StateVector
from .oracles import bit_oracle


def bernstein_vazirani(s_bits, rng=None):
    n_in = len(s_bits)

    def f(x):
        bits = [(x >> (n_in - 1 - i)) & 1 for i in range(n_in)]
        return sum(si * bi for si, bi in zip(s_bits, bits)) & 1

    n = n_in + 1
    sv = StateVector(n, rng=rng)
    sv.apply(gates.X, n - 1)
    for q in range(n):
        sv.apply(gates.H, q)
    sv.amps = bit_oracle(f, n_in) @ sv.amps
    for q in range(n_in):
        sv.apply(gates.H, q)
    return [sv.measure(q) for q in range(n_in)]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_bernstein_vazirani.py -v`
Expected: PASS (the input register collapses deterministically to `s`).

- [ ] **Step 5: Commit**

```bash
git add qfs/algorithms/bernstein_vazirani.py tests/test_bernstein_vazirani.py
git commit -m "feat: Bernstein-Vazirani algorithm"
```

> **Notebook checkpoint (post 2):** post-2 notebook ("Circuits and interference: Deutsch-Jozsa and Bernstein-Vazirani") can now be authored.

---

### Task 11: Grover's search

**Files:**
- Create: `qfs/algorithms/grover.py`
- Test: `tests/test_grover.py`

**Interfaces:**
- Consumes: numpy; `qfs.gates`; `qfs.statevector.StateVector`; `qfs.algorithms.oracles.phase_oracle`.
- Produces:
  - `diffusion(n) -> np.ndarray`: the Grover diffusion operator `2|s><s| - I` over the uniform superposition `|s>`.
  - `optimal_iterations(n, num_marked=1) -> int`: `floor((pi/4) * sqrt(2**n / num_marked))`.
  - `grover_search(marked, n, iterations=None, rng=None) -> StateVector`: prepare uniform superposition, apply oracle+diffusion `iterations` times (default `optimal_iterations(n)`), return the final state.

- [ ] **Step 1: Write the failing tests**

`tests/test_grover.py`:
```python
import numpy as np

from qfs.algorithms.grover import diffusion, optimal_iterations, grover_search


def test_optimal_iterations_n3():
    assert optimal_iterations(3) == 2  # floor(pi/4 * sqrt(8)) = 2


def test_diffusion_is_unitary():
    d = diffusion(3)
    assert np.allclose(d @ d.conj().T, np.eye(8))


def test_grover_amplifies_marked_state():
    sv = grover_search(marked=5, n=3, rng=np.random.default_rng(0))
    probs = sv.probabilities()
    assert int(np.argmax(probs)) == 5
    assert probs[5] > 0.9


def test_grover_each_marked_state():
    for w in range(8):
        sv = grover_search(marked=w, n=3, rng=np.random.default_rng(0))
        assert int(np.argmax(sv.probabilities())) == w


def test_grover_sampling_mostly_returns_marked():
    sv = grover_search(marked=2, n=4, rng=np.random.default_rng(3))
    counts = sv.sample(1000)
    top = max(counts, key=counts.get)
    assert top == format(2, "04b")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_grover.py -v`
Expected: FAIL (cannot import qfs.algorithms.grover).

- [ ] **Step 3: Implement `qfs/algorithms/grover.py`**

```python
"""Grover's search: amplitude amplification of a marked basis state."""

import numpy as np

from .. import gates
from ..statevector import StateVector
from .oracles import phase_oracle


def diffusion(n):
    dim = 2 ** n
    s = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    return 2 * np.outer(s, s.conj()) - np.eye(dim, dtype=np.complex128)


def optimal_iterations(n, num_marked=1):
    return int(np.floor((np.pi / 4) * np.sqrt((2 ** n) / num_marked)))


def grover_search(marked, n, iterations=None, rng=None):
    if iterations is None:
        iterations = optimal_iterations(n)
    sv = StateVector(n, rng=rng)
    for q in range(n):
        sv.apply(gates.H, q)
    oracle = phase_oracle(marked, n)
    diff = diffusion(n)
    for _ in range(iterations):
        sv.amps = oracle @ sv.amps
        sv.amps = diff @ sv.amps
    return sv
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_grover.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qfs/algorithms/grover.py tests/test_grover.py
git commit -m "feat: Grover's search via amplitude amplification"
```

> **Notebook checkpoint (post 3):** post-3 notebook ("Grover's search") can now be authored.

---

### Task 12: Qiskit differential test harness

**Files:**
- Create: `tests/test_qiskit_differential.py`

**Interfaces:**
- Consumes: `qfs.circuit.Circuit`; `qfs.statevector` (via Circuit.run); qiskit (dev-only). Skips cleanly if qiskit is absent.
- Produces: no library code. A property-style differential test: random circuits over {H, X, Y, Z, S, T, CNOT, CZ} run on both our simulator and Qiskit must agree up to global phase, after reconciling qubit-ordering conventions (we are big-endian, Qiskit is little-endian).

- [ ] **Step 1: Write the test**

`tests/test_qiskit_differential.py`:
```python
import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qfs import gates
from qfs.circuit import Circuit

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
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_qiskit_differential.py -v`
Expected: PASS for all 25 seeds. If a case fails, the printed `prog` reproduces it; the usual cause is a qubit-ordering or global-phase convention error, isolated to one gate's mapping.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests across tasks 1-12 PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_qiskit_differential.py
git commit -m "test: differential validation of the simulator against Qiskit"
```

---

## Self-Review

**1. Spec coverage (against `2026-06-18-quantum-from-scratch-design.md`):**
- From-scratch Python on NumPy substrate: Tasks 2-11 (no QC library used in `qfs/`). Covered.
- `StateVector` engine, shared-interface ops (apply/measure/expectation/probabilities): Tasks 4-5. Covered. (`partial_trace` belongs to the density-matrix arc, post 8, out of this plan's 0-3 scope: intentionally deferred.)
- Gate library (1-qubit + parametric; CNOT/CZ via controls): Tasks 2-3, 8. Covered. (SWAP and general 2-qubit unitaries are not needed through Grover and are deferred to engine B / later, per spec section 7.)
- Circuit abstraction runnable on the engine: Task 8. Covered.
- Canonical algorithms through Grover (DJ, BV, Grover; QFT/phase-estimation/Shor are posts 4-6, out of scope): Tasks 9-11. Covered for 0-3.
- Visualization (Bloch, histograms): Task 6. Covered.
- Dense/Kronecker gate application first; einsum engine deferred to post 7: honored (Task 3 is dense; no einsum here). Covered.
- Testing: per-gate unit (Task 2), normalization/invariants (Tasks 4-5), analytic checks (Tasks 7,9,10,11), Qiskit differential oracle as dev-only (Task 12). Covered.
- Global-phase + endianness reconciliation in the oracle test (spec risk section 12): Task 12 `_equal_up_to_phase` + qubit remap. Covered.

**2. Placeholder scan:** No "TBD/TODO/handle edge cases" steps; every code step contains complete code; every test step has real assertions. Notebook prose is explicitly out of scope (writing pass, flagged at checkpoints), not a hidden placeholder.

**3. Type consistency:** `embed(U, target, n, controls=())` signature is consistent across dense.py (Task 3), StateVector.apply (Task 4), and algorithm modules. `bit_oracle(f, n_in)` and `phase_oracle(marked, n)` names/signatures match between Task 9 (definition) and Tasks 10-11 (use). `StateVector` constructor `(n, rng=None)` is used consistently everywhere. `Circuit.run() -> StateVector` returns the `.amps`-bearing object used by Task 12. No drift found.

## Notes on scope

This plan delivers posts 0-3 (foundations through Grover) as a working, fully tested library plus a Qiskit differential oracle. Posts 4-10 (QFT, phase estimation, Shor, the einsum engine, density matrices, noise, QEC) are re-planned in a follow-up once this foundation is solid, per the design spec's phased-planning decision. Companion notebooks and blog prose are authored per-post after each library increment passes its tests (a writing pass in the soul voice, tracked separately from this engineering plan).
