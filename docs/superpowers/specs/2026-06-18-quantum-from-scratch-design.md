# Quantum Computing From Scratch: Design Spec

- **Date:** 2026-06-18
- **Author:** Alexander Towell
- **Status:** Approved (brainstorming complete); pending implementation plan
- **Repo home:** `metafunctor-series/quantum-from-scratch/`

## 1. Goal & motivation

**Learning-first.** The primary success criterion is that *I genuinely understand
quantum computing from the linear algebra up*. The "spooky" parts (entanglement,
interference, measurement collapse, decoherence) become ordinary operations on arrays
that I have written myself.

Blogging is a valued **byproduct**: the build doubles as a metafunctor.com series
("learning in public"), matching the `wire-formats` / `sicp` / `scratchnn` pattern.

Contributing to Qiskit is **not** a goal of this project, but it is an explicit,
deferred **Phase 2** that this project is deliberately the on-ramp for (see Section 10).

## 2. Non-goals

- Not a production or competitive simulator (Qiskit, Cirq, QuTiP already exist).
- Not Qiskit contribution (that is Phase 2, a separate spec).
- Not maximal qubit count or performance. Roughly 10 to 12 qubits is enough for every
  algorithm in scope; the density-matrix arc needs far fewer.

## 3. Approach

**From-scratch Python**, where "from scratch" means the **quantum semantics**:
tensor products, gate application, measurement and collapse, partial trace, Kraus maps.

**NumPy is the numerical substrate** (complex arrays, `matmul`, `eigh`). Reimplementing
complex linear algebra is explicitly out of scope: it is not the lesson, and it would
obscure the lesson. (This is a deliberate departure from `scratchnn`'s "no NumPy"
purism, which served a different lesson.)

## 4. Scope / summit

Two arcs:

1. **Pure-state canonical algorithms:** statevector, gates, entanglement,
   Deutsch-Jozsa / Bernstein-Vazirani, Grover, QFT, phase estimation, and **Shor**
   (capstone of arc 1).
2. **Realism:** density matrices, noise channels and decoherence, and a **taste of
   quantum error correction** (summit).

## 5. Artifact shape

Library core (clean, tested package) **plus** companion Jupyter notebooks
(jupytext-paired), one per post, exported to metafunctor.com.

```
quantum-from-scratch/
  qfs/                 # the durable library (clean, tested)   [name TBD in plan]
  notebooks/           # one companion notebook per post (jupytext-paired .py + .ipynb)
  tests/               # pytest suite
  posts/               # exported prose -> metafunctor.com
  docs/                # specs, plans
```

## 6. Architecture: two engines, one interface

The "+realism" summit forces exactly one real structural decision, and it is clean:
**two state engines behind a shared interface.**

- **`StateVector`** (pure states): a length-2^n complex array. Home of all canonical algorithms.
- **`DensityMatrix`** (mixed states): a 2^n by 2^n complex array. Home of noise, decoherence, QEC.
- **Shared interface:** `apply(gate, targets)`, `measure(qubits)`, `expectation(observable)`,
  `partial_trace(keep)`, `probabilities()`.
- **`Gate` library:** 1-qubit (I, X, Y, Z, H, S, T, Rx, Ry, Rz, phase),
  2-qubit (CNOT, CZ, SWAP, controlled-U), parametric.
- **`Circuit`:** a printable, runnable sequence of ops that executes on *either* engine unchanged.
- **`Channel`** (density-matrix only): noise via Kraus operators, namely depolarizing,
  amplitude damping, phase damping, bit-flip and phase-flip.
- **`viz`:** Bloch sphere (1-qubit), amplitude bar charts, measurement histograms,
  density-matrix heatmaps (watch off-diagonals decay, i.e. decoherence on screen).

**Key invariant / conceptual bridge:** a *noiseless* `Circuit` must produce identical
observable results on both engines. This is simultaneously a test and the central lesson.
A mixed state is a pure state "with uncertainty allowed," and decoherence is just
entanglement with an environment you have stopped tracking (the partial trace).

## 7. Gate-application strategy

- **(A) Dense / Kronecker:** build the full 2^n by 2^n operator via Kronecker products
  (`I (x) U (x) I ...`) and matrix-multiply. Literally the textbook math; maximally
  transparent. Cost O(4^n), which is fine to about 10 qubits.
- **(B) In-place tensor contraction:** reshape the statevector to a rank-n tensor and
  apply the gate only to its axes (`np.einsum`). Cost O(2^n); this is how real simulators
  dodge the blowup.

**Decision: start with (A); introduce (B) as its own mid-series post** ("why the obvious
simulator dies at 12 qubits, and how to fix it"). Build the clear thing, hit the
exponential wall, then optimize.

## 8. Build sequence (module increment = post = notebook)

| # | Post / notebook | Library increment |
|---|---|---|
| 0 | What *is* a qubit? Bloch sphere, 1-qubit gates, measurement | `StateVector` (n=1), core gates, Bloch viz |
| 1 | Many qubits, tensor products, **entanglement** (Bell states) | n-qubit register, Kronecker apply, CNOT |
| 2 | Circuits & **interference**: Deutsch-Jozsa, Bernstein-Vazirani | `Circuit` abstraction |
| 3 | **Grover's search** (amplitude amplification, the geometry) | oracles, diffusion operator |
| 4 | The **QFT** | controlled-phase ladder |
| 5 | **Phase estimation** | QPE routine |
| 6 | **Shor's algorithm** (capstone of pure-state arc) | period-finding + classical post-processing |
| 7 | **Scaling the simulator**: the einsum approach | engine (B) |
| 8 | **Mixed states & density matrices**: the two faces of rho | `DensityMatrix` engine |
| 9 | **Noise & decoherence**: Kraus channels, watching coherence die | `Channel` library |
| 10 | **A taste of error correction** (3-qubit / Shor code): QEC as fighting decoherence | encode / syndrome / correct |

**Planning scope:** the full arc above is the **roadmap**. The **first implementation plan
covers posts 0 through 3** (foundations through Grover); the remainder is re-planned once
the foundation feels solid. (Phased to respect YAGNI and to keep each plan executable.)

## 9. Testing strategy

- Per-gate unit tests (unitarity, known matrices).
- Property tests: statevector stays normalized; rho stays trace-1, Hermitian,
  positive-semidefinite.
- Analytic checks: Bell-state probabilities, Grover success probability, QFT vs classical
  DFT, Deutsch-Jozsa determinism.
- **Differential test against Qiskit:** run batteries of random circuits through both the
  from-scratch simulator and Qiskit; assert statevector agreement up to global phase.
  Qiskit is the **oracle**, a **test-only dev dependency**, never the implementation.
- Density-matrix engine cross-validated against the statevector engine on noiseless
  circuits; noise channels validated against analytic decay rates.

## 10. Where Qiskit fits (closing the loop on the original question)

1. **Differential-testing oracle** (dev dependency).
2. **Optional capstone posts:** "here is the same algorithm in Qiskit / on real IBM
   hardware," written *after* the from-scratch understanding is solid.
3. **On-ramp to Phase 2:** the close reading of Qiskit's API as an oracle, plus full
   domain fluency, is exactly what would make a future Qiskit *contribution* tractable.

## 11. Open questions (resolve in the plan)

- Package name (`qfs`? `quantum`? something else).
- Notebook tooling: jupytext pairing plus nbconvert or quarto for post export.
- Python version and dependency pins (numpy, matplotlib, pytest, `qiskit` as dev-only).
- Bloch-sphere rendering choice (matplotlib 3D vs. a small dependency such as
  `qutip` or `kaleido`).

## 12. Risks

- **Scope creep**, mitigated by phased planning (plan only 0 through 3 first).
- **Density-matrix memory** is O(4^n), tighter than statevector's O(2^n). Fine, because
  the QEC toy codes use about 5 qubits or fewer.
- **Global-phase or endianness mismatches vs Qiskit** in the differential tests are
  anticipated: the oracle comparison must normalize global phase and fix a qubit-ordering
  convention.
