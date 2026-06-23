# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The engine, in full
#
# *Quantum computing from scratch, an appendix.*
#
# Across the series I kept saying the simulator was small: a qubit is an array, a
# gate is a matrix, the whole thing is a few hundred lines. This appendix makes good
# on that the literate way. We build the statevector engine, the part every algorithm
# in the series ran on, in one sitting, with nothing hidden. Every line below is the
# actual `qfs` library code, presented in the order you would write it. By the end we
# will have a working quantum computer simulator and we will use it to make a Bell
# state, having written every operation it rests on ourselves.
#
# There are exactly three pieces: the gates (data), the embedding (how a small gate
# acts on a big register), and the state (the array, plus apply and measure).

# %%
import numpy as np

# %% [markdown]
# ## 1. Gates are constant matrices
#
# A single-qubit gate is a 2x2 complex matrix, and the standard ones are just
# numbers you can write down. This is the entire gate library; there is no cleverness
# to it, which is the point. The Pauli matrices X, Y, Z, the Hadamard H that builds
# superposition, and the phase gates S and T.

# %%
I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)

# %% [markdown]
# The continuous rotations are the only gates that need to be functions, because they
# take an angle. `Ry` rotates a qubit's Bloch vector around the y axis; it is the one
# we use below to make an arbitrary state.

# %%
def Ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def phase(lam):
    return np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=np.complex128)

# %% [markdown]
# The only property a gate must have is that it is unitary, $U^\dagger U = I$, which
# is exactly the condition that keeps a normalized state normalized. We can check it.

# %%
for name, U in [("X", X), ("H", H), ("S", S), ("Ry(0.7)", Ry(0.7))]:
    print(f"{name:8s} unitary:", np.allclose(U.conj().T @ U, I))

# %% [markdown]
# ## 2. Embedding a small gate into a big register
#
# Here is the first real idea. A gate is 2x2, but an $n$-qubit state lives in
# $2^n$ dimensions. To apply a single-qubit gate to qubit `target` of an $n$-qubit
# register, we need the full $2^n \times 2^n$ operator that acts as the gate on that
# one qubit and as the identity on every other. The textbook way to write it is a
# Kronecker product: identity on each untouched wire, the gate on the target wire.

# %%
def kron_embed(U, target, n):
    """Embed single-qubit U on `target` via Kronecker products (big-endian)."""
    op = np.array([[1.0 + 0j]])
    for q in range(n):
        op = np.kron(op, U if q == target else np.eye(2, dtype=np.complex128))
    return op

# %% [markdown]
# That is correct and clear, and it is how you should first understand gate
# application. But it cannot do a controlled gate (where the gate fires only when some
# other qubit is 1), and those are half of every interesting circuit. So the library
# uses a slightly lower-level version that builds the operator column by column. For
# each basis state (each column), it checks whether all the control qubits are 1; if
# so it applies the gate to the target bit, otherwise it leaves the column alone. With
# no controls this does exactly what `kron_embed` does; with controls it is a
# controlled gate. This one function is the engine's true workhorse.

# %%
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

# %% [markdown]
# The two agree when there are no controls, which is the sanity check that the
# clever version did not break anything.

# %%
print("embed == kron_embed (no controls):", np.allclose(embed(H, 1, 3), kron_embed(H, 1, 3)))

# %% [markdown]
# And the controlled case gives a CNOT: control qubit 0, target qubit 1, it flips
# the target exactly when the control is set. On the two-qubit basis that is the
# permutation that sends $|10\rangle$ to $|11\rangle$ and back.

# %%
cnot = embed(X, target=1, n=2, controls=(0,))
print("CNOT is the expected 4x4 permutation:")
print(cnot.real.astype(int))

# %% [markdown]
# ## 3. The state, and the two things you do to it
#
# Now the state itself. An $n$-qubit pure state is a length-$2^n$ array of complex
# amplitudes, starting in $|00\ldots0\rangle$. There are exactly two operations: apply
# a gate (multiply by the embedded operator), and measure (sample by the Born rule and
# collapse). That is the whole class.

# %%
class StateVector:
    def __init__(self, n, rng=None):
        self.n = n
        self.amps = np.zeros(2 ** n, dtype=np.complex128)
        self.amps[0] = 1.0
        self.rng = rng if rng is not None else np.random.default_rng()

    def apply(self, U, target, controls=()):
        self.amps = embed(U, target, self.n, controls) @ self.amps
        return self

    def probabilities(self):
        return np.abs(self.amps) ** 2

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

# %% [markdown]
# Look at how little is there. `apply` is one matrix multiply. `probabilities` is the
# squared magnitudes, the Born rule in one line. `measure` does the only subtle thing
# in the whole engine: it computes the probability that the chosen qubit is 1 by
# summing the squared amplitudes of the basis states where that bit is set, flips a
# weighted coin, then zeroes out the amplitudes inconsistent with the outcome and
# renormalizes. Collapse is just deleting the part of the array you did not see.

# %% [markdown]
# ## 4. A quantum computer, used
#
# That is the entire engine: three matrices, one embedding, one small class. Everything
# the series did, Grover, the Fourier transform, Shor, ran on top of exactly this.
# Let us prove it works by building a Bell state, the canonical piece of entanglement,
# from the parts we just wrote: put qubit 0 in superposition with a Hadamard, then
# entangle qubit 1 with a CNOT.

# %%
bell = StateVector(2).apply(H, 0).apply(X, 1, controls=(0,))
print("Bell state amplitudes:", np.round(bell.amps, 3))
print("equal weight on |00> and |11>, nothing on |01> or |10>")

# %% [markdown]
# And the signature of entanglement, that measuring one qubit instantly fixes the
# other. We built `measure` ourselves a moment ago; here it is enforcing the
# correlation. Across many runs the two qubits always agree.

# %%
rng = np.random.default_rng(0)
agree = 0
for _ in range(1000):
    sv = StateVector(2, rng=rng).apply(H, 0).apply(X, 1, controls=(0,))
    if sv.measure(0) == sv.measure(1):
        agree += 1
print(f"the two qubits agreed in {agree}/1000 measurements")

# %% [markdown]
# ## Where this leaves us
#
# Forty lines, give or take, and we have a quantum computer simulator that entangles
# qubits and measures them correctly. The rest of the `qfs` library is more of the
# same shape: the algorithms are circuits built from these gates, the Fourier transform
# is a particular pattern of controlled phases, the density matrix swaps the length
# $2^n$ vector for a $2^n \times 2^n$ matrix and the noise channels act on that. But the
# beating heart is what you just read in full. There was never anything behind the
# curtain, because we wrote the curtain. That was the whole idea of building it from
# scratch: not to have a simulator, but to have no mysteries left in it.
