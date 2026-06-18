# Circuits and interference

*Quantum computing from scratch, post 2.*

So far the qubits have mostly sat still. This post puts them to work. The two
algorithms here, Deutsch-Jozsa and Bernstein-Vazirani, are the first place where
a quantum circuit does something a classical one cannot do as cheaply, and they
both run on the same mechanism: interference. Amplitudes are complex numbers, and
complex numbers can cancel. Arrange a circuit so that the amplitudes of the wrong
answers cancel and the amplitudes of the right answer add, and one run of the
circuit hands you a global fact about a function that classically would take many
queries to pin down.


```python
import numpy as np

from qfs import gates
from qfs.statevector import StateVector
from qfs.algorithms.oracles import bit_oracle
from qfs.algorithms.deutsch_jozsa import deutsch_jozsa
from qfs.algorithms.bernstein_vazirani import bernstein_vazirani
```

## The oracle: a function written into a circuit

Both algorithms are handed a function $f$ on bit strings as a black box, called
an oracle. The reversible way to apply a classical function on a quantum computer
is to put the answer into an extra qubit:

$$|x\rangle |y\rangle \;\longmapsto\; |x\rangle\, |y \oplus f(x)\rangle.$$

That XOR keeps the map reversible (do it twice and you are back where you
started), which it has to be, because every quantum gate is unitary. `qfs` builds
this oracle matrix from any Python function `f`.


```python
# f(x) = low bit of x, on a 2-bit input plus 1 ancilla: the oracle is a permutation
U = bit_oracle(lambda x: x & 1, n_in=2)
np.allclose(U @ U, np.eye(U.shape[0]))  # applying it twice is the identity
```




    True



## Deutsch-Jozsa: one query for a global property

Promise: $f$ is either *constant* (the same value on every input) or *balanced*
(0 on exactly half the inputs, 1 on the other half). Decide which. Classically,
in the worst case, you have to evaluate $f$ on just over half the inputs before
you can be sure. The quantum circuit decides it in a single query.

The trick is phase kickback. Put the answer qubit in the $|-\rangle$ state. Then
writing $f(x)$ into it does not flip a bit you read later, it multiplies the whole
branch by $(-1)^{f(x)}$. The value of $f$ has moved into the phase of $|x\rangle$.
A layer of Hadamards on the input register then makes those phases interfere.

Here is the punchline first, then the mechanism. `deutsch_jozsa` runs the full
circuit and reports the answer.


```python
problems = {
    "constant 0": lambda x: 0,
    "constant 1": lambda x: 1,
    "balanced (parity)": lambda x: bin(x).count("1") & 1,
    "balanced (first bit)": lambda x: (x >> 2) & 1,
}
for name, f in problems.items():
    print(f"{name:24s} -> {deutsch_jozsa(f, n_in=3)}")
```

    constant 0               -> constant
    constant 1               -> constant
    balanced (parity)        -> balanced
    balanced (first bit)     -> balanced


Why it works: after the kickback and the final Hadamards, the amplitude of the
all-zeros input is the average of $(-1)^{f(x)}$ over every input $x$. If $f$ is
constant that average is $\pm 1$, so all the probability piles onto $|00\ldots0\rangle$.
If $f$ is balanced the $+1$ and $-1$ terms cancel exactly, the all-zeros amplitude
is zero, and you are guaranteed to measure something else. We can read that
cancellation straight off the state. Here is the probability of measuring the
input register as all-zeros, computed directly.


```python
def prob_all_zero_inputs(f, n_in):
    n = n_in + 1
    sv = StateVector(n).apply(gates.X, n - 1)        # ancilla -> |1>
    for q in range(n):
        sv.apply(gates.H, q)                          # Hadamard everywhere -> ancilla in |->
    sv.amps = bit_oracle(f, n_in) @ sv.amps           # phase kickback
    for q in range(n_in):
        sv.apply(gates.H, q)                          # interfere the input register
    probs = sv.probabilities()
    return float(probs[0] + probs[1])                 # input bits all zero (ancilla is the last qubit)

for name, f in problems.items():
    print(f"{name:24s} P(inputs all zero) = {prob_all_zero_inputs(f, 3):.3f}")
```

    constant 0               P(inputs all zero) = 1.000
    constant 1               P(inputs all zero) = 1.000
    balanced (parity)        P(inputs all zero) = 0.000
    balanced (first bit)     P(inputs all zero) = 0.000


Constant functions land exactly on probability 1, balanced functions exactly on 0.
That is interference doing the work: not approximately, exactly, because the wrong
branches cancel term for term.

## Bernstein-Vazirani: read a hidden string in one query

Same machinery, sharper payoff. Now $f(x) = s \cdot x \bmod 2$ for some hidden
bit string $s$, the parity of the bits of $x$ that $s$ selects. Classically,
learning all $n$ bits of $s$ takes $n$ queries, one per bit (query $x = 100\ldots$,
then $010\ldots$, and so on). The quantum circuit, the exact same kickback plus
Hadamards, returns all of $s$ from a single query. The Hadamard layer maps the
phase pattern $(-1)^{s\cdot x}$ straight back to the basis state $|s\rangle$.


```python
for s in ([1, 0, 1], [1, 1, 0, 1], [0, 0, 0], [1, 1, 1, 1]):
    recovered = bernstein_vazirani(s, rng=np.random.default_rng(0))
    print(f"hidden s = {s}  ->  recovered {recovered}  ({'match' if recovered == s else 'MISMATCH'})")
```

    hidden s = [1, 0, 1]  ->  recovered [1, 0, 1]  (match)
    hidden s = [1, 1, 0, 1]  ->  recovered [1, 1, 0, 1]  (match)
    hidden s = [0, 0, 0]  ->  recovered [0, 0, 0]  (match)
    hidden s = [1, 1, 1, 1]  ->  recovered [1, 1, 1, 1]  (match)


It is deterministic: the input register collapses onto $|s\rangle$ with
probability 1, so the seed does not matter and the answer is exact.

## Where this leaves us

Neither algorithm is doing search or trying many inputs at once, which is the
usual bad metaphor. The register holds one vector. What the circuit does is encode
a property of $f$ into the relative phases of that vector, and then use Hadamard
interference to concentrate the answer onto a basis state you can measure. The
wrong answers are not skipped, they cancel.

Both of these solve artificial promise problems. The next post is the first
algorithm with a real job: Grover's search, which finds a marked item among $N$
in about $\sqrt{N}$ steps, and where interference shows up as a slow, geometric
rotation you can watch climb toward the answer.
