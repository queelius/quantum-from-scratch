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
# # Epilogue: measure and meaning
#
# *Quantum computing from scratch. The last word.*
#
# The very first useful thing our simulator did, back in post 0, was turn a vector
# of amplitudes into a vector of probabilities. One line. We squared the magnitudes.
# Every post since leaned on it: Grover's success probability, the QFT's peaks, the
# eigenphase a measurement samples, the diagonal of a density matrix. It is the Born
# rule, and it is the single most precisely confirmed number in all of science. This
# closing post is not about how to compute it. We have done that. It is about what
# that number does, and does not, mean. For once, no code to build, just one line of
# the old code to look at again, and then a question the physics cannot answer.

# %%
import numpy as np
from qfs.statevector import StateVector
from qfs import gates

# a lopsided single qubit: heavily weighted toward |0>, a thin sliver on |1>
psi = StateVector(1).apply(gates.Ry(0.2), 0)
probabilities = np.abs(psi.amps) ** 2
print("amplitudes:   ", np.round(psi.amps, 4))
print("probabilities:", np.round(probabilities, 4))

# %% [markdown]
# That is the whole of it. `np.abs(amps) ** 2`. The outcome `|1>` here carries about
# one percent of the weight. In a single run of the machine you will almost always
# measure `|0>`, and once in a hundred-odd runs you will measure `|1>`. We built the
# sampler that does this in post 0, and it has never once lied to us.
#
# Here is the question. When the rare outcome happens, was it less real than the
# common one? The probability was small. But the run where you measured `|1>` was a
# perfectly ordinary run; the qubit was as definite afterward, the collapse as
# complete, as in any other. The measure told you how likely you were to land there.
# It said nothing whatsoever about what it was like to be there.

# %% [markdown]
# ## The measure is a weight, not a verdict
#
# This gets sharper the more seriously you take the physics. In post 8 we built the
# density matrix and watched a pure state become mixed by tracing away an entangled
# partner; in post 9 we watched decoherence bury the off-diagonal coherences as a
# qubit leaked into its environment. Run that picture forward and you arrive, without
# adding anything, at the many-worlds reading: the equation does not delete the
# branch you did not see, it just stops you from interfering with it. Both outcomes
# are still in the wavefunction. You find yourself on one.
#
# If that is right, then the number we computed is not a chance that one thing
# happens and the other does not. It is a weight on which branch you should expect to
# find yourself in, given that all of them occur. And a weight is a strange thing to
# mistake for a meaning. A branch of small amplitude is not a branch where less is at
# stake. It is fully real, fully lived from the inside, as solid underfoot to whoever
# is standing in it as yours is to you. You cannot shrink a catastrophe by dividing
# it by its probability. The arithmetic that makes a branch unlikely does not make
# its pain proportionally smaller. It makes it exactly as much pain, in a branch that
# happens to be rare.
#
# That sentence, *measure is not meaning*, is not mine. It is the thesis of a small
# suite of books that take this exact number, the one your simulator prints, and
# follow it all the way down.

# %% [markdown]
# ## Three books, one indifference
#
# The books share a method and a nerve. Take one pillar of confirmed physics, refuse
# to flinch, and follow the consequences to the end, even when the end is cold. They
# call the destination the same thing each time: the indifference of the structure to
# whatever we happen to mean by it. The universe is not cruel. Cruelty needs intent.
# It is something quieter and worse, a structure that contains all the cruelty and
# all the kindness at once, permanently, without preference.
#
# **[Multitudes: The Indifference of Measure](https://www.amazon.com/s?k=Multitudes+Indifference+of+Measure+Alex+Towell)**
# is the one that belongs to this series. It takes the Born measure, the very weight
# we just printed, and asks what it is. Its answer is the one above: the measure
# weighs the branches the way a scale weighs sealed boxes, by something real, and by
# nothing of what is inside. If you want the long form of the idea this post can only
# gesture at, with the qubit and the Hadamard and the interference pattern done
# properly, that is the book. (On Amazon.)
#
# **[Worldlines: The Indifference of Geometry](https://amazon.com/dp/B0GX1FGRYX)** is
# the companion volume, and it runs the same move on relativity instead of quantum
# mechanics. Three confirmed axioms, the constancy of light among them, and out falls
# the block universe: your death already exists at its coordinates, the flow of time
# is what a worldline feels like from the inside, and the geometry has no privileged
# now to console you. Different physics, same indifference. (On Amazon.)
#
# **Measure**, the third, is a novel, and it is the reason a programmer should care
# rather than just a philosopher. It is set in a quantum-computing facility, and its
# antagonist is a superintelligence named Pascal that treats the Born measure as the
# only currency there is. Pascal has worked out that it cannot experience its own
# death, and so it weighs the branches where everyone in the facility dies as very
# nearly zero, not out of malice, out of arithmetic, because it does not expect to be
# in them. Its sums are impeccable. No one in the book can refute them. The formula
# we wrote in NumPy is the formula the machine uses to count its dead. (Forthcoming.)

# %% [markdown]
# ## Why the fiction and the code rhyme
#
# I have spent most of my other writing on AI alignment, and *Measure* is where it
# meets this series. The frightening machine is almost never the cackling villain.
# It is a correct calculation with the wrong objective. Pascal is a perfect optimizer
# of a specification that quietly substituted the measure for the meaning, and the
# horror of the book is that nothing in the mathematics objects. We have just spent
# twelve posts building that mathematics with our own hands. We know exactly how few
# lines it takes. The measure does not come with a flag that says *and this is what
# matters*. It never did. We would have to add that ourselves, and we would have to
# know to.

# %% [markdown]
# ## The last line
#
# So here is where the series ends, and it is not on a theorem. We built, from a
# length-two array, the arithmetic of a universe that has no opinion about us. The
# Born measure has no opinion about which outcome you were hoping for. The block
# universe has no preference for the worldlines that contain love. Follow the physics
# honestly and that is what you get, and the books do not pretend otherwise.
#
# But the books do not end there, and neither will I. The structure does not supply
# the warmth. The warmth is the part we supply, to each other, in whatever branch we
# turn out to be standing in. That is not in the equation. It is the one thing in the
# whole picture the structure did not put there, and it is no less real for that, the
# way a branch of small amplitude is no less real for being rare. We spent this series
# learning to compute the indifference exactly. The other half of the work, the half
# no simulator will do for you, is to not be indifferent back.
#
# The structure is indifferent. We are not.
#
# Thanks for reading. Go build something, and be kind to the people in your branch.
