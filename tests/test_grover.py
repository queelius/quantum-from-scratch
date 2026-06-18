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
