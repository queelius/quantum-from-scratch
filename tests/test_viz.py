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
