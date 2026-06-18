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
