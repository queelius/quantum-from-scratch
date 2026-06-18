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
