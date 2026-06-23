import numpy as np

from qfs import gates
from qfs.dense import embed
from qfs.qec import (
    encode_bitflip,
    bitflip_syndrome,
    correct_bitflip,
    decode_bitflip,
    run_bitflip_code,
    run_phaseflip_code,
    _BITFLIP_SYNDROME,
)


def _random_logical(rng):
    a = rng.normal() + 1j * rng.normal()
    b = rng.normal() + 1j * rng.normal()
    norm = np.hypot(abs(a), abs(b))
    return a / norm, b / norm


def test_encode_produces_the_repetition_state():
    a, b = 0.6, 0.8
    psi = encode_bitflip(a, b)
    assert np.isclose(psi[0b000], a)
    assert np.isclose(psi[0b111], b)
    assert np.allclose(psi[[1, 2, 3, 4, 5, 6]], 0)


def test_bitflip_code_corrects_every_single_x_error():
    rng = np.random.default_rng(0)
    for _ in range(8):
        a, b = _random_logical(rng)
        for e in (None, 0, 1, 2):
            ra, rb = run_bitflip_code(a, b, error_qubit=e)
            assert np.allclose([ra, rb], [a, b])


def test_phaseflip_code_corrects_every_single_z_error():
    rng = np.random.default_rng(1)
    for _ in range(8):
        a, b = _random_logical(rng)
        for e in (None, 0, 1, 2):
            ra, rb = run_phaseflip_code(a, b, error_qubit=e)
            assert np.allclose([ra, rb], [a, b])


def test_syndrome_identifies_the_flipped_qubit():
    for e in (None, 0, 1, 2):
        psi = encode_bitflip(0.6, 0.8)
        if e is not None:
            psi = embed(gates.X, e, 3) @ psi
        assert _BITFLIP_SYNDROME[bitflip_syndrome(psi)] == e


def test_code_fails_on_two_errors():
    psi = encode_bitflip(1, 0)
    psi = embed(gates.X, 0, 3) @ embed(gates.X, 1, 3) @ psi
    out = decode_bitflip(correct_bitflip(psi))
    assert not np.allclose([out[0], out[0b100]], [1, 0])


def test_phaseflip_code_on_plus_logical():
    a = b = 1 / np.sqrt(2)
    for e in (0, 1, 2):
        ra, rb = run_phaseflip_code(a, b, error_qubit=e)
        assert np.allclose([ra, rb], [a, b])
