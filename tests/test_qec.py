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
    # two different (a, b) pairs so a constant-returning encoder cannot pass
    for a, b in ((0.6, 0.8), (0.5 + 0.5j, 0.5 - 0.5j)):
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
    # random complex logical states: the syndrome is amplitude-independent, so a
    # mislabeled table cannot pass by amplitude coincidence
    rng = np.random.default_rng(7)
    for _ in range(5):
        a, b = _random_logical(rng)
        for e in (None, 0, 1, 2):
            psi = encode_bitflip(a, b)
            if e is not None:
                psi = embed(gates.X, e, 3) @ psi
            assert _BITFLIP_SYNDROME[bitflip_syndrome(psi)] == e


def test_two_errors_cause_a_logical_flip():
    # the distance-3 code corrects ONE error; two flips are miscorrected into a
    # logical bit-flip. Encode |0>_L, apply two X errors, "correct", decode -> |1>_L
    # (recovers (0, 1), not (1, 0)). All three error pairs fail this same way.
    for pair in ((0, 1), (0, 2), (1, 2)):
        psi = encode_bitflip(1, 0)
        for q in pair:
            psi = embed(gates.X, q, 3) @ psi
        out = decode_bitflip(correct_bitflip(psi))
        assert np.allclose([out[0], out[0b100]], [0, 1])


def test_bitflip_code_does_not_correct_phase_errors():
    # The bit-flip code only catches X errors; a Z error commutes with the Z-parity
    # syndrome and is left uncorrected. This is exactly why the phase-flip code
    # conjugates by Hadamards: that H layer is load-bearing, not decoration.
    a = b = 1 / np.sqrt(2)
    psi = encode_bitflip(a, b)
    psi = embed(gates.Z, 0, 3) @ psi
    out = decode_bitflip(correct_bitflip(psi))
    assert not np.allclose([out[0], out[0b100]], [a, b])


def test_phaseflip_code_on_plus_logical():
    a = b = 1 / np.sqrt(2)
    for e in (0, 1, 2):
        ra, rb = run_phaseflip_code(a, b, error_qubit=e)
        assert np.allclose([ra, rb], [a, b])
