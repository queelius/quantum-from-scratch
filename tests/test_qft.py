import numpy as np

from qfs.algorithms.qft import qft_matrix


def test_qft_is_unitary():
    for n in (1, 2, 3):
        F = qft_matrix(n)
        assert np.allclose(F @ F.conj().T, np.eye(2 ** n))


def test_qft_matches_scaled_inverse_fft():
    rng = np.random.default_rng(0)
    for n in (1, 2, 3):
        N = 2 ** n
        v = rng.normal(size=N) + 1j * rng.normal(size=N)
        assert np.allclose(qft_matrix(n) @ v, np.sqrt(N) * np.fft.ifft(v))


def test_qft_of_zero_is_uniform():
    N = 8
    F = qft_matrix(3)
    col0 = F @ np.eye(N)[:, 0]            # QFT of |000>
    assert np.allclose(col0, np.ones(N) / np.sqrt(N))
