"""Validate chaos_tools against systems with known answers.

These aren't just shape/smoke tests -- they check that the numbers land in
the right ballpark against textbook values (Lorenz: LLE ~0.906, D2 ~2.05,
state dimension 3) and that the tools correctly tell chaos apart from a
periodic signal and from noise, which is the actual point of the package.
Run with: pytest tests/
"""

from __future__ import annotations

import numpy as np
import pytest

from chaos_tools import dimension, embedding, lyapunov, recurrence


def _lorenz(n: int, dt: float = 0.01, x0=(1.0, 1.0, 1.0)) -> np.ndarray:
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    traj = np.empty((n, 3))
    traj[0] = x0
    for t in range(1, n):
        x, y, z = traj[t - 1]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        traj[t] = traj[t - 1] + dt * np.array([dx, dy, dz])
    return traj


@pytest.fixture(scope="module")
def lorenz_x():
    traj = _lorenz(9000, dt=0.01)
    return traj[1000:, 0], 0.01  # discard transient; return x(t) and dt


@pytest.fixture(scope="module")
def sine_x():
    rng = np.random.default_rng(0)
    t = np.arange(4000) * 0.05
    return np.sin(t) + 0.01 * rng.standard_normal(len(t)), 0.05


@pytest.fixture(scope="module")
def noise_x():
    rng = np.random.default_rng(0)
    return rng.standard_normal(4000), 1.0


# --- embedding -------------------------------------------------------------


def test_delay_embed_shape_and_values():
    x = np.arange(10, dtype=float)
    emb = embedding.delay_embed(x, dim=3, delay=2)
    assert emb.shape == (6, 3)
    np.testing.assert_array_equal(emb[0], [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(emb[-1], [5.0, 7.0, 9.0])


def test_delay_embed_rejects_too_short_series():
    with pytest.raises(ValueError):
        embedding.delay_embed(np.arange(5, dtype=float), dim=3, delay=3)


def test_mutual_information_self_vs_independent():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    y = rng.standard_normal(2000)
    mi_self = embedding.mutual_information(x, x, bins=16)
    mi_independent = embedding.mutual_information(x, y, bins=16)
    assert mi_self > mi_independent
    assert mi_independent < 0.15  # near zero for genuinely independent data


def test_estimate_dimension_recovers_lorenz_state_dimension(lorenz_x):
    x, _ = lorenz_x
    tau = embedding.estimate_delay(x, max_lag=80)
    dim, fractions = embedding.estimate_dimension(x, tau, max_dim=6)
    assert dim == 3
    assert fractions[2] < 0.05  # false-neighbor fraction should have collapsed by dim 3


def test_estimate_dimension_does_not_collapse_for_noise(noise_x):
    x, _ = noise_x
    tau = embedding.estimate_delay(x, max_lag=60)
    _, fractions = embedding.estimate_dimension(x, tau, max_dim=8)
    # noise has no low-dimensional structure to find
    assert np.all(fractions > 0.1)


# --- lyapunov ----------------------------------------------------------------


def test_lyapunov_exponent_lorenz_matches_textbook_value(lorenz_x):
    x, dt = lorenz_x
    tau = embedding.estimate_delay(x, max_lag=80)
    dim, _ = embedding.estimate_dimension(x, tau, max_dim=6)
    lle, times, log_div, fit_range = lyapunov.largest_lyapunov_exponent(x, dim, tau, dt=dt)
    assert len(times) == len(log_div)
    assert fit_range[0] < fit_range[1]
    # textbook value ~0.906; a coarse Euler integration and finite samples
    # earn some slack, but it should be unambiguously in the right regime
    assert 0.6 < lle < 1.3


def test_lyapunov_exponent_periodic_signal_is_near_zero(sine_x):
    x, dt = sine_x
    tau = embedding.estimate_delay(x, max_lag=60)
    dim, _ = embedding.estimate_dimension(x, tau, max_dim=6)
    lle, _, _, _ = lyapunov.largest_lyapunov_exponent(x, dim, tau, dt=dt)
    assert abs(lle) < 0.15


# --- dimension -----------------------------------------------------------------


def test_correlation_dimension_lorenz_matches_textbook_value(lorenz_x):
    x, _ = lorenz_x
    tau = embedding.estimate_delay(x, max_lag=80)
    dim, _ = embedding.estimate_dimension(x, tau, max_dim=6)
    d2, radii, corr_sum, fit_range = dimension.correlation_dimension(x, dim, tau)
    assert len(radii) == len(corr_sum)
    assert fit_range[0] < fit_range[1]
    assert 1.7 < d2 < 2.4  # textbook value ~2.05


def test_correlation_dimension_periodic_signal_is_near_one(sine_x):
    x, dt = sine_x
    tau = embedding.estimate_delay(x, max_lag=60)
    dim, _ = embedding.estimate_dimension(x, tau, max_dim=6)
    d2, _, _, _ = dimension.correlation_dimension(x, dim, tau)
    assert 0.7 < d2 < 1.4


def test_correlation_dimension_grows_with_embedding_dim_for_noise(noise_x):
    x, _ = noise_x
    tau = embedding.estimate_delay(x, max_lag=60)
    d2_low, *_ = dimension.correlation_dimension(x, dim=2, delay=tau)
    d2_high, *_ = dimension.correlation_dimension(x, dim=6, delay=tau)
    # noise fills whatever space you embed it in; a real attractor's
    # dimension estimate would plateau instead of climbing like this
    assert d2_high > d2_low + 1.5


# --- recurrence ------------------------------------------------------------


def test_recurrence_matrix_hits_requested_rate(lorenz_x):
    x, _ = lorenz_x
    tau = embedding.estimate_delay(x, max_lag=80)
    dim, _ = embedding.estimate_dimension(x, tau, max_dim=6)
    R, threshold = recurrence.recurrence_matrix(x[:1000], dim, tau, recurrence_rate=0.1)
    assert R.shape[0] == R.shape[1]
    off_diag_rate = (R.sum() - np.trace(R)) / (R.size - R.shape[0])
    assert abs(off_diag_rate - 0.1) < 0.02
    assert threshold > 0


def test_determinism_decreases_with_min_length(lorenz_x):
    x, _ = lorenz_x
    tau = embedding.estimate_delay(x, max_lag=80)
    dim, _ = embedding.estimate_dimension(x, tau, max_dim=6)
    R, _ = recurrence.recurrence_matrix(x[:1000], dim, tau, recurrence_rate=0.1)
    det_short = recurrence.determinism(R, min_length=2)
    det_long = recurrence.determinism(R, min_length=20)
    assert 0 <= det_long <= det_short <= 1
