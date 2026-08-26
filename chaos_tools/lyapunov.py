"""Estimate the largest Lyapunov exponent from a scalar time series.

This is the single number this whole repo keeps coming back to in one form
or another: how fast do two trajectories that started almost identical pull
apart? A positive largest Lyapunov exponent (LLE) is the actual mathematical
definition of "chaotic" -- it's what the double-pendulum toy's live
divergence readout is gesturing at informally, with a physics simulation
standing in for a real measured signal. This module estimates it from data
alone, the way you'd have to if you only had that signal and not the
equations.

Implements Rosenstein, Collins & De Luca's method (*Physica D* 65, 117,
1993): find each point's nearest neighbor in the reconstructed phase space,
track how fast the two trajectories separate, and read the Lyapunov
exponent off the slope of log-divergence vs. time.
"""

from __future__ import annotations

import numpy as np

from ._scaling import find_longest_stable_slope_region
from .embedding import delay_embed


def _nearest_neighbor_indices(emb: np.ndarray, min_tsep: int) -> np.ndarray:
    """Nearest neighbor of each point, excluding temporal neighbors within
    `min_tsep` samples (the "Theiler window") so the pair is actually a
    close *recurrence*, not just the same bit of trajectory twice."""
    n = len(emb)
    idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        diffs = emb - emb[i]
        dist2 = np.einsum("ij,ij->i", diffs, diffs)
        lo = max(0, i - min_tsep)
        hi = min(n, i + min_tsep + 1)
        dist2[lo:hi] = np.inf
        idx[i] = np.argmin(dist2)
    return idx


def divergence_curve(
    x: np.ndarray,
    dim: int,
    delay: int,
    dt: float = 1.0,
    min_tsep: int | None = None,
    max_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average log-divergence between nearest-neighbor trajectory pairs, as
    a function of time step `k`. This is the curve whose slope (over
    whatever region turns out to be linear) is the Lyapunov exponent --
    return it so you can look at it, which the method genuinely requires.

    Returns
    -------
    times : array, shape (max_k + 1,)
        `k * dt` for k = 0..max_k.
    log_divergence : array, shape (max_k + 1,)
        Mean of `ln ||X[i+k] - X[nn(i)+k]||` over all valid `i`, or NaN
        where no pair had data that far ahead.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    emb = delay_embed(x, dim, delay)
    n = len(emb)
    if min_tsep is None:
        min_tsep = delay
    if max_k is None:
        # Wide enough, by default, to have a real shot at reaching past the
        # short-term transient into the exponential-divergence plateau
        # before the curve saturates -- see _find_linear_region, which then
        # only actually searches its first search_frac. Still just a
        # default: genuinely different systems saturate on genuinely
        # different timescales, so tune this for your own data.
        max_k = max(1, min(n // 2, 3000))
    max_k = min(max_k, n - 1)

    nn = _nearest_neighbor_indices(emb, min_tsep)

    log_div = np.full(max_k + 1, np.nan)
    all_i = np.arange(n)
    for k in range(max_k + 1):
        valid = (all_i + k < n) & (nn + k < n)
        i_idx = all_i[valid]
        j_idx = nn[i_idx]
        d = np.linalg.norm(emb[i_idx + k] - emb[j_idx + k], axis=1)
        d = d[d > 0]
        if len(d):
            log_div[k] = np.mean(np.log(d))

    return np.arange(max_k + 1) * dt, log_div


def _find_linear_region(times: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Find the longest stretch of the divergence curve over which the
    local slope stays roughly constant, and return its `(t_min, t_max)`.

    Divergence curves have a noisy, often steep transient at small k
    (nearest neighbors are close enough that non-exponential short-term
    behavior dominates), then -- if `max_k` reaches far enough -- a
    genuine exponential-divergence regime, and finally saturate once pairs
    are as far apart as the attractor is wide, at which point "divergence"
    stops growing and the curve just wobbles around a ceiling.

    Delegates to `_scaling.find_longest_stable_slope_region`: a fixed-width
    window with the smallest fit-residual variance sounds like it would
    work, but doesn't -- a curve this shape almost always has brief
    inflection points that are locally dead straight for a window or two,
    indistinguishable from the real thing on residual variance alone.
    Requiring the same slope to hold over the longest *run* of windows is
    a stronger test. Still just a default, not a replacement for plotting
    `times, log_divergence` yourself.
    """
    return find_longest_stable_slope_region(times, y)


def largest_lyapunov_exponent(
    x: np.ndarray,
    dim: int,
    delay: int,
    dt: float = 1.0,
    min_tsep: int | None = None,
    max_k: int | None = None,
    fit_range: tuple[float, float] | None = None,
) -> tuple[float, np.ndarray, np.ndarray, tuple[float, float]]:
    """Estimate the largest Lyapunov exponent (Rosenstein et al., 1993).

    `dim` and `delay` should come from `chaos_tools.embedding` (or be
    known already, if you're validating against a system you understand).
    `fit_range`, if given, is `(t_min, t_max)` in the same time units as
    `dt` -- the region of the divergence curve to fit a line to. Divergence
    curves are noisy at small k and saturate at large k (once the
    trajectories are as far apart as the attractor is wide), so the right
    range is a genuinely visual judgment call; by default, `_find_linear_region`
    picks the longest stretch of consistent slope as a starting point, not
    a substitute for looking at the plot yourself.

    Returns
    -------
    lle : float
        Estimated largest Lyapunov exponent, in units of 1/`dt`-units.
        Positive means chaotic (by this measure); the more positive, the
        faster nearby trajectories diverge.
    times, log_divergence : arrays
        The full curve from `divergence_curve`, for plotting.
    fit_range : (float, float)
        The `(t_min, t_max)` actually used for the fit -- either what you
        passed in, or whatever the automatic search chose, handed back so
        you can draw it on the plot without recomputing it.
    """
    times, log_div = divergence_curve(x, dim, delay, dt=dt, min_tsep=min_tsep, max_k=max_k)

    valid = ~np.isnan(log_div)
    if valid.sum() < 2:
        raise ValueError("not enough valid points on the divergence curve to fit")
    if fit_range is None:
        lo, hi = _find_linear_region(times, log_div)
    else:
        lo, hi = fit_range

    mask = valid & (times >= lo) & (times <= hi)
    if mask.sum() < 2:
        raise ValueError(f"fit_range {(lo, hi)} contains fewer than 2 valid points")

    slope, _ = np.polyfit(times[mask], log_div[mask], 1)
    return float(slope), times, log_div, (lo, hi)
