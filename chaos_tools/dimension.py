"""Correlation dimension: how many numbers does it actually take to pin
down a state on this attractor?

An integer dimension means an ordinary object -- 1 for a line, 2 for a
sheet. A *fractional* one is the signature of a strange attractor: the
trajectory never repeats and never crosses itself, yet stays confined to a
bounded region forever, and it does that by folding into structure at every
scale (the same self-similarity the bifurcation-diagram toy shows directly).
The Lorenz attractor's dimension is about 2.05 -- almost a surface, but not
quite, forever.

Implements the Grassberger-Procaccia algorithm (*Physica D* 9, 189, 1983):
count, for shrinking radii `r`, what fraction of all point-pairs in the
reconstructed phase space are closer together than `r`. For a fractal of
dimension `D`, that fraction (the "correlation sum" `C(r)`) scales as
`r**D`, so `D` is the slope of `log C(r)` vs. `log r` in the scaling region.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from ._scaling import find_longest_stable_slope_region
from .embedding import delay_embed


def correlation_sum(emb: np.ndarray, radii: np.ndarray, max_pairs: int | None = 4_000_000) -> np.ndarray:
    """Fraction of point-pairs in `emb` closer together than each radius.

    `max_pairs`: if `emb` would produce more pairs than this, it's
    randomly subsampled first -- correlation sum is O(n^2) pairs, and this
    keeps a few-thousand-point embedding tractable without silently hanging
    on a much larger one.
    """
    n = len(emb)
    if max_pairs is not None and n * (n - 1) // 2 > max_pairs:
        n_keep = max(2, int((2 * max_pairs) ** 0.5))
        idx = np.random.default_rng(0).choice(n, size=n_keep, replace=False)
        emb = emb[idx]

    d = pdist(emb)
    d.sort()
    radii = np.asarray(radii, dtype=np.float64)
    counts = np.searchsorted(d, radii, side="left")
    return counts / len(d)


def correlation_dimension(
    x: np.ndarray,
    dim: int,
    delay: int,
    n_radii: int = 40,
    fit_range: tuple[float, float] | None = None,
    max_pairs: int | None = 4_000_000,
) -> tuple[float, np.ndarray, np.ndarray, tuple[float, float]]:
    """Estimate the correlation dimension D2 (Grassberger & Procaccia, 1983).

    `fit_range`, if given, is `(r_min, r_max)` -- the radius range to fit
    `log C(r)` vs `log r` over. Like the Lyapunov exponent's divergence
    curve, the scaling region is a real judgment call: too-small `r` is
    dominated by a handful of pairs (noisy, and in practice the true
    scaling region often starts here, at radii well below the bulk of the
    pairwise-distance distribution -- see `find_longest_stable_slope_region`
    in `_scaling.py`), and too-large `r` saturates as C(r) approaches 1,
    i.e. "every pair is within this radius." The default searches for the
    longest stretch of `log C(r)` vs `log r` with consistent slope, the
    same method `chaos_tools.lyapunov` uses for its divergence curve.

    Returns
    -------
    d2 : float
        Estimated correlation dimension.
    radii, corr_sum : arrays, shape (n_radii,)
        The full C(r) curve, for plotting.
    fit_range : (float, float)
        The `(r_min, r_max)` actually used for the fit -- either what you
        passed in, or whatever the automatic search chose, handed back so
        you can draw it on the plot without recomputing it.
    """
    emb = delay_embed(x, dim, delay)
    n = len(emb)
    if max_pairs is not None and n * (n - 1) // 2 > max_pairs:
        n_keep = max(2, int((2 * max_pairs) ** 0.5))
        idx = np.random.default_rng(0).choice(n, size=n_keep, replace=False)
        sample = emb[idx]
    else:
        sample = emb

    d = pdist(sample)
    d = d[d > 0]
    d.sort()
    # Reach down toward the smallest pairwise distances, not just the bulk
    # of the distribution -- the true power-law scaling region is often
    # down there, before the correlation sum has accumulated enough pairs
    # to look smooth.
    r_min, r_max = np.percentile(d, [0.02, 60])
    radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)
    corr_sum = np.searchsorted(d, radii, side="left") / len(d)

    valid = corr_sum > 0
    log_r = np.log(radii[valid])
    log_c = np.log(corr_sum[valid])

    if fit_range is None:
        lo, hi = find_longest_stable_slope_region(log_r, log_c)
    else:
        lo, hi = np.log(fit_range[0]), np.log(fit_range[1])
    mask = (log_r >= lo) & (log_r <= hi)

    if mask.sum() < 2:
        raise ValueError("fit_range contains fewer than 2 valid points")

    slope, _ = np.polyfit(log_r[mask], log_c[mask], 1)
    return float(slope), radii, corr_sum, (float(np.exp(lo)), float(np.exp(hi)))
