"""Shared internal helper: find the longest stretch of a curve over which
its local slope stays roughly constant.

Both the Lyapunov divergence curve (log-divergence vs. time) and the
correlation-sum curve (log C(r) vs. log r) are, in the region that
actually matters, supposed to be straight lines whose slope is the answer
-- and both are contaminated by real curve shape on either side of that
region (a transient/small-sample-noise regime and a saturation regime).
A single detection method serves both.
"""

from __future__ import annotations

import numpy as np


def find_longest_stable_slope_region(
    x: np.ndarray,
    y: np.ndarray,
    window_frac: float = 0.05,
    tol: float = 0.15,
    min_window: int = 8,
) -> tuple[float, float]:
    """Return `(x_min, x_max)` of the longest run of consecutive local-slope
    windows whose slopes agree with each other to within `tol` (relative).

    Fitting a fixed-width window and taking whichever has the smallest
    residual variance sounds equivalent but isn't: real curves of this
    shape have brief inflection points -- in the transient, in the
    saturated/noisy tail, anywhere concavity flips -- that are locally
    dead straight for a window or two, and residual variance alone can't
    tell those apart from the genuine scaling region. Requiring the same
    slope to hold over the longest *run* of windows is a stronger test:
    transients and noise change slope quickly, but a real power law (or
    exponential) holds its rate over a real stretch of the curve.
    """
    valid = ~np.isnan(y) & ~np.isnan(x) & np.isfinite(y) & np.isfinite(x)
    xv, yv = np.asarray(x)[valid], np.asarray(y)[valid]
    n = len(xv)
    window = max(min_window, int(n * window_frac))
    if n - window < 2:
        return float(xv[0]), float(xv[-1]) if n else (0.0, 0.0)

    local_slopes = np.array(
        [np.polyfit(xv[i : i + window], yv[i : i + window], 1)[0] for i in range(n - window)]
    )
    starts = xv[: len(local_slopes)]

    best_span = -1.0
    best_range = (float(xv[0]), float(xv[-1]))
    i = 0
    while i < len(local_slopes):
        j = i
        ref = local_slopes[i]
        while j + 1 < len(local_slopes) and abs(local_slopes[j + 1] - ref) <= tol * max(abs(ref), 1e-12):
            j += 1
        end_idx = min(j + window, n - 1)
        span = xv[end_idx] - starts[i]
        if span > best_span:
            best_span = span
            best_range = (float(starts[i]), float(xv[end_idx]))
        i = j + 1 if j > i else i + 1
    return best_range
