"""Phase-space reconstruction from a single scalar time series.

Takens' theorem (1981) says that if you only ever measured one variable of a
higher-dimensional deterministic system -- one thermometer, one microphone,
one predator count -- you can still reconstruct a phase space that is
(generically) equivalent to the real one, just by stacking delayed copies of
that one signal: X(t) = [x(t), x(t+tau), x(t+2*tau), ..., x(t+(m-1)*tau)].

The two knobs are the delay `tau` and the embedding dimension `m`. This
module estimates both from data instead of asking the caller to guess:
`tau` from the first local minimum of the average mutual information, and
`m` from the false-nearest-neighbors test (Kennel, Brown & Abarbanel, 1992).
"""

from __future__ import annotations

import numpy as np


def delay_embed(x: np.ndarray, dim: int, delay: int) -> np.ndarray:
    """Reconstruct phase space from a scalar series via delay coordinates.

    Parameters
    ----------
    x : array, shape (n,)
        Scalar time series.
    dim : int
        Embedding dimension `m` (>= 1).
    delay : int
        Delay `tau` in samples (>= 1).

    Returns
    -------
    array, shape (n - (dim - 1) * delay, dim)
        Row `i` is `[x[i], x[i + delay], ..., x[i + (dim - 1) * delay]]`.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    n_rows = len(x) - (dim - 1) * delay
    if n_rows < 1:
        raise ValueError(
            f"series too short: need > {(dim - 1) * delay} points for "
            f"dim={dim}, delay={delay}, got {len(x)}"
        )
    out = np.empty((n_rows, dim), dtype=np.float64)
    for d in range(dim):
        out[:, d] = x[d * delay : d * delay + n_rows]
    return out


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 16) -> float:
    """Mutual information (in nats) between two equal-length 1D arrays,
    estimated by binning into a 2D histogram."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    joint, _, _ = np.histogram2d(x, y, bins=bins)
    joint = joint / joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = joint / (px * py)
        terms = joint * np.log(ratio)
    terms = np.where(joint > 0, terms, 0.0)
    return float(terms.sum())


def estimate_delay(x: np.ndarray, max_lag: int = 100, bins: int = 16) -> int:
    """Estimate a good embedding delay `tau` as the first local minimum of
    the average mutual information between `x(t)` and `x(t + lag)`.

    A local minimum of AMI is the point where the delayed copy has stopped
    telling you what the original already told you, but before it's so
    delayed that (for a chaotic system) it's telling you nothing at all --
    the standard justification for this choice (Fraser & Swinney, 1986).
    Falls back to the lag of steepest AMI drop if no interior minimum is
    found within `max_lag`.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    max_lag = min(max_lag, len(x) // 4)
    if max_lag < 2:
        return 1

    ami = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        ami[lag - 1] = mutual_information(x[:-lag], x[lag:], bins=bins)

    for i in range(1, max_lag - 1):
        if ami[i] < ami[i - 1] and ami[i] < ami[i + 1]:
            return i + 1  # lags are 1-indexed; ami[i] is lag i+1

    return int(np.argmin(np.diff(ami))) + 1


def false_nearest_neighbors(
    x: np.ndarray,
    delay: int,
    max_dim: int = 10,
    rtol: float = 15.0,
    atol: float = 2.0,
) -> np.ndarray:
    """False-nearest-neighbors fraction at each candidate embedding dimension.

    For each dimension `m`, every point's nearest neighbor in the embedded
    space is checked against the same pair one dimension higher: if adding
    that extra coordinate makes them fly apart, the neighbor was an
    artifact of projecting a higher-dimensional attractor down into too few
    dimensions -- a "false" neighbor. The fraction of false neighbors drops
    toward zero once `m` reaches the true embedding dimension (Kennel,
    Brown & Abarbanel, *Phys. Rev. A* 45, 3403, 1992).

    Returns
    -------
    array, shape (max_dim,)
        `result[m - 1]` is the false-neighbor fraction at dimension `m`.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    attractor_size = float(np.std(x))
    fractions = np.zeros(max_dim)

    for m in range(1, max_dim + 1):
        emb_m = delay_embed(x, m, delay)
        emb_m1 = delay_embed(x, m + 1, delay)
        n = len(emb_m1)  # emb_m has n + delay extra rows; only the first n line up
        pts_m = emb_m[:n]

        false_count = 0
        for i in range(n):
            diffs = pts_m - pts_m[i]
            dist2 = np.einsum("ij,ij->i", diffs, diffs)
            dist2[i] = np.inf
            j = int(np.argmin(dist2))
            d_m = np.sqrt(dist2[j])
            if d_m == 0:
                continue

            extra = emb_m1[i, m] - emb_m1[j, m]
            d_m1 = np.sqrt(d_m * d_m + extra * extra)

            criterion1 = abs(extra) / d_m > rtol
            criterion2 = attractor_size > 0 and d_m1 / attractor_size > atol
            if criterion1 or criterion2:
                false_count += 1

        fractions[m - 1] = false_count / n if n > 0 else 0.0

    return fractions


def estimate_dimension(
    x: np.ndarray, delay: int, max_dim: int = 10, threshold: float = 0.05
) -> tuple[int, np.ndarray]:
    """Suggest an embedding dimension via false nearest neighbors.

    Returns `(dim, fractions)` where `dim` is the smallest dimension whose
    false-neighbor fraction drops below `threshold` (or the dimension with
    the smallest fraction, if none clears the threshold), and `fractions`
    is the full per-dimension curve from `false_nearest_neighbors`.
    """
    fractions = false_nearest_neighbors(x, delay, max_dim=max_dim)
    below = np.where(fractions < threshold)[0]
    dim = int(below[0]) + 1 if below.size else int(np.argmin(fractions)) + 1
    return dim, fractions
