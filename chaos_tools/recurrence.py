"""Recurrence plots: when does the system come back to somewhere it's been?

A deterministic system, chaotic or not, keeps revisiting the neighborhood of
old states -- it's confined to an attractor, so it has nowhere else to go.
A recurrence plot marks every pair of times `(i, j)` where the system was
in a similar state, and the resulting texture is diagnostic almost by eye:
scattered dust looks like noise, solid blocks mean the system got stuck
near a fixed point, evenly spaced diagonal lines mean periodicity, and
short, broken, off-and-on diagonals -- recognizable structure that never
quite repeats -- are the signature of chaos.

Introduced by Eckmann, Kamphorst & Ruelle (*Europhys. Lett.* 4, 973, 1987).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .embedding import delay_embed


def recurrence_matrix(
    x: np.ndarray,
    dim: int,
    delay: int,
    threshold: float | None = None,
    recurrence_rate: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Binary recurrence matrix `R[i, j] = 1` if states `i` and `j` are
    within `threshold` of each other.

    If `threshold` isn't given, it's chosen so that (approximately)
    `recurrence_rate` of all off-diagonal pairs recur -- 5-15% is a
    conventional starting range; too low and the plot is empty, too high
    and it's solid black.

    Returns
    -------
    R : array, shape (n, n), dtype uint8
    threshold : float
        The distance threshold actually used.
    """
    emb = delay_embed(x, dim, delay)
    d = squareform(pdist(emb))
    if threshold is None:
        off_diag = d[np.triu_indices_from(d, k=1)]
        threshold = float(np.quantile(off_diag, recurrence_rate))
    R = (d <= threshold).astype(np.uint8)
    return R, threshold


def determinism(R: np.ndarray, min_length: int = 2) -> float:
    """Fraction of recurrence points that lie on a diagonal line of at
    least `min_length` consecutive points, excluding the trivial main
    diagonal. Low determinism reads as noise-like; high but imperfect
    determinism (not 1.0 -- a genuinely periodic signal would score close
    to that) is the classic chaotic signature: real structure, broken up.
    """
    n = R.shape[0]
    total_points = 0
    diag_points = 0
    for k in range(1, n):  # positive offsets only; matrix is symmetric
        diag = np.diagonal(R, offset=k)
        total_points += 2 * int(diag.sum())  # symmetric: offset k and -k
        if diag.sum() == 0:
            continue
        # run-length encode consecutive 1s
        padded = np.concatenate(([0], diag, [0]))
        edges = np.diff(padded.astype(np.int8))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        lengths = ends - starts
        diag_points += 2 * int(lengths[lengths >= min_length].sum())
    if total_points == 0:
        return 0.0
    return diag_points / total_points
