"""Candidate grids and tail-envelope inversion; sparse helpers retain historical reproducibility.

On a pre-specified grid the bound is the largest searched size not rejected by the tail envelope
P_r^up = max_{t >= r} P_t; the next searched size r* is rejected, and r* - 1 is an upper confidence bound
for the size over all integer sizes (the certificate of Appendix E.6).
"""
from __future__ import annotations

import numpy as np


def paper_grid(n):
    """Historical 5%-of-n grid: step max(1, round(0.05 n)); neither 1 nor n is forced in.

    Current paper experiments use all integer sizes; this helper is an explicit legacy option."""
    n = int(n)
    if n < 1:
        raise ValueError('n must be a positive integer')
    step = max(1, round(0.05 * n))
    return list(range(step, n + 1, step))


def ggl_grid(n):
    """Historical sparse Gerber-Green-Larimer candidate lengths: fixed lengths
    50, 100, 200, 500, 1000, 2000, 5000, 10000 together with 25 lengths equally spaced on a log scale
    between 50 and n."""
    if n < 50:
        raise ValueError('the GGL grid starts at 50')
    anchors = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    targets = sorted(set([a for a in anchors if a <= n]))
    log_pts = np.unique(np.round(np.logspace(np.log10(50), np.log10(n), 25)).astype(int))
    grid = sorted(set(list(targets) + list(log_pts.tolist())))
    return [int(r) for r in grid if 50 <= r <= n]


def invert_path(grid, P, alpha, n):
    """Invert a p-value path on its grid with the tail envelope; no monotonicity is assumed.

    Returns grid_bound (largest searched size not rejected by the envelope; 0 if every searched size is
    rejected), first_rejected (r*, or None) and certificate (r* - 1, or n when nothing is rejected)."""
    grid = [int(k) for k in grid]
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 1 or len(P) != len(grid):
        raise ValueError('P must have one entry per grid size')
    envelope = np.maximum.accumulate(P[::-1])[::-1] if len(P) else P.copy()
    rejected = np.flatnonzero(envelope <= alpha)
    if len(rejected):
        j = int(rejected[0])
        first = int(grid[j])
        bound = int(grid[j - 1]) if j else 0
    else:
        first = None
        bound = int(grid[-1]) if len(grid) else 0
    return dict(grid_bound=bound, first_rejected=first, certificate=int(n) if first is None else first - 1,
                envelope=envelope.tolist())


def grid_scan(evaluator, n, alpha, grid):
    """Evaluate max_p on the grid from the largest size downwards, stopping at the first size that is not
    rejected, and invert the evaluated path. This gives the same grid bound and certificate as evaluating
    the whole grid. Returns the inversion with the evaluated sizes (ascending) and their p-values."""
    values = {}
    for k in reversed(list(grid)):
        values[int(k)] = float(evaluator.max_p(int(k)))
        if values[int(k)] > alpha:
            break
    sizes = sorted(values)
    result = invert_path(sizes, [values[k] for k in sizes], alpha, n)
    result.update(grid=sizes, P=[values[k] for k in sizes])
    return result


def envelope_bound(P, grid, alpha):
    """For a path {size: P} on the whole grid: the grid bound (largest size with P > alpha) and the
    first size r* rejected by the tail envelope (None if no size is rejected)."""
    ks = sorted(int(k) for k in grid)
    bound = max([k for k in ks if P[k] > alpha], default=0)
    env, rstar = 0.0, None
    for k in reversed(ks):
        env = max(env, P[k])
        if env <= alpha:
            rstar = k
    return int(bound), (int(rstar) if rstar is not None else None)
