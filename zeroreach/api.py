"""upper_bound(): the procedure of the paper for one dataset, in one call."""
from __future__ import annotations

import numpy as np

from . import search as S
from .grids import envelope_bound, paper_grid
from .randomization import sample_crd, sample_within_strata


def upper_bound(W, Y, Z, alpha=0.05, M=5000, grid='all', seed=0, draws=None, strata=None):
    """Upper confidence bound for the size of the largest zero-effect region.

    W      pre-treatment covariate: shape (n,) for blocks of consecutive units in the order of W, or (n, 2)
           for nearest-neighbour regions in normalized ranks. Ties are broken by unit index.
    Y, Z   outcomes and binary treatment indicators (Z = 1 treated).
    grid   'all' (default) for integer candidate sizes 1,...,n, using exact descending search without
           a monotonicity assumption. 'paper' explicitly reproduces the historical 5%-of-n candidate
           grid; an explicit list of integer sizes is also accepted. Sparse-grid bounds concern the
           searched-grid target; certificate is the corresponding upper bound over all integer sizes.
    draws  optional (M, n) array of assignments drawn from the design; by default M completely randomized
           draws with the observed number of treated units (within strata if `strata` is given).

    The statistic is sum((Z-pi)*(Y-mean_B(Y))) within each candidate region.
    Returns a dict with the bound, the first tail-envelope-rejected size r* (None if none is rejected),
    and the certificate r* - 1 (n if r* is None). For 'all', certificate equals bound. It also returns
    the largest p-value at each evaluated size, and `region`: a boolean mask (in the original unit order)
    of the units covered by candidate regions of the reported size that are not rejected. The covered
    units are descriptive; they are not a confidence set for the location of a zero-effect region."""
    W = np.asarray(W, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64)
    z = np.asarray(Z).astype(np.int8)
    n = len(y)
    if W.shape[0] != n or len(z) != n:
        raise ValueError('W, Y and Z must have the same number of units')
    if draws is None:
        rng = np.random.default_rng(seed)
        draws = (sample_crd(n, int(z.sum()), M, rng) if strata is None
                 else sample_within_strata(np.asarray(strata), z, M, rng))
    draws = np.asarray(draws, dtype=np.int8)
    if W.ndim == 1:
        order = np.lexsort((np.arange(n), W))
        ev = S.Shared1D(y[order], z[order], draws[:, order])
        orders = None
    else:
        order = None
        orders = S.knn_orders(W)
        ev = S.Shared2D(y, z, draws, orders)
    out = dict(n=n, M=len(draws), alpha=alpha, statistic='within_region_centered_covariance',
               grid_mode=grid if isinstance(grid, str) else 'explicit', regions_not_rejected=0)
    if isinstance(grid, str) and grid == 'all':
        bound = S.descending(ev, n, alpha)
        out.update(bound=int(bound), first_rejected=int(bound) + 1 if bound < n else None,
                   certificate=int(bound), candidate_sizes=list(range(1, n + 1)))
    else:
        if isinstance(grid, str):
            if grid != 'paper':
                raise ValueError("grid must be 'all', 'paper', or an explicit list of integer sizes")
            sizes = paper_grid(n)
        else:
            values = list(grid)
            if any(not np.isfinite(k) or int(k) != k or not 1 <= k <= n for k in values):
                raise ValueError('Explicit candidate sizes must be integers between 1 and n')
            sizes = sorted(set(int(k) for k in values))
        out['candidate_sizes'] = sizes
        P = {k: float(ev.max_p(k)) for k in sorted(sizes, reverse=True)}
        bound, rstar = envelope_bound(P, sizes, alpha)
        out.update(bound=int(bound), first_rejected=rstar, certificate=n if rstar is None else rstar - 1)
    out['P'] = {int(k): float(v) for k, v in sorted(ev.cache.items())}
    region = np.zeros(n, bool)
    if out['bound'] > 0:
        p = ev.local_p(out['bound'])
        if W.ndim == 1:
            for s in np.flatnonzero(p > alpha):
                region[order[s:s + out['bound']]] = True
        else:
            for c in np.flatnonzero(p > alpha):
                region[orders[c, :out['bound']]] = True
        out['regions_not_rejected'] = int((p > alpha).sum())
    out['region'] = region
    return out
