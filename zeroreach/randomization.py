"""Randomization draws: complete randomization, randomization within strata, and permutation within cells.

Every sampler takes a numpy Generator and fixes the order in which it is used, so a seed reproduces the
draws of the paper exactly. Draws are returned as an (M, n) int8 array of treatment indicators.
"""
from __future__ import annotations

import numpy as np


def sample_crd(n, n1, M, rng):
    """M independent completely randomized assignments of n1 treated units among n."""
    n, n1, M = int(n), int(n1), int(M)
    if n < 1 or M < 1 or not 0 <= n1 <= n:
        raise ValueError('need n >= 1, M >= 1 and 0 <= n1 <= n')
    if not isinstance(rng, np.random.Generator):
        raise ValueError('rng must be a numpy.random.Generator')
    draws = np.zeros((M, n), dtype=np.int8)
    for row in draws:
        row[rng.choice(n, n1, replace=False)] = 1
    return draws


def sample_within_strata(strata, z, M, rng):
    """M assignments that randomize completely within each stratum, keeping the observed number of
    treated units per stratum (ACTG 175, Section 6.8). Draw by draw, the strata are visited in sorted
    order of their labels."""
    strata = np.asarray(strata)
    z = np.asarray(z)
    n = len(strata)
    groups = [np.flatnonzero(strata == lv) for lv in np.unique(strata)]
    treated = [int(z[g].sum()) for g in groups]
    draws = np.zeros((int(M), n), np.int8)
    for b in range(int(M)):
        for g, n1 in zip(groups, treated):
            draws[b, rng.choice(g, n1, replace=False)] = 1
    return draws


def sample_half_within_strata(strata, M, rng):
    """M within-stratum complete randomizations with half of each stratum treated (stratified design of
    Appendix H.1)."""
    n = len(strata)
    draws = np.zeros((M, n), np.int8)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        n1 = len(idx) // 2
        u = rng.random((M, len(idx)))
        order = np.argsort(u, axis=1)
        chosen = idx[order[:, :n1]]
        rows = np.repeat(np.arange(M), n1)
        draws[rows, chosen.ravel()] = 1
    return draws


def permutations_within_cells(cells, z, M, seed):
    """Observed assignment in row 0 and M permutations of the assignment within each cell in rows 1..M
    (the recorded design of the Gerber-Green-Larimer experiment, Section 6.7). Cells are visited in order
    of first appearance."""
    import pandas as pd
    z0 = np.asarray(z, dtype=np.int8)
    n = len(z0)
    rng = np.random.default_rng(seed)
    by_cell = pd.DataFrame({'cell_id': np.asarray(cells)}).groupby('cell_id', sort=False).indices
    perms = np.empty((int(M) + 1, n), dtype=np.int8)
    perms[0] = z0
    cell_idx = list(by_cell.values())
    for m in range(1, int(M) + 1):
        row = z0.copy()
        for idx in cell_idx:
            row[idx] = z0[idx][rng.permutation(len(idx))]
        perms[m] = row
    return perms
