"""The compiled evaluators agree with direct computations on small problems."""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zeroreach import search as S                         # noqa: E402
from zeroreach.grids import envelope_bound, ggl_grid, invert_path, paper_grid   # noqa: E402
from zeroreach.randomization import sample_crd            # noqa: E402


def brute_max_p(y, z, draws, regions):
    """Largest plus-one p-value over the given regions, centred statistic k sum(zY) - sum(Y) sum(z)."""
    M = len(draws)
    best = 0.0
    for reg in regions:
        k = len(reg)
        yr = y[reg]
        stat = lambda zz: k * np.dot(zz[reg], yr) - yr.sum() * zz[reg].sum()   # noqa: E731
        t_obs = stat(z)
        count = sum(stat(d) >= t_obs for d in draws)
        best = max(best, (count + 1) / (M + 1))
    return best


@pytest.mark.parametrize('integer', [False, True])
def test_shared1d_matches_brute_force(integer):
    rng = np.random.default_rng(1)
    n, M = 40, 200
    y = rng.integers(0, 4, n).astype(float) if integer else rng.normal(size=n)
    z = sample_crd(n, 20, 1, rng)[0]
    draws = sample_crd(n, 20, M, rng)
    ev = S.Shared1D(y, z, draws)
    for k in (1, 2, 5, 13, 40):
        regions = [np.arange(s, s + k) for s in range(n - k + 1)]
        assert ev.max_p(k) == pytest.approx(brute_max_p(y, z, draws, regions), abs=0)


def test_shared2d_matches_brute_force():
    rng = np.random.default_rng(2)
    n, M = 30, 150
    W = rng.uniform(size=(n, 2))
    y = rng.normal(size=n)
    z = sample_crd(n, 15, 1, rng)[0]
    draws = sample_crd(n, 15, M, rng)
    orders = S.knn_orders(W)
    for k in (30, 17, 6, 2):                       # the evaluator moves downwards through the sizes
        ev = S.Shared2D(y, z, draws, orders)
        regions = [orders[c, :k] for c in range(n)]
        assert ev.max_p(k) == pytest.approx(brute_max_p(y, z, draws, regions), abs=0)


def test_neighbour_orders():
    """Normalized-rank nearest neighbours, the unit itself first, distance ties by the first rank."""
    rng = np.random.default_rng(3)
    W = rng.uniform(size=(25, 2))
    orders = S.knn_orders(W)
    r = np.column_stack([np.argsort(np.argsort(W[:, j], kind='stable')) + 1 for j in (0, 1)])
    for i in range(25):
        d = ((r - r[i]) ** 2).sum(1)
        expected = sorted(range(25), key=lambda j: (d[j], r[j, 0]))
        assert list(orders[i]) == expected


def test_conditional_matches_exact_enumeration():
    """Conditional p-value of one block against the exact enumeration of all assignments."""
    rng = np.random.default_rng(4)
    k, c, M = 10, 4, 40000
    y = rng.normal(size=k)
    z = np.zeros(k, np.int8)
    z[rng.choice(k, c, replace=False)] = 1
    t_obs = y[z == 1].sum()
    sums = [y[list(s)].sum() for s in itertools.combinations(range(k), c)]
    exact = np.mean([s >= t_obs for s in sums])
    ev = S.Cond1D(y, z, M, seed=7)
    p = ev.max_p(k)
    assert abs(p - exact) < 4 * math.sqrt(exact * (1 - exact) / M) + 2 / M


class _Path:
    """Evaluator with a given p-value path."""

    def __init__(self, P):
        self.P, self.cache, self.gauss_cache = P, {}, {}

    def max_p(self, k):
        self.cache[k] = self.P[k]
        return self.P[k]

    def gaussian_max(self, k):
        return self.P[k]


def test_bisection_and_descending_agree_under_monotonicity():
    n = 200
    for cut in (0, 1, 37, 150, 200):
        P = {k: (0.5 if k <= cut else 0.01) for k in range(1, n + 1)}
        assert S.bisection(_Path(P), n)[0] == S.descending(_Path(P), n) == cut
        assert S.bisection(_Path(P), n, warm=True)[0] == cut


def test_grid_inversion():
    rng = np.random.default_rng(5)
    for _ in range(200):
        grid = sorted(rng.choice(np.arange(1, 400), 15, replace=False).tolist())
        P = {k: float(rng.choice([0.01, 0.2])) for k in grid}
        inv = invert_path(grid, [P[k] for k in grid], 0.05, 400)
        bound, rstar = envelope_bound(P, grid, 0.05)
        assert (inv['grid_bound'], inv['first_rejected']) == (bound, rstar)
        assert inv['certificate'] == (400 if rstar is None else rstar - 1)


def test_grids():
    assert paper_grid(1054)[:2] == [53, 106] and len(paper_grid(1054)) == 19
    assert paper_grid(352) == list(range(18, 343, 18))
    g = ggl_grid(119999)
    assert len(g) == 32 and g[0] == 50 and 3388 in g and 4686 in g
