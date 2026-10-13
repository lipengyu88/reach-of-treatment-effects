"""Exact small-design oracles for the raw-versus-centered comparison.

These enumerate global complete randomization and evaluate the paper's formulas
with rational arithmetic, independently of the prefix/running-sum implementation.
The production plus-one convention is retained even with an enumerated draw bank.
"""
import itertools
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zeroreach import search as S  # noqa: E402


def assignments(n, n1):
    result = []
    for treated in itertools.combinations(range(n), n1):
        z = np.zeros(n, np.int8)
        z[list(treated)] = 1
        result.append(z)
    return np.asarray(result)


def local_oracle(y, z, draws, regions, centered=False):
    values = [Fraction.from_float(float(v)) for v in y]
    pi = Fraction(int(z.sum()), len(z))
    result = []
    for region in regions:
        avg = sum(values[i] for i in region) / len(region) if centered else Fraction(0)

        def statistic(assignment):
            return sum((int(assignment[i]) - pi) * (values[i] - avg) for i in region)

        observed = statistic(z)
        count = sum(statistic(draw) >= observed for draw in draws)
        result.append((1 + count) / (1 + len(draws)))
    return np.asarray(result)


def make_evaluator(dim, y, z, draws, centered=False, layout='staggered'):
    n = len(y)
    if dim == 1:
        regions = lambda k: [np.arange(s, s + k) for s in range(n - k + 1)]
        cls = S.Shared1D if centered else S.Shared1DRaw
        return cls(y, z, draws), regions
    W = np.column_stack((np.arange(n), np.roll(np.arange(n)[::-1], 2)))
    orders = S.knn_orders(W)
    regions = lambda k: [orders[c, :k] for c in range(n)]
    cls = S.Shared2D if centered else S.Shared2DRaw
    return cls(y, z, draws, orders, layout=layout), regions


@pytest.mark.parametrize('dim', [1, 2])
@pytest.mark.parametrize('n1', [0, 1, 3, 5, 6])
@pytest.mark.parametrize('y', [
    np.array([-3., 0., 2., -1., 4., 2.]),
    np.ones(6),
    -np.ones(6),
    np.zeros(6),
])
def test_raw_all_regions_against_exact_global_randomization(dim, n1, y):
    draws = assignments(len(y), n1)
    # Every possible observed allocation, all sizes, and every candidate region.
    for z in draws:
        ev, regions = make_evaluator(dim, y, z, draws)
        for k in range(len(y), 0, -1):
            expected = local_oracle(y, z, draws, regions(k))
            np.testing.assert_array_equal(ev.local_p(k), expected)
            assert ev.max_p(k) == expected.max()
        assert ev.unresolved == 0


@pytest.mark.parametrize('dim', [1, 2])
def test_raw_cancellation_and_nonstructural_ties(dim):
    # The one-unit 2**-80 difference can disappear in a prefix difference.
    # Exact expansions must distinguish it from a tie and also include true
    # ties between different selected subsets (the two equal outcomes 3).
    y = np.array([2. ** 40, 2. ** -80, -(2. ** 40), 3., 3., 0.])
    draws = assignments(len(y), 3)
    for z in draws:
        ev, regions = make_evaluator(dim, y, z, draws)
        for k in (6, 1, 5, 2, 6, 3, 4, 1):
            np.testing.assert_array_equal(ev.local_p(k), local_oracle(y, z, draws, regions(k)))
        assert ev.unresolved == 0


@pytest.mark.parametrize('dim', [1, 2])
@pytest.mark.parametrize('alpha', [0.05, 0.3, 0.8, 1.0])
def test_raw_descending_matches_complete_integer_inversion(dim, alpha):
    y = np.array([-3., 0., 2., -1., 4., 2.])
    draws = assignments(len(y), 3)
    for z in draws:
        ev, regions = make_evaluator(dim, y, z, draws)
        path = {k: local_oracle(y, z, draws, regions(k)).max() for k in range(1, len(y) + 1)}
        expected = max([k for k, p in path.items() if p > alpha], default=0)
        assert ev.descending(alpha, chunk=2) == expected
        ev, _ = make_evaluator(dim, y, z, draws)
        assert S.descending(ev, len(y), alpha) == expected


@pytest.mark.parametrize('dim', [1, 2])
def test_centered_default_is_unchanged_and_differs_from_raw(dim):
    y = np.ones(6)
    draws = assignments(len(y), 3)
    z = draws[0]
    centered, regions = make_evaluator(dim, y, z, draws, centered=True)
    raw, _ = make_evaluator(dim, y, z, draws)
    for k in range(1, len(y) + 1):
        np.testing.assert_array_equal(centered.local_p(k), np.ones(len(regions(k))))
    # A treated singleton has raw p < 1, even with a constant outcome.
    assert raw.local_p(1).min() < 1
    assert raw.local_p(len(y)).min() == 1

    y = np.array([-3., 0., 2., -1., 4., 2.])
    for z in draws:
        centered, regions = make_evaluator(dim, y, z, draws, centered=True)
        raw, _ = make_evaluator(dim, y, z, draws)
        for k in range(len(y), 0, -1):
            expected = local_oracle(y, z, draws, regions(k), centered=True)
            np.testing.assert_array_equal(centered.local_p(k), expected)
        # At the whole sample, fixed global treatment count makes raw and
        # centered p-values equal. This does not hold for smaller regions.
        np.testing.assert_array_equal(raw.local_p(len(y)), centered.local_p(len(y)))


def test_raw_2d_layouts_and_centered_gaussian_proposal_guard():
    y = np.array([-3., 0., 2., -1., 4., 2.])
    draws = assignments(len(y), 3)
    z = draws[0]
    ev, _ = make_evaluator(2, y, z, draws)
    legacy, _ = make_evaluator(2, y, z, draws, layout='legacy')
    for k in (6, 2, 5, 1):
        np.testing.assert_array_equal(ev.local_p(k), legacy.local_p(k))
    for dim in (1, 2):
        raw, _ = make_evaluator(dim, y, z, draws)
        with pytest.raises(NotImplementedError, match='centered Gaussian'):
            raw.gaussian_max(3)
