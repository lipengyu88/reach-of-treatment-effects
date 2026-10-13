"""Full-integer real-data inversion and reported regions match independent enumeration."""
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runners import run_real_data as R
from zeroreach import search as S
from zeroreach.randomization import sample_crd, sample_within_strata


def _direct(y, z, draws, regions):
    """Direct statistic comparison, independent of the compensated kernels and searches."""
    result = []
    for indices in regions:
        yy = y[indices]
        observed = len(indices) * np.dot(z[indices], yy) - yy.sum() * z[indices].sum()
        simulated = len(indices) * (draws[:, indices] @ yy) - yy.sum() * draws[:, indices].sum(axis=1)
        result.append((1 + np.count_nonzero(simulated >= observed)) / (len(draws) + 1))
    return np.asarray(result)


@pytest.mark.parametrize('values', [
    [0.8, 0.01, 0.7, 0.01, 0.01],  # a nonmonotone path: bisection must not replace descending
    [0.8] * 5,
    [0.01] * 5,
])
def test_full_integer_search_nonmonotone_and_endpoints(values):
    class Evaluator:
        def __init__(self):
            self.cache = {}

        def max_p(self, k):
            self.cache[k] = values[k - 1]
            return self.cache[k]

    n = len(values)
    inv, path = R._search_sizes(Evaluator(), n, 'all')
    expected = max((k for k in range(1, n + 1) if values[k - 1] > R.ALPHA), default=0)
    assert inv == dict(grid_bound=expected, first_rejected=expected + 1 if expected < n else None,
                       certificate=expected)
    assert sorted(path) == list(range(max(1, expected), n + 1))


@pytest.fixture
def actg_sample():
    rng = np.random.default_rng(197)
    n = 42
    z = sample_crd(n, n // 2, 1, rng)[0]
    baseline = np.sort(rng.integers(10, 100, n))
    frame = pd.DataFrame(dict(pidnum=np.arange(n), cd40=baseline,
                              cd420=baseline + 20 * z + rng.integers(-10, 10, n),
                              age=rng.integers(18, 70, n)))
    return frame, z, np.arange(n) % 3


def test_actg_one_masks_follow_all_integer_bound(actg_sample):
    frame, z, strata = actg_sample
    results, draws = R._actg_one(frame, z, strata, 59, 19, True)
    n = len(frame)
    for score, y in [('raw', frame.cd420.to_numpy()), ('change', (frame.cd420 - frame.cd40).to_numpy())]:
        paths = {k: _direct(y, z, draws, [np.arange(s, s + k) for s in range(n - k + 1)])
                 for k in range(1, n + 1)}
        bound = max(k for k, p in paths.items() if p.max() > R.ALPHA)
        mask = np.zeros(n, bool)
        for start in np.flatnonzero(paths[bound] > R.ALPHA):
            mask[start:start + bound] = True
        rec = results[score]
        assert rec['bound'] == rec['certificate'] == bound
        assert rec['evaluated_sizes'] == list(range(bound, n + 1))
        assert rec['covered_units'] == int(mask.sum())
        np.testing.assert_array_equal(rec['mask'], mask)


def test_actg_two_masks_follow_integer_not_old_grid(actg_sample, monkeypatch, tmp_path):
    frame, z, strata = actg_sample
    monkeypatch.setattr(R.RD, 'load_actg', lambda _: actg_sample)
    monkeypatch.setattr(R, 'ACTG_M', 59)
    monkeypatch.setattr(R, 'ACTG_SEED', 19)
    R.run_actg(tmp_path, tmp_path / 'output', False, True, primary=False)
    result = json.loads((tmp_path / 'output/two_covariates.json').read_text())
    masks = np.load(tmp_path / 'output/local/two_covariates_points.npz')
    draws = sample_within_strata(strata, z, 59, np.random.default_rng(19))
    n = len(frame)
    orders = S.knn_orders(frame[['cd40', 'age']].to_numpy())
    for score, y in [('raw', frame.cd420.to_numpy()), ('change', (frame.cd420 - frame.cd40).to_numpy())]:
        paths = {k: _direct(y, z, draws, [orders[c, :k] for c in range(n)]) for k in range(1, n + 1)}
        bound = max(k for k, p in paths.items() if p.max() > R.ALPHA)
        mask = np.zeros(n, bool)
        for centre in np.flatnonzero(paths[bound] > R.ALPHA):
            mask[orders[centre, :bound]] = True
        assert result[score]['bound'] == result[score]['descending_bound'] == bound
        assert result[score]['evaluated_sizes'] == list(range(bound, n + 1))
        np.testing.assert_array_equal(masks[score], mask)


def test_noy_exact_scores_all_integer_and_legacy_grid():
    rng = np.random.default_rng(81)
    n, M, seed = 42, 59, 17
    z = sample_crd(n, n // 2, 1, rng)[0]
    y = rng.integers(0, 8, n) + 6 * z
    frame = pd.DataFrame(dict(treatment=z, W_exact=[Fraction(k, 6) for k in range(n)],
                              quality_exact=[Fraction(int(v), 3) for v in y]))
    draws = sample_crd(n, int(z.sum()), M, np.random.default_rng(seed))
    paths = {k: _direct(y, z, draws, [np.arange(s, s + k) for s in range(n - k + 1)])
             for k in range(1, n + 1)}
    for mode in ('all', 'paper'):
        rec = R._noy_one(frame, 'quality', M, seed, True, mode)
        candidates = R._candidate_sizes(n, mode)
        expected = max((k for k in candidates if paths[k].max() > R.ALPHA), default=0)
        assert rec['bound'] == expected
        assert rec['grid'] == candidates
        if expected:
            mask = np.zeros(n, bool)
            for start in np.flatnonzero(paths[expected] > R.ALPHA):
                mask[start:start + expected] = True
            np.testing.assert_array_equal(rec['mask'], mask)
