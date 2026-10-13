"""upper_bound() gives the same bound as the step-by-step procedure."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zeroreach                                          # noqa: E402
from zeroreach import designs as D                        # noqa: E402
from zeroreach import search as S                         # noqa: E402
from zeroreach.grids import envelope_bound                # noqa: E402


def test_upper_bound_one_covariate():
    d = D.generate(1, 400, 0, M=500)
    out = zeroreach.upper_bound(d['W'], d['y'], d['z'], draws=d['draws'], grid='paper')
    ev = S.Shared1D(d['y'], d['z'], d['draws'])
    P = {k: ev.max_p(k) for k in d['grid']}
    assert (out['bound'], out['first_rejected']) == envelope_bound(P, d['grid'], 0.05)
    assert out['bound'] >= D.true_size(d) - 20 and out['region'].sum() >= out['bound']


def test_upper_bound_two_covariates_and_unsorted_input():
    d = D.generate(2, 300, 0, M=300)
    perm = np.random.default_rng(0).permutation(300)
    a = zeroreach.upper_bound(d['W'], d['y'], d['z'], draws=d['draws'], grid='all')
    b = zeroreach.upper_bound(d['W'][perm], d['y'][perm], d['z'][perm], draws=d['draws'][:, perm], grid='all')
    assert a['bound'] == b['bound']
    assert np.array_equal(a['region'][perm], b['region'])


def test_default_is_all_integer_with_integer_certificate():
    d = D.generate(1, 60, 0, M=100)
    out = zeroreach.upper_bound(d['W'], d['y'], d['z'], draws=d['draws'])
    ev = S.Shared1D(d['y'], d['z'], d['draws'])
    path = {k: ev.max_p(k) for k in range(1, 61)}
    bound, first = envelope_bound(path, range(1, 61), .05)
    assert out['bound'] == out['certificate'] == bound
    assert out['first_rejected'] == first
    assert out['candidate_sizes'] == list(range(1, 61))
    assert out['grid_mode'] == 'all'
    assert out['statistic'] == 'within_region_centered_covariance'


def test_explicit_sparse_candidates_remain_supported():
    d = D.generate(1, 60, 0, M=100)
    sizes = [3, 17, 39, 60]
    out = zeroreach.upper_bound(d['W'], d['y'], d['z'], draws=d['draws'], grid=sizes)
    ev = S.Shared1D(d['y'], d['z'], d['draws'])
    path = {k: ev.max_p(k) for k in sizes}
    assert (out['bound'], out['first_rejected']) == envelope_bound(path, sizes, .05)
    assert out['candidate_sizes'] == sizes and out['grid_mode'] == 'explicit'
    assert out['certificate'] >= out['bound']


def test_invalid_candidate_sizes_rejected():
    import pytest
    d = D.generate(1, 20, 0, M=10)
    for grid in ('unknown', [0, 2], [1, 21], [1.5, 20]):
        with pytest.raises(ValueError):
            zeroreach.upper_bound(d['W'], d['y'], d['z'], draws=d['draws'], grid=grid)
