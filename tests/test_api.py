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
    out = zeroreach.upper_bound(d['W'], d['y'], d['z'], draws=d['draws'])
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
