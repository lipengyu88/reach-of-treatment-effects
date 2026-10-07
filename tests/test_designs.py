"""Simulation designs and randomization draws."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zeroreach import designs as D                                    # noqa: E402
from zeroreach import search as S                                     # noqa: E402
from zeroreach.randomization import (permutations_within_cells, sample_crd,  # noqa: E402
                                     sample_within_strata)

# first 16 hex digits of designs.array_hash for the versions in requirements-frozen.txt (numpy 2.0.2); the
# random streams of numpy.random.Generator can change between numpy versions
HASHES = {
    (1, 200, 0, 'centre10'): dict(W='a2237b7412f644e0', z='27d90fa79f11f5f4', y='d84f464308723864', draws='e33456cfb7ce86dc'),
    (2, 200, 3, 'centre10'): dict(W='d253143f4a946f08', z='105a5a2fa0f5ddc7', y='db6c4e9c84ddf51c', draws='6566a1bebf0ec3ff'),
    (2, 400, 1, 'two_regions'): dict(W='c974fc65e2dcbb4e', z='9c79f3d804411240', y='0008f88f43261299', draws='939e9173ae3a7b1c'),
    (2, 300, 0, 'circle'): dict(W='51c11c148e50544e', z='bd7c4376e0766006', y='09f5710bf123f437', draws='8cbd5432f0bb6acb'),
}


@pytest.mark.parametrize('key', sorted(HASHES))
def test_design_streams(key):
    dim, n, rep, zero = key
    d = D.generate(dim, n, rep, zero=zero, M=50, size_index=0 if zero == 'circle' else None)
    assert {k: D.array_hash(d[k])[:16] for k in HASHES[key]} == HASHES[key]


@pytest.mark.parametrize('dim', [1, 2])
@pytest.mark.parametrize('n', [200, 1000])
def test_centre_region_is_one_candidate_region(dim, n):
    d = D.generate(dim, n, 0)
    orders = S.knn_orders(d['W']) if dim == 2 else None
    assert int(d['zmask'].sum()) == n // 10
    assert D.true_size(d, orders) == n // 10


def test_two_regions_true_size():
    d = D.generate(2, 1000, 0, zero='two_regions')
    assert D.true_size(d) == 200 and int(d['zmask'].sum()) == 300


def test_samplers_keep_counts():
    rng = np.random.default_rng(0)
    draws = sample_crd(50, 20, 30, rng)
    assert draws.shape == (30, 50) and (draws.sum(1) == 20).all()
    strata = np.repeat([0, 1, 2], [10, 15, 25])
    z = np.zeros(50, np.int8)
    z[[0, 1, 10, 11, 12, 30]] = 1
    dz = sample_within_strata(strata, z, 40, rng)
    for s in (0, 1, 2):
        assert (dz[:, strata == s].sum(1) == z[strata == s].sum()).all()
    cells = np.repeat(np.arange(10), 5)
    perms = permutations_within_cells(cells, np.tile([1, 0, 0, 1, 0], 10), 20, 3)
    assert (perms[0] == np.tile([1, 0, 0, 1, 0], 10)).all()
    assert (perms.reshape(21, 10, 5).sum(2) == 2).all()
