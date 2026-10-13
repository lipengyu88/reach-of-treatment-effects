"""Scientific equivalence and gap detection for the sharded integer GGL scan."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zeroreach.realdata import int_window_counts
from zeroreach.utils import write_json

_path = Path(__file__).resolve().parents[1] / 'runners' / 'run_ggl_integer.py'
_spec = importlib.util.spec_from_file_location('ggl_integer', _path)
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)


def reference_counts(y, z, draws, sizes):
    n = len(y)
    out = np.zeros((len(sizes), n), dtype=np.int64)
    # Independent direct scalar calculation; no cumulative sums or production kernel.
    for j, r in enumerate(sizes):
        for s in range(n - r + 1):
            yy, zz = y[s:s+r], z[s:s+r]
            observed = r * int(np.dot(yy, zz)) - int(yy.sum()) * int(zz.sum())
            for row in draws[:, s:s+r]:
                simulated = r * int(np.dot(yy, row)) - int(yy.sum()) * int(row.sum())
                out[j, s] += simulated >= observed
    return out


def fixture_arrays():
    rng = np.random.default_rng(31)
    y = np.array([7, 9, 8, 7, 9, 8, 1, 2, 0, 0, 1, 2], np.int64)
    z = np.array([1]*6 + [0]*6, np.int8)
    draws = np.stack([rng.permutation(z) for _ in range(99)])
    return y, z, draws


def test_integer_kernel_equals_direct_numpy():
    y, z, draws = fixture_arrays()
    sizes = np.arange(1, len(y) + 1, dtype=np.int64)
    actual = int_window_counts(y, z, draws, sizes, 2)
    np.testing.assert_array_equal(actual, reference_counts(y, z, draws, sizes))


def test_five_original_tie_rules_and_draw_reordering():
    from runners.run_real_data import _md5_key
    sub = pd.DataFrame(dict(cell_id=[2, 1, 2, 1, 3, 3], hh_id=[7, 5, 9, 6, 1, 2],
                            age_key=[30, 30, 30, 30, 45, 45], members=[2, 1, 1, 2, 3, 2]))
    cell, hh, age = (sub[c].to_numpy() for c in ('cell_id', 'hh_id', 'age_key'))
    old_keys = dict(A_primary=(hh, cell, age), B_reverse=(-hh, -cell, age),
                    C_household_size=(hh, cell, sub.members.to_numpy(), age),
                    D_hash1=(_md5_key(cell, hh, 1), age), E_hash2=(_md5_key(cell, hh, 2), age))
    original_draws = np.random.default_rng(8).integers(0, 2, size=(19, len(sub)), dtype=np.int8)
    primary = np.lexsort(old_keys['A_primary'])
    for rule, keys in old_keys.items():
        order = G._ggl_order(sub, rule)
        np.testing.assert_array_equal(order, np.lexsort(keys))
        reordered = G._reorder_draws(original_draws[:, primary], primary, order)
        np.testing.assert_array_equal(reordered, original_draws[:, order])


def test_shards_equal_all_integer_reference_and_resume(tmp_path):
    y, z, draws = fixture_arrays()
    n = len(y)
    shared = tmp_path / 'shared'
    shared.mkdir()
    for name, arr in dict(y=y, z=z, draws=draws, age=np.arange(n)+30., rate=y/9.).items():
        np.save(shared / (name + '.npy'), arr)
    manifest = dict(n=n, M=len(draws), seed=31, alpha=.05, ties='fixture', source_sha256='fixture',
                    sha256={}, n1=int(z.sum()), cells=1)
    write_json(shared / 'manifest.json', manifest)
    p1, p2 = tmp_path / 'part1', tmp_path / 'part2'
    G.scan(shared, p1, 1, 6, batch=2, nchunk=1)
    G.scan(shared, p2, 7, n, batch=3, nchunk=2)
    old = (p1 / 'checkpoint.json').read_bytes()
    G.scan(shared, p1, 1, 6, batch=3, nchunk=1)
    assert (p1 / 'checkpoint.json').read_bytes() == old
    summary = G.merge(shared, [p2, p1], tmp_path / 'merged')
    counts = reference_counts(y, z, draws, np.arange(1, n+1))
    p = (counts.max(axis=1) + 1.) / (len(draws) + 1.)
    expected = int(np.flatnonzero(p > .05)[-1]) + 1
    assert summary['bound'] == expected
    starts = np.flatnonzero((counts[expected-1, :n-expected+1] + 1.)/(len(draws)+1.) > .05)
    expected_mask = np.zeros(n, bool)
    for s in starts:
        expected_mask[s:s+expected] = True
    np.testing.assert_array_equal(np.load(tmp_path / 'merged/local/covered_mask.npy'), expected_mask)
    assert not (tmp_path / 'merged/covered_mask.npy').exists()
    with pytest.raises(ValueError, match='gap or overlap'):
        G.merge(shared, [p2], tmp_path / 'bad')
    state = json.loads((p2 / 'checkpoint.json').read_text())
    state['evaluated'].pop(0)
    write_json(p2 / 'checkpoint.json', state)
    with pytest.raises(ValueError, match='contiguous descending'):
        G.merge(shared, [p1, p2], tmp_path / 'bad2')
