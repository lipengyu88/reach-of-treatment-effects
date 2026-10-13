"""Integer-search runners preserve p-values, exact inversion and archived search reuse."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runners import run_calibration as C
from runners import run_supplementary as R
from zeroreach import designs as D, search as S
from zeroreach.grids import envelope_bound


def test_exact_search_keeps_nonmonotone_larger_acceptance():
    class PathEvaluator:
        def __init__(self):
            self.cache = {}

        def max_p(self, k):
            self.cache[k] = {6: .01, 5: .01, 4: .2, 3: .01, 2: .3, 1: 1.0}[k]
            return self.cache[k]
    ev = PathEvaluator()
    assert R.exact_bound(ev, 6) == (4, 5)
    assert list(ev.cache) == [6, 5, 4]


def _check_numpy_path(effect):
    d = D.generate(1, 40, 0, zero='none' if effect in ('dispersed5', 'cancel5') else 'centre10',
                   effect=effect, M=500)
    sizes = list(range(1, 41))
    for two_sided in (False, True):
        expect = S.numpy_grid_pvalues(d['y'], d['z'], d['draws'], sizes, two_sided)
        ev = R.NumpyPath(d['y'], d['z'], d['draws'], two_sided)
        assert {k: ev.max_p(k) for k in sizes} == expect
        assert R.exact_bound(ev, 40) == envelope_bound(expect, sizes, .05)


def test_numpy_path_matches_original_pvalues():
    for effect in ['const', 'asym', 'dispersed5', 'cancel5']:
        _check_numpy_path(effect)


def _check_calibration(dim, tmp_path):
    n = 40
    d = D.generate(dim, n, 0, M=C.M)
    orders = S.knn_orders(d['W']) if dim == 2 else None
    rec = C.run_task((dim, n, 0), 'centre10', tmp_path, display=True)
    with np.load(tmp_path / f'display_d{dim}_n{n}_rep0.npz') as display:
        for name in ('raw', 'adjusted'):
            cls = (S.Shared1DRaw if name == 'raw' else S.Shared1D) if dim == 1 else (
                   S.Shared2DRaw if name == 'raw' else S.Shared2D)
            ev = cls(d['y'], d['z'], d['draws'], *([orders] if dim == 2 else []))
            values = {k: ev.max_p(k) for k in range(n, 0, -1)}
            bound, _ = envelope_bound(values, range(1, n + 1), .05)
            assert rec[name + '_full_bound'] == bound
            assert rec[name + '_covers'] == (bound >= rec['K0'])
            assert rec[name + '_certificate'] == bound
            mask = np.zeros(n, bool)
            pvalues = ev.local_p(bound) if bound else np.array([])
            for c in np.flatnonzero(pvalues > .05):
                if dim == 1:
                    mask[c:c + bound] = True
                else:
                    mask[orders[c, :bound]] = True
            assert np.array_equal(display[name + '_mask'], mask)


def test_calibration_full_path_and_display(tmp_path):
    for dim in (1, 2):
        _check_calibration(dim, tmp_path)


def test_stress_archive_reuse_matches_recomputation():
    archive = Path(__file__).resolve().parents[1] / 'results/supplementary/stress'
    records = R.load_reuse_records(str(archive))
    assert len(records) == 9000
    task = ('stress', 1, 'S0', 200, 0)
    fresh = R.run_task(task)
    reused = R.run_task(task, reuse_record=records[task])
    assert reused['raw_desc_reused']
    for name in ('raw', 'indep', 'insample'):
        assert fresh[name + '_full_bound'] == reused[name + '_full_bound']
        assert fresh[name + '_covers'] == reused[name + '_covers']
    for key in ('raw_desc_bound', 'raw_desc_covers', 'raw_bis_bound', 'raw_bis_covers', 'raw_bis_violations'):
        assert fresh[key] == reused[key] == records[task][key]


def test_conditional_full_path():
    task = ('shared_vs_conditional', 1, 'centre10-const', 30, 0)
    rec = R.run_task(task)
    d = D.generate(1, 30, 0)
    for name, ev in [('shared', S.Shared1D(d['y'], d['z'], d['draws'])),
                     ('conditional', S.Cond1D(d['y'], d['z'], 500, R.COND_SEED))]:
        path = {k: ev.max_p(k) for k in range(30, 0, -1)}
        bound, first = envelope_bound(path, range(1, 31), .05)
        assert rec[name + '_full_bound'] == bound
        assert rec[name + '_rstar'] == first
        assert rec[name + '_covers'] == (bound >= D.true_size(d))


def test_old_records_rejected(tmp_path):
    task = (1, 200, 0)
    path = C.record_path(tmp_path, task)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'design': 'centre10', 'dim': 1, 'n': 200, 'rep': 0}))
    try:
        C.run_if_missing(task, 'centre10', tmp_path)
    except ValueError as exc:
        assert 'Incompatible existing record' in str(exc)
    else:
        raise AssertionError('Old calibration record accepted')
    try:
        R.validate_record({'experiment': 'stress'}, 'stress', 'all')
    except ValueError as exc:
        assert 'Incompatible existing record' in str(exc)
    else:
        raise AssertionError('Old supplementary record accepted')


if __name__ == '__main__':
    from tempfile import TemporaryDirectory
    for fn in (test_exact_search_keeps_nonmonotone_larger_acceptance,
               test_numpy_path_matches_original_pvalues, test_stress_archive_reuse_matches_recomputation,
               test_conditional_full_path):
        fn()
        print(fn.__name__, 'PASS', flush=True)
    for fn in (test_calibration_full_path_and_display, test_old_records_rejected):
        with TemporaryDirectory() as tmp:
            fn(Path(tmp))
        print(fn.__name__, 'PASS', flush=True)
