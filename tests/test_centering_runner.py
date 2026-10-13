"""Integration checks for the uncentered-versus-centered comparison."""
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runners import run_calibration as C
from zeroreach import designs as D, search as S, summaries as SU
from zeroreach.grids import grid_scan


def legacy_record(task, grid='all'):
    """Build the old centered-raw source independently of the new runner."""
    dim, n, rep = task
    d = D.generate(dim, n, rep, M=C.M)
    orders = S.knn_orders(d['W']) if dim == 2 else None
    ev = S.Shared1D(d['y'], d['z'], d['draws']) if dim == 1 else S.Shared2D(
        d['y'], d['z'], d['draws'], orders)
    K0 = D.true_size(d, orders)
    sizes = list(range(1, n + 1)) if grid == 'all' else d['grid']
    res = grid_scan(ev, n, C.ALPHA, sizes)
    bound, cert = res['grid_bound'], res['certificate']
    config = C.run_config('centre10', grid)
    config = {k: v for k, v in config.items() if k not in (
        'raw_statistic', 'adjusted_statistic', 'outcome_source')}
    config['schema_version'] = 2
    truth = max([0] + [k for k in sizes if k <= K0])
    return dict(**config, dim=dim, n=n, rep=rep, K0=K0, true_grid_size=truth,
                zero_count=int(d['zmask'].sum()), **{'seed_' + k: v for k, v in d['seeds'].items()},
                raw_bound=bound, raw_grid_bound=bound, raw_full_bound=bound if grid == 'all' else None,
                raw_first_rejected=res['first_rejected'], raw_certificate=cert,
                raw_grid_covers=bound >= truth, raw_covers=bound >= K0, raw_integer_covers=cert >= K0,
                raw_P=';'.join(f'{k}:np.float64({float(p)!r})' for k, p in zip(res['grid'], res['P'])),
                # An invalid old OLS branch makes any accidental reuse of it conspicuous.
                adjusted_bound=-123456, adjusted_P='do not reuse OLS')


def put_source(source, task, rec):
    path = C.record_path(source, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec) + '\n')
    return path


def scientific_fields(rec):
    return {k: v for k, v in rec.items() if k not in (
        'cpu_seconds', 'wall_seconds', 'adjusted_reused', 'centered_source')}


def test_reuse_matches_fresh_and_display(tmp_path):
    for dim in (1, 2):
        for grid in ('all', 'paper'):
            task = (dim, 40, 0)
            source = tmp_path / f'source_{dim}_{grid}'
            old = legacy_record(task, grid)
            source_path = put_source(source, task, old)
            before = source_path.read_bytes()
            fresh_out, reuse_out = tmp_path / f'fresh_{dim}_{grid}', tmp_path / f'reuse_{dim}_{grid}'
            fresh = C.run_task(task, 'centre10', fresh_out, display=True, grid=grid)
            reused = C.run_task(task, 'centre10', reuse_out, display=True, grid=grid,
                                reuse_centered_from=source)
            assert scientific_fields(fresh) == scientific_fields(reused)
            assert reused['adjusted_bound'] == old['raw_bound']
            assert not fresh['adjusted_reused'] and reused['adjusted_reused']
            assert reused['centered_source']['sha256'] == hashlib.sha256(before).hexdigest()
            assert reused['centered_source']['verified_config']['seed_monte_carlo'] == 20001
            assert reused['centered_source']['source_score'] == 'raw'
            assert source_path.read_bytes() == before
            assert 'np.float64' not in reused['adjusted_P']
            name = f'display_d{dim}_n40_rep0.npz'
            with np.load(fresh_out / name) as a, np.load(reuse_out / name) as b:
                assert set(a.files) == set(b.files)
                for key in a.files:
                    assert np.array_equal(a[key], b[key]), key
                assert a['schema_version'] == C.SCHEMA_VERSION
                assert str(a['display_selection']) == 'preselected_rep_0'


def test_reuse_skips_centered_search_and_rejects_ols(tmp_path):
    task = (1, 40, 0)
    source = tmp_path / 'source'
    old = legacy_record(task)
    put_source(source, task, old)
    original_descending = S.Shared1D.descending

    def guard(ev, *args, **kwargs):
        if type(ev) is S.Shared1D:
            raise AssertionError('Centered search was rerun')
        return original_descending(ev, *args, **kwargs)

    with patch.object(S.Shared1D, 'descending', guard):
        reused = C.run_task(task, 'centre10', tmp_path / 'new', reuse_centered_from=source)
    assert reused['adjusted_bound'] == old['raw_bound'] != old['adjusted_bound']


def test_source_mismatch_rejected(tmp_path):
    task = (1, 40, 0)
    source, out = tmp_path / 'source', tmp_path / 'new'
    old = legacy_record(task)
    changes = {'schema_version': 3, 'design': 'two_regions', 'grid_mode': 'paper', 'M': 499,
               'alpha': .1, 'seed_data': 99, 'seed_assignment': 99, 'seed_monte_carlo': 99,
               'seed_calibration': 99, 'K0': 3, 'zero_count': 3, 'n': 41, 'rep': 1,
               'raw_certificate': -1, 'raw_P': '40:0.2', 'raw_statistic': 'unrecognized'}
    for key, value in changes.items():
        put_source(source, task, dict(old, **{key: value}))
        try:
            C.run_task(task, 'centre10', out, reuse_centered_from=source)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Accepted incompatible source: {key}')
    put_source(source, task, old)
    for forbidden in (source, source / 'new', tmp_path):
        try:
            C.run_task(task, 'centre10', forbidden, reuse_centered_from=source)
        except ValueError as exc:
            assert 'separate' in str(exc)
        else:
            raise AssertionError('Allowed output to overlap the source')
    try:
        C.validate_record(old, 'centre10', 'all')
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted old schema as a new comparison result')


def test_summary_rejects_partial_and_legacy_and_fixes_rep_zero(tmp_path):
    import pandas as pd
    rows = [dict(C.run_config('centre10', 'all'), dim=dim, n=40, rep=rep,
                 raw_grid_bound=(1 if rep == 0 else 2), adjusted_grid_bound=(3 if rep == 0 else 4))
            for dim in (1, 2) for rep in range(2)]
    folder = tmp_path / 'calibration/centre10'
    folder.mkdir(parents=True)
    details = folder / 'details.csv'
    with patch.object(SU, 'CALIBRATION_RESULTS', tmp_path), patch.object(SU, 'SIZES', [40]):
        pd.DataFrame(rows).to_csv(details, index=False)
        # Both medians are fractional: selection remains defined and independent of the bounds.
        assert SU.display_rep(1, 40, expected_reps=2) == 0
        for bad in (rows[:-1], rows + [rows[0]], [dict(r, schema_version=2) for r in rows]):
            pd.DataFrame(bad).to_csv(details, index=False)
            try:
                SU.calibration_records(expected_reps=2)
            except ValueError:
                pass
            else:
                raise AssertionError('Accepted incomplete, duplicate or legacy plotting data')


if __name__ == '__main__':
    for fn in (test_reuse_matches_fresh_and_display, test_reuse_skips_centered_search_and_rejects_ols,
               test_source_mismatch_rejected, test_summary_rejects_partial_and_legacy_and_fixes_rep_zero):
        with TemporaryDirectory() as tmp:
            fn(Path(tmp))
        print(fn.__name__, 'PASS', flush=True)
