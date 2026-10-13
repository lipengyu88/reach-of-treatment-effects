"""Main uncentered-versus-centered calibration and region displays over all integer sizes.

The separate run_two_regions.py reproduces the centred original-outcome results in the two-region appendix.

For every dimension d, sample size n and repetition r the script draws one dataset (designs.generate),
computes the true size K0 (L0 in one dimension), and compares the uncentered statistic
sum((Z-pi)*Y) with the within-region centered statistic sum((Z-pi)*(Y-Ybar_B)). Both use the same
original outcomes, assignment and Monte Carlo draws; no OLS adjustment enters this comparison.
Each statistic evaluates the largest shared-test p-value at integer sizes from the largest
size downwards, stopping at the first non-rejection, and records the bound, r* and the certificate
r* - 1. M = 500 shared draws, alpha = 0.05.

usage:
  python runners/run_calibration.py --design centre10 [--dims 1 2] [--sizes ...] [--reps 200] [--workers 8]
  python runners/run_calibration.py --design centre10 --display 1 7000 0     (one region display)
  python runners/run_calibration.py --smoke                                   (two small tasks, seconds)
Each task writes one JSON record under <out>/records and is skipped if the record exists; when all tasks
are done the records are collected into <out>/details.csv and <out>/summary.csv. Existing records must
match the run configuration and statistic schema. --grid paper uses the former 5% grid,
but retains the new raw-versus-centered comparison. --reuse-centered-from OLD_OUTPUT reuses the old
schema-2 raw results as the new adjusted results after validating their configuration and seeds.
The old OLS-adjusted results are never reused. Source records remain unchanged.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import re
import sys
import time
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np                                    # noqa: E402

SIZES = [200, 400, 600, 800, 1000, 4000, 7000, 10000, 13000]
M, ALPHA, REPS = 500, 0.05, 200
FRACTION = {'centre10': 0.10}                      # main comparison: K0 / n
DISPLAY = {(1, 7000, 0), (2, 7000, 0)}               # replicates whose region display is stored
SCHEMA_VERSION = 3
RAW_STATISTIC = 'uncentered_assignment_covariance'
ADJUSTED_STATISTIC = 'within_region_centered_covariance'
SCORE_FIELDS = ('bound', 'grid_bound', 'full_bound', 'first_rejected', 'certificate',
                'grid_covers', 'covers', 'integer_covers', 'P')


def run_config(design, grid):
    if design not in FRACTION:
        raise ValueError('The main comparison uses centre10; use run_two_regions.py for the appendix')
    if grid not in ('all', 'paper'):
        raise ValueError(f'Unknown grid {grid!r}')
    return dict(schema_version=SCHEMA_VERSION, design=design, grid_mode=grid, M=M, alpha=ALPHA,
                raw_statistic=RAW_STATISTIC, adjusted_statistic=ADJUSTED_STATISTIC,
                outcome_source='original_y_for_both_statistics',
                search='exact_descending', candidate_sizes='1..n' if grid == 'all' else 'paper_5_percent')


def validate_record(rec, design, grid, task=None):
    expected = run_config(design, grid)
    if task is not None:
        expected.update(zip(('dim', 'n', 'rep'), task))
    for key, value in expected.items():
        if rec.get(key) != value:
            raise ValueError(f'Incompatible existing record: {key}={rec.get(key)!r}, expected {value!r}; '
                             'use a new --out directory')


def validate_output_location(out, reuse_centered_from):
    """Never write inside the archived source, or use an ancestor containing it as the new output."""
    if reuse_centered_from is not None:
        dest, source = Path(out).resolve(), Path(reuse_centered_from).resolve()
        if dest == source or source in dest.parents or dest in source.parents:
            raise ValueError('The new --out must be separate from --reuse-centered-from')


def _parse_path(value):
    """Read historical np.float64(...) strings without eval, then write portable plain floats."""
    path = {}
    for item in value.split(';'):
        key, number = item.split(':', 1)
        match = re.fullmatch(r'np\.float64\(([^()]+)\)', number)
        p = float(match.group(1) if match else number)
        k = int(key)
        if k in path or not np.isfinite(p):
            raise ValueError('Duplicate size or nonfinite p-value in centered source')
        path[k] = p
    return path


def _format_path(path):
    return ';'.join(f'{k}:{float(p)!r}' for k, p in sorted(path.items()))


def load_centered_source(source, task, design, grid, current):
    """Validate a schema-2 centered-raw record and map only its raw fields to adjusted.

    The source predates array fingerprints, so compatibility is established by the original seed
    plan, design, grid, Monte Carlo count, alpha and truth, plus the full recorded descending path.
    Its byte hash and path are retained for an auditable, read-only reuse operation.
    """
    source_path = record_path(Path(source), task)
    contents = source_path.read_bytes()
    old = json.loads(contents)
    expected = {key: value for key, value in current.items()
                if key in ('design', 'grid_mode', 'M', 'alpha', 'search', 'candidate_sizes',
                           'dim', 'n', 'rep', 'K0', 'true_grid_size', 'zero_count') or key.startswith('seed_')}
    expected['schema_version'] = 2
    for key, value in expected.items():
        if old.get(key) != value:
            raise ValueError(f'Incompatible centered source {source_path}: {key}={old.get(key)!r}, '
                             f'expected {value!r}')
    # A schema with explicit statistic labels is not an audited legacy source.
    if 'raw_statistic' in old or 'adjusted_statistic' in old:
        raise ValueError(f'Incompatible centered source statistic schema: {source_path}')
    dim, n, rep = task
    from zeroreach.grids import paper_grid
    sizes = list(range(1, n + 1)) if grid == 'all' else paper_grid(n)
    b = old.get('raw_bound')
    if type(b) is not int or b not in [0] + sizes:
        raise ValueError(f'Invalid centered source bound: {b!r}')
    first = next((k for k in sizes if k > b), None)
    certificate = n if first is None else first - 1
    required = dict(bound=b, grid_bound=b, full_bound=b if grid == 'all' else None,
                    first_rejected=first, certificate=certificate,
                    grid_covers=b >= current['true_grid_size'], covers=b >= current['K0'],
                    integer_covers=certificate >= current['K0'])
    for field, value in required.items():
        if 'raw_' + field not in old or old['raw_' + field] != value:
            raise ValueError(f'Inconsistent centered source raw_{field}: {source_path}')
    path = _parse_path(old['raw_P'])
    if sorted(path) != [k for k in sizes if k >= b]:
        raise ValueError(f'Incomplete descending path in centered source: {source_path}')
    for k, p in path.items():
        count_plus_one = p * (M + 1)
        if (not 1 - 1e-8 <= count_plus_one <= M + 1 + 1e-8
                or abs(count_plus_one - round(count_plus_one)) > 1e-8
                or (k > b and p > ALPHA) or (k == b and p <= ALPHA)):
            raise ValueError(f'Invalid p-value in centered source at size {k}: {source_path}')
    required['P'] = _format_path(path)
    return ({'adjusted_' + key: value for key, value in required.items()},
            dict(path=str(source_path.resolve()), sha256=hashlib.sha256(contents).hexdigest(),
                 source_schema_version=2, source_score='raw', target_score='adjusted',
                 statistic=ADJUSTED_STATISTIC, verified_config=expected))


def input_fingerprint(d):
    """Fingerprint the shared experiment inputs; OLS data are intentionally excluded."""
    digest = hashlib.sha256()
    for name in ('W', 'y', 'z', 'draws'):
        arr = np.ascontiguousarray(d[name])
        digest.update(name.encode())
        digest.update(str(arr.dtype).encode())
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def run_task(task, design, out, display=False, grid='all', reuse_centered_from=None):
    from zeroreach import designs as D
    from zeroreach import search as S
    from zeroreach.grids import grid_scan
    validate_output_location(out, reuse_centered_from)
    config = run_config(design, grid)
    dim, n, rep = task
    d = D.generate(dim, n, rep, zero=design, M=M)
    orders = S.knn_orders(d['W']) if dim == 2 else None
    K0 = D.true_size(d, orders)
    if K0 != int(np.floor(FRACTION[design] * n)):
        raise ValueError(f'K0 = {K0} differs from {FRACTION[design]} n for d={dim} n={n} rep={rep}')
    sizes = d['grid'] if grid == 'paper' else range(1, n + 1)
    true_grid = max([0] + [k for k in sizes if k <= K0])
    rec = dict(**config, dim=dim, n=n, rep=rep, **{'seed_' + k: v for k, v in d['seeds'].items()},
               K0=K0, true_grid_size=true_grid, zero_count=int(d['zmask'].sum()))
    rec.update(input_sha256=input_fingerprint(d), adjusted_reused=reuse_centered_from is not None,
               centered_source=None)
    if reuse_centered_from is not None:
        centered, provenance = load_centered_source(reuse_centered_from, task, design, grid, rec)
        rec.update(centered, centered_source=provenance)
    t0, c0 = time.perf_counter(), time.process_time()
    disp = {}
    for score in ('raw', 'adjusted'):
        reused = score == 'adjusted' and reuse_centered_from is not None
        # Same outcomes and randomizations: the only difference is within-region centering.
        ev = None
        if not reused or display:
            cls = (S.Shared1DRaw if score == 'raw' else S.Shared1D) if dim == 1 else (
                   S.Shared2DRaw if score == 'raw' else S.Shared2D)
            args = (d['y'], d['z'], d['draws']) + ((orders,) if dim == 2 else ())
            ev = cls(*args)
        if reused:
            b, cert = rec['adjusted_bound'], rec['adjusted_certificate']
        else:
            if grid == 'all':
                bound = int(S.descending(ev, n, ALPHA))
                evaluated = sorted(ev.cache)
                res = dict(grid_bound=bound, certificate=bound,
                           first_rejected=bound + 1 if bound < n else None,
                           grid=evaluated, P=[ev.cache[k] for k in evaluated])
            else:
                res = grid_scan(ev, n, ALPHA, sizes)
            b, cert = res['grid_bound'], res['certificate']
            # Compatibility aliases refer to the configured candidate set, never silently to a sparse grid.
            rec.update({f'{score}_bound': b, f'{score}_grid_bound': b,
                        f'{score}_full_bound': b if grid == 'all' else None,
                        f'{score}_first_rejected': res['first_rejected'],
                        f'{score}_certificate': cert, f'{score}_grid_covers': bool(b >= true_grid),
                        f'{score}_covers': bool(b >= K0),
                        f'{score}_integer_covers': bool(cert >= K0),
                        f'{score}_P': _format_path(dict(zip(res['grid'], res['P'])))})
        if display:
            mask = np.zeros(n, bool)
            p = ev.local_p(b) if b else np.empty(0)
            for c in np.flatnonzero(p > ALPHA):
                if dim == 1:
                    mask[c:c + b] = True
                else:
                    mask[orders[c, :b]] = True
            disp.update({f'{score}_mask': mask, f'{score}_bound': b, f'{score}_grid_bound': b,
                         f'{score}_integer_certificate': cert,
                         f'{score}_windows': int(np.sum(p > ALPHA))})
    if display:
        regions = d['regions']
        side = np.full(n, '', dtype='<U6')
        for s, r in regions.items():
            side[r['members']] = s
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f'display_d{dim}_n{n}_rep{rep}.npz'
        np.savez_compressed(path, W=d['W'], true_zero=d['zmask'], region=side, ranks=d['ranks'], n=n, true_size=K0,
                            grid_mode=grid, candidate_sizes=config['candidate_sizes'],
                            schema_version=SCHEMA_VERSION, raw_statistic=RAW_STATISTIC,
                            adjusted_statistic=ADJUSTED_STATISTIC, input_sha256=rec['input_sha256'],
                            display_selection='preselected_rep_0' if rep == 0 else 'explicit_cli_rep',
                            region_names=np.array(list(regions)),
                            centres=np.array([r['centre'] for r in regions.values()]), **disp)
    rec.update(cpu_seconds=time.process_time() - c0, wall_seconds=time.perf_counter() - t0)
    return rec


def record_path(out, task):
    dim, n, rep = task
    return Path(out) / 'records' / f'd{dim}_n{n}_rep{rep:03d}.json'


def run_if_missing(task, design, out, grid='all', reuse_centered_from=None):
    validate_output_location(out, reuse_centered_from)
    path = record_path(out, task)
    if path.exists():
        validate_record(json.loads(path.read_text()), design, grid, task)
        return None
    rec = run_task(task, design, out, display=task in DISPLAY, grid=grid,
                   reuse_centered_from=reuse_centered_from)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(rec) + '\n')
    os.replace(tmp, path)
    return task


def collect(out, design, sizes, dims, reps, grid='all'):
    import pandas as pd
    from zeroreach.utils import wilson, environment, write_json
    rows = [json.loads(p.read_text()) for p in sorted((out / 'records').glob('*.json'))]
    if not rows:
        return
    for rec in rows:
        validate_record(rec, design, grid)
    det = pd.DataFrame(rows).sort_values(['dim', 'n', 'rep']).reset_index(drop=True)
    det.to_csv(out / 'details.csv', index=False)
    summ = []
    for (dim, n), g in det.groupby(['dim', 'n']):
        for score in ('raw', 'adjusted'):
            b = g[f'{score}_bound'].to_numpy() / n
            k = int(g[f'{score}_covers' if grid == 'all' else f'{score}_grid_covers'].sum())
            p, lo, hi = wilson(k, len(g))
            summ.append(dict(dim=dim, n=n, score=score, grid_mode=grid, schema_version=SCHEMA_VERSION,
                             statistic=RAW_STATISTIC if score == 'raw' else ADJUSTED_STATISTIC,
                             R=len(g), mean_bound_over_n=b.mean(),
                             mcse=b.std(ddof=1) / math.sqrt(len(b)) if len(b) > 1 else None, coverage=p,
                             wilson_lo=lo, wilson_hi=hi,
                             covered=k, K0_over_n=float((g.K0 / n).mean()),
                             integer_coverage=float(g[f'{score}_integer_covers'].mean())))
    pd.DataFrame(summ).to_csv(out / 'summary.csv', index=False)
    write_json(out / 'run_meta.json', dict(
        experiment='calibration', design=design, dims=dims, sizes=sizes, repetitions=reps, M=M, alpha=ALPHA,
        run_config=run_config(design, grid),
        grid='all integer sizes 1..n' if grid == 'all' else '5% of n: step max(1, round(0.05 n))',
        compatibility_aliases='*_grid_bound equals *_bound; *_grid_covers uses the configured candidate set',
        scores=['raw: sum((Z-pi)*Y)', 'adjusted: sum((Z-pi)*(Y-Ybar_B))'],
        ols_adjustment=False, centered_reused_records=int(det.adjusted_reused.sum()),
        display_selection='preselected rep=0 in both dimensions at n=7000',
        timing_scope='per-record search and optional display; reused centered search costs excluded',
        seeds='designs.seed_plan(rep, size_index): data 1+rep, assignment 10001+rep, Monte Carlo 20001+1000 rep, '
              'calibration 20260619 + 10^6 size_index + 600000 + rep', environment=environment()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--design', choices=sorted(FRACTION), default='centre10')
    ap.add_argument('--grid', choices=['all', 'paper'], default='all')
    ap.add_argument('--dims', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--sizes', type=int, nargs='+', default=SIZES)
    ap.add_argument('--reps', type=int, default=REPS)
    ap.add_argument('--rep-start', type=int, default=0)
    ap.add_argument('--workers', type=int, help='parallel processes (default: the available CPUs); each '
                    'process uses NUMBA_NUM_THREADS threads (default: CPUs / processes)')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--reuse-centered-from', type=Path,
                    help='legacy schema-2 calibration/<design> output; reuse only its raw centered results')
    ap.add_argument('--display', type=int, nargs=3, metavar=('DIM', 'N', 'REP'))
    ap.add_argument('--smoke', action='store_true', help='two small tasks into results/smoke/')
    a = ap.parse_args()
    from zeroreach.utils import check_blas, cpu_quota
    cpus = cpu_quota() or os.cpu_count() or 1
    a.workers = a.workers or cpus
    os.environ.setdefault('NUMBA_NUM_THREADS', str(max(1, cpus // a.workers)))   # read by every process
    check_blas()
    if a.smoke:
        a.out = a.out or REPO / 'results/smoke/centering_comparison/calibration' / a.grid / a.design
        a.dims, a.sizes, a.reps, a.workers = [1, 2], [200], 1, 1
    out = a.out or REPO / 'results/centering_comparison' / (
        'full_integer' if a.grid == 'all' else 'paper_grid') / 'calibration' / a.design
    validate_output_location(out, a.reuse_centered_from)
    (out / 'records').mkdir(parents=True, exist_ok=True)
    if a.display:
        rec = run_task(tuple(a.display), a.design, out, display=True, grid=a.grid,
                       reuse_centered_from=a.reuse_centered_from)
        print(json.dumps({k: rec[k] for k in ('dim', 'n', 'rep', 'raw_grid_bound', 'adjusted_grid_bound')}))
        return
    tasks = [(d, n, r) for n in sorted(a.sizes, reverse=True) for d in a.dims
             for r in range(a.rep_start, a.rep_start + a.reps)]
    for task in tasks:
        path = record_path(out, task)
        if path.exists():
            validate_record(json.loads(path.read_text()), a.design, a.grid, task)
    tasks = [t for t in tasks if not record_path(out, t).exists()]
    print(f'{len(tasks)} tasks, {a.workers} processes', flush=True)
    done = 0
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for _ in pool.imap_unordered(partial(run_if_missing, design=a.design, out=out, grid=a.grid,
                                                reuse_centered_from=a.reuse_centered_from), tasks, chunksize=1):
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f'{time.strftime("%H:%M:%S")} {done}/{len(tasks)}', flush=True)
    collect(out, a.design, a.sizes, a.dims, a.reps, grid=a.grid)
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
