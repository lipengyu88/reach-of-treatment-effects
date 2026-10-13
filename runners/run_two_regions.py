"""Two disjoint zero-effect regions: centred original outcomes and an independent OLS diagnostic.

This runner reproduces the two-region appendix, whose table uses only the centred original-outcome
branch, labelled raw for compatibility with the retained results. The adjusted branch centres residuals
from a linear model fitted on independent calibration data. It is not the main uncentred-versus-centred
comparison; use run_calibration.py for that comparison.

For every dimension d, sample size n and repetition r the script draws one dataset (designs.generate),
computes the true size K0 (L0 in one dimension), and for the raw outcome and the outcome adjusted with an
independent calibration sample evaluates the largest shared-test p-value at integer sizes from the largest
size downwards, stopping at the first non-rejection, and records the bound, r* and the certificate
r* - 1. M = 500 shared draws, alpha = 0.05.

usage:
  python runners/run_two_regions.py [--dims 1 2] [--sizes ...] [--reps 200] [--workers 8]
  python runners/run_two_regions.py --display 1 7000 0
  python runners/run_two_regions.py --smoke
Each task writes one JSON record under <out>/records and is skipped if the record exists; when all tasks
are done the records are collected into <out>/details.csv and <out>/summary.csv. Existing records must
match the run configuration. --grid paper reproduces the former 5% grid in a separate output directory.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
import json
import math
import multiprocessing as mp
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
FRACTION = {'two_regions': 0.20}                    # largest candidate region, not total zero count
DISPLAY = {(1, 7000, 0), (2, 7000, 0)}               # replicates whose region display is stored
SCHEMA_VERSION = 4
RAW_STATISTIC = 'within_region_centered_covariance'
ADJUSTED_STATISTIC = 'independent_ols_residual_centered_covariance'


def run_config(design, grid):
    if design != 'two_regions':
        raise ValueError('This runner is for two_regions; use run_calibration.py for centre10')
    if grid not in ('all', 'paper'):
        raise ValueError(f'Unknown grid {grid!r}')
    return dict(schema_version=SCHEMA_VERSION, design=design, grid_mode=grid, M=M, alpha=ALPHA,
                raw_statistic=RAW_STATISTIC, adjusted_statistic=ADJUSTED_STATISTIC,
                outcome_source='original_y_and_independent_ols_residuals',
                search='exact_descending', candidate_sizes='1..n' if grid == 'all' else 'paper_5_percent')


def validate_record(rec, design, grid, task=None):
    expected = run_config(design, grid)
    if task is not None:
        expected.update(zip(('dim', 'n', 'rep'), task))
    for key, value in expected.items():
        if rec.get(key) != value:
            raise ValueError(f'Incompatible existing record: {key}={rec.get(key)!r}, expected {value!r}; '
                             'use a new --out directory')


def run_task(task, design, out, display=False, grid='all'):
    from zeroreach import designs as D
    from zeroreach import search as S
    from zeroreach.grids import grid_scan
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
    t0, c0 = time.perf_counter(), time.process_time()
    disp = {}
    for score, y in (('raw', d['y']), ('adjusted', d['adj_indep'])):
        ev = S.Shared1D(y, d['z'], d['draws']) if dim == 1 else S.Shared2D(y, d['z'], d['draws'], orders)
        if grid == 'all':
            bound = int(S.descending(ev, n, ALPHA))
            evaluated = sorted(ev.cache)
            res = dict(grid_bound=bound, certificate=bound,
                       first_rejected=bound + 1 if bound < n else None,
                       grid=evaluated, P=[ev.cache[k] for k in evaluated])
        else:
            res = grid_scan(ev, n, ALPHA, sizes)
        b, cert = res['grid_bound'], res['certificate']
        # *_grid_bound / *_grid_covers remain explicit compatibility aliases. In all mode they refer
        # to the full integer candidate set, never to the former sparse grid.
        rec.update({f'{score}_bound': b, f'{score}_grid_bound': b,
                    f'{score}_full_bound': b if grid == 'all' else None,
                    f'{score}_first_rejected': res['first_rejected'],
                    f'{score}_certificate': cert, f'{score}_grid_covers': bool(b >= true_grid),
                    f'{score}_covers': bool(b >= K0),
                    f'{score}_integer_covers': bool(cert >= K0),
                    f'{score}_P': ';'.join(f'{k}:{float(p)!r}' for k, p in zip(res['grid'], res['P']))})
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
                            adjusted_statistic=ADJUSTED_STATISTIC,
                            region_names=np.array(list(regions)),
                            centres=np.array([r['centre'] for r in regions.values()]), **disp)
    rec.update(cpu_seconds=time.process_time() - c0, wall_seconds=time.perf_counter() - t0)
    return rec


def record_path(out, task):
    dim, n, rep = task
    return Path(out) / 'records' / f'd{dim}_n{n}_rep{rep:03d}.json'


def run_if_missing(task, design, out, grid='all'):
    path = record_path(out, task)
    if path.exists():
        validate_record(json.loads(path.read_text()), design, grid, task)
        return None
    rec = run_task(task, design, out, display=task in DISPLAY, grid=grid)
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
                             statistic=RAW_STATISTIC if score == 'raw' else ADJUSTED_STATISTIC, R=len(g), mean_bound_over_n=b.mean(),
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
        scores=['raw: within-region centred original outcomes',
                'adjusted: within-region centred independent-OLS residuals'],
        reported_score='raw',
        seeds='designs.seed_plan(rep, size_index): data 1+rep, assignment 10001+rep, Monte Carlo 20001+1000 rep, '
              'calibration 20260619 + 10^6 size_index + 600000 + rep', environment=environment()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--design', choices=sorted(FRACTION), default='two_regions')
    ap.add_argument('--grid', choices=['all', 'paper'], default='all')
    ap.add_argument('--dims', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--sizes', type=int, nargs='+', default=SIZES)
    ap.add_argument('--reps', type=int, default=REPS)
    ap.add_argument('--rep-start', type=int, default=0)
    ap.add_argument('--workers', type=int, help='parallel processes (default: the available CPUs); each '
                    'process uses NUMBA_NUM_THREADS threads (default: CPUs / processes)')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--display', type=int, nargs=3, metavar=('DIM', 'N', 'REP'))
    ap.add_argument('--smoke', action='store_true', help='two small tasks into results/smoke/')
    a = ap.parse_args()
    from zeroreach.utils import check_blas, cpu_quota
    cpus = cpu_quota() or os.cpu_count() or 1
    a.workers = a.workers or cpus
    os.environ.setdefault('NUMBA_NUM_THREADS', str(max(1, cpus // a.workers)))   # read by every process
    check_blas()
    if a.smoke:
        a.out = a.out or REPO / 'results/smoke/two_regions' / a.grid
        a.dims, a.sizes, a.reps, a.workers = [1, 2], [200], 1, 1
    output_root = REPO / 'results' if a.grid == 'all' else REPO / 'results/paper_grid_reproduction'
    out = a.out or output_root / 'calibration' / a.design
    (out / 'records').mkdir(parents=True, exist_ok=True)
    if a.display:
        rec = run_task(tuple(a.display), a.design, out, display=True, grid=a.grid)
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
        for _ in pool.imap_unordered(partial(run_if_missing, design=a.design, out=out, grid=a.grid), tasks, chunksize=1):
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f'{time.strftime("%H:%M:%S")} {done}/{len(tasks)}', flush=True)
    collect(out, a.design, a.sizes, a.dims, a.reps, grid=a.grid)
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
