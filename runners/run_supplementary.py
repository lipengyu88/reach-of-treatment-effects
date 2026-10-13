"""Simulations of Section 6.3 (rate of convergence) and Appendix H.

Here raw denotes original outcomes tested with the centred statistic; adjusted denotes centred OLS
residuals. These labels are distinct from the main uncentred-versus-centred calibration comparison.
Experiments (all one-dimensional unless stated; M = 500 shared draws, alpha = 0.05):
  rates                  K0 and the shared descending bound over all integer sizes, raw outcome; central
                         zero-effect region with a constant effect or an effect growing with the distance to
                         the region (one and two covariates), and a strip region (two covariates);
                         n = 500, ..., 16000, 100 repetitions (Section 6.3, Table 5).
  stress                 scenarios S0-S5: integer bounds for the raw, independently adjusted and in-sample
                         adjusted outcomes, descending and bisection bounds over all integer sizes, and the
                         two-sided statistic for S4; n = 200, 1000, 4000, 500 repetitions (Appendix H.1).
  dispersed              dispersed and cancelling effects, integer sizes, centred and two-sided statistics;
                         n = 1000, 4000, 100 repetitions (Appendix H.5).
  two_sided              constant, negative and asymmetric effects outside the central block, integer sizes,
                         centred and two-sided statistics; n = 200, 1000, 4000, 500 repetitions (H.6).
  shared_vs_conditional  shared and conditional tests on the same datasets, integer sizes; n = 1000, 4000,
                         200 repetitions (Appendix H.7).
  effect_size            descending bound over all integer sizes for effects 0.1 to 2 outside the central
                         region, one and two covariates; n = 4000, 200 repetitions (Appendix H.8).

usage:
  python runners/run_supplementary.py --experiment rates [--workers 8] [--reps-max R]
  python runners/run_supplementary.py --experiment stress --smoke
Each task writes one JSON record under results/supplementary/<experiment>/records and is skipped if the
record exists with matching configuration; when all tasks are done they are collected into details.csv.
All-integer outputs use results/supplementary. --grid paper reproduces the original 5% grids (1% for
dispersed) in a separate directory. Use NUMBA_NUM_THREADS=1 with one process per core.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
import json
import hashlib
import multiprocessing as mp
import sys
import time
from functools import partial, lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ALPHA = 0.05
NS_RATE = [500, 1000, 2000, 4000, 8000, 16000]
DELTAS = ['0.1', '0.2', '0.3', '0.5', '1', '2']
COND_SEED = 20260929                                  # base of the per-region streams of conditional tests
SCHEMA_VERSION = 2


def run_config(experiment, grid):
    if grid not in ('all', 'paper'):
        raise ValueError(f'Unknown grid {grid!r}')
    mode = 'all' if experiment in ('rates', 'effect_size') else grid
    return dict(schema_version=SCHEMA_VERSION, experiment=experiment, grid_mode=mode, M=500, alpha=ALPHA,
                search='exact_descending' if mode == 'all' else 'paper_sparse_grid')


def validate_record(rec, experiment, grid):
    for key, value in run_config(experiment, grid).items():
        if rec.get(key) != value:
            raise ValueError(f'Incompatible existing record: {key}={rec.get(key)!r}, expected {value!r}; '
                             'use a new --out directory')


class NumpyPath:
    """Cache prefixes once and retain the original numpy statistic and tolerance exactly.

    Unlike evaluating numpy_grid_pvalues over the complete grid, this supports stopping after the
    largest non-rejected integer size. It makes no monotonicity assumption.
    """
    def __init__(self, y, z, draws, two_sided):
        import numpy as np
        self.n, self.M, self.two_sided = len(y), len(draws), two_sided
        zf, Dy = z.astype(np.float64), draws.astype(np.float64)
        self.Y = np.concatenate(([0.0], np.cumsum(y)))
        self.Zo = np.concatenate(([0.0], np.cumsum(zf)))
        self.ZYo = np.concatenate(([0.0], np.cumsum(zf * y)))
        self.Zd = np.concatenate((np.zeros((self.M, 1)), np.cumsum(Dy, axis=1)), axis=1)
        self.ZYd = np.concatenate((np.zeros((self.M, 1)), np.cumsum(Dy * y[None, :], axis=1)), axis=1)
        self.tol = 1e-9 * (np.abs(y).sum() + 1.0)
        self.cache = {}

    def max_p(self, k):
        import numpy as np
        if k not in self.cache:
            s = np.arange(0, self.n - k + 1)
            sy = self.Y[s + k] - self.Y[s]
            tobs = (self.ZYo[s + k] - self.ZYo[s]) - sy / k * (self.Zo[s + k] - self.Zo[s])
            td = ((self.ZYd[:, s + k] - self.ZYd[:, s])
                  - sy[None, :] / k * (self.Zd[:, s + k] - self.Zd[:, s]))
            if self.two_sided:
                cnt = (np.abs(td) >= np.abs(tobs)[None, :] - self.tol).sum(axis=0)
            else:
                cnt = (td >= tobs[None, :] - self.tol).sum(axis=0)
            self.cache[int(k)] = float(((cnt + 1.0) / (self.M + 1.0)).max())
        return self.cache[k]


def exact_bound(ev, n):
    from zeroreach.search import descending
    b = int(descending(ev, n, ALPHA))
    return b, b + 1 if b < n else None


def _reuse_stress_record(rec, task):
    if rec is None:
        return None
    experiment, dim, variant, n, rep = task
    for key, expected in dict(experiment=experiment, dim=dim, variant=variant, n=n, rep=rep).items():
        if rec.get(key) != expected:
            raise ValueError(f'Mismatched reuse record: {key}')
    if experiment != 'stress' or not all(k in rec for k in
            ('raw_desc_bound', 'raw_desc_covers', 'raw_bis_bound', 'raw_bis_covers', 'raw_bis_violations')):
        raise ValueError('Reuse requires a matching stress record with the original complete integer searches')
    return rec


@lru_cache(maxsize=2)
def load_reuse_records(source):
    """Read archived CSV once per worker, retaining typed identity and full-integer fields."""
    import csv
    source = Path(source)
    csv_path = source if source.is_file() else source / 'details.csv'
    meta = json.loads((csv_path.parent / 'run_meta.json').read_text())
    if meta.get('experiment') != 'stress' or meta.get('M') != 500 or meta.get('alpha') != ALPHA:
        raise ValueError('Stress reuse archive has incompatible experiment, M or alpha')
    records = {}
    integer_keys = ('dim', 'n', 'rep', 'L0', 'raw_desc_bound', 'raw_bis_bound', 'raw_bis_violations')
    boolean_keys = ('raw_desc_covers', 'raw_bis_covers')
    with csv_path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            rec = {key: row[key] for key in ('experiment', 'variant')}
            rec.update({key: int(row[key]) for key in integer_keys})
            for key in boolean_keys:
                if row[key] not in ('True', 'False'):
                    raise ValueError(f'Invalid reuse boolean {key}={row[key]!r}')
                rec[key] = row[key] == 'True'
            key = (rec['experiment'], rec['dim'], rec['variant'], rec['n'], rec['rep'])
            if key in records:
                raise ValueError(f'Duplicate reuse task {key}')
            records[key] = rec
    return records


def tasks(experiment):
    """(experiment, dim, variant, n, rep) for every task of the paper."""
    t = []
    if experiment == 'rates':
        for variant in ('centre10-const', 'centre10-grad1'):
            for dim in (1, 2):
                t += [(experiment, dim, variant, n, r) for n in NS_RATE for r in range(100)]
        t += [(experiment, 2, 'strip20-const', n, r) for n in NS_RATE for r in range(100)]
    elif experiment == 'stress':
        t += [(experiment, 1, s, n, r) for s in ('S0', 'S1', 'S2', 'S3', 'S4', 'S5')
              for n in (200, 1000, 4000) for r in range(500)]
    elif experiment == 'dispersed':
        t += [(experiment, 1, v, n, r) for v in ('dispersed5', 'cancel5') for n in (1000, 4000) for r in range(100)]
    elif experiment == 'two_sided':
        t += [(experiment, 1, v, n, r) for v in ('const', 'neg', 'asym') for n in (200, 1000, 4000) for r in range(500)]
    elif experiment == 'shared_vs_conditional':
        t += [(experiment, 1, 'centre10-const', n, r) for n in (1000, 4000) for r in range(200)]
    elif experiment == 'effect_size':
        t += [(experiment, dim, f'centre10-const-d{d}', 4000, r) for dim in (1, 2) for d in DELTAS for r in range(200)]
    else:
        raise ValueError(experiment)
    return t


def run_task(task, grid='all', reuse_record=None):
    from zeroreach import designs as D
    from zeroreach import search as S
    from zeroreach.grids import envelope_bound, paper_grid
    experiment, dim, variant, n, rep = task
    t0, c0 = time.perf_counter(), time.process_time()
    mode = grid
    rec = dict(**run_config(experiment, mode), dim=dim, variant=variant, n=n, rep=rep)
    reuse_record = _reuse_stress_record(reuse_record, task)
    if experiment in ('rates', 'effect_size'):
        parts = variant.split('-')
        zero, effect = parts[0], parts[1]
        delta = float(parts[2][1:]) if len(parts) == 3 else 1.0
        d = D.generate(dim, n, rep, zero=zero, effect=effect, delta=delta)
        orders = S.knn_orders(d['W']) if dim == 2 else None
        K0 = D.true_size(d, orders)
        ev = S.Shared1D(d['y'], d['z'], d['draws']) if dim == 1 else S.Shared2D(d['y'], d['z'], d['draws'], orders)
        c1 = time.process_time()
        khat = S.descending(ev, n, ALPHA)
        if experiment == 'effect_size':
            rec['delta'] = delta
        rec.update(K0=K0, zero_count=int((d['tau'] == 0).sum()), Khat=int(khat), excess=(khat - K0) / n,
                   covers=bool(khat >= K0), search_cpu=time.process_time() - c1,
                   unresolved=int(ev.unresolved), structural=int(ev.structural), mc_sizes=len(ev.mc_queries))
    elif experiment == 'stress':
        d = D.generate(1, n, rep, scenario=variant)
        L0 = D.true_size(d)
        grid = list(range(1, n + 1)) if mode == 'all' else paper_grid(n)
        L0K = max([k for k in grid if k <= L0], default=0)
        rec.update(L0=L0, L0K=L0K)
        for name, y in (('raw', d['y']), ('indep', d['adj_indep']), ('insample', d['adj_insample'])):
            if name == 'raw' and mode == 'all' and reuse_record is not None:
                b = int(reuse_record['raw_desc_bound'])
                if (reuse_record.get('L0', L0) != L0 or not 0 <= b <= n
                        or bool(reuse_record['raw_desc_covers']) != (b >= L0)):
                    raise ValueError('Inconsistent reused raw descending bound')
                rstar = b + 1 if b < n else None
                ev = None
                rec['raw_desc_reused'] = True
                rec['raw_desc_reuse_sha256'] = hashlib.sha256(
                    json.dumps(reuse_record, sort_keys=True).encode()).hexdigest()
            else:
                ev = S.Shared1D(y, d['z'], d['draws'])
                if mode == 'all':
                    b, rstar = exact_bound(ev, n)
                else:
                    P = {k: float(ev.max_p(k)) for k in grid}
                    b, rstar = envelope_bound(P, grid, ALPHA)
            rec.update({f'{name}_grid_bound': b, f'{name}_rstar': rstar, f'{name}_covers': bool(b >= L0K),
                        f'{name}_bound': b, f'{name}_full_bound': b if mode == 'all' else None,
                        f'{name}_unresolved': int(ev.unresolved) if ev is not None else None})
            if name == 'raw':
                if mode == 'all' and reuse_record is not None:
                    rec.update({k: reuse_record[k] for k in ('raw_desc_bound', 'raw_desc_covers',
                                'raw_bis_bound', 'raw_bis_covers', 'raw_bis_violations')})
                else:
                    kd = b if mode == 'all' else S.descending(ev, n, ALPHA)
                    ev2 = S.Shared1D(y, d['z'], d['draws'])
                    kb, _ = S.bisection(ev2, n, ALPHA)
                    rec.update(raw_desc_bound=int(kd), raw_desc_covers=bool(kd >= L0), raw_bis_bound=int(kb),
                               raw_bis_covers=bool(kb >= L0),
                               raw_bis_violations=len(S.monotonicity_violations(ev2.cache)))
        if variant == 'S4':
            if mode == 'all':
                b2, r2 = exact_bound(NumpyPath(d['y'], d['z'], d['draws'], True), n)
            else:
                b2, r2 = envelope_bound(S.numpy_grid_pvalues(d['y'], d['z'], d['draws'], grid, True), grid, ALPHA)
            rec.update(raw_two_sided_grid_bound=b2, raw_two_sided_bound=b2,
                       raw_two_sided_full_bound=b2 if mode == 'all' else None,
                       raw_two_sided_rstar=r2, raw_two_sided_covers=bool(b2 >= L0K))
    elif experiment == 'dispersed':
        d = D.generate(1, n, rep, zero='none', effect=variant)
        step = max(1, round(0.01 * n))
        grid = list(range(step, n + 1, step))
        if mode == 'all':
            b1, r1 = exact_bound(NumpyPath(d['y'], d['z'], d['draws'], False), n)
            b2, r2 = exact_bound(NumpyPath(d['y'], d['z'], d['draws'], True), n)
        else:
            b1, r1 = envelope_bound(S.numpy_grid_pvalues(d['y'], d['z'], d['draws'], grid, False), grid, ALPHA)
            b2, r2 = envelope_bound(S.numpy_grid_pvalues(d['y'], d['z'], d['draws'], grid, True), grid, ALPHA)
        rec.update(L0=D.true_size(d), zero_fraction=float((d['tau'] == 0).mean()), one_sided_bound=b1,
                   one_sided_rstar=r1, two_sided_bound=b2, two_sided_rstar=r2)
    elif experiment == 'two_sided':
        d = D.generate(1, n, rep, zero='centre10', effect=variant)
        L0 = D.true_size(d)
        grid = list(range(1, n + 1)) if mode == 'all' else paper_grid(n)
        L0K = max([k for k in grid if k <= L0], default=0)
        if mode == 'all':
            b1, r1 = exact_bound(NumpyPath(d['y'], d['z'], d['draws'], False), n)
            b2, r2 = exact_bound(NumpyPath(d['y'], d['z'], d['draws'], True), n)
        else:
            b1, r1 = envelope_bound(S.numpy_grid_pvalues(d['y'], d['z'], d['draws'], grid, False), grid, ALPHA)
            b2, r2 = envelope_bound(S.numpy_grid_pvalues(d['y'], d['z'], d['draws'], grid, True), grid, ALPHA)
        rec.update(L0=L0, L0K=L0K, one_sided_bound=b1, one_sided_rstar=r1, one_sided_covers=bool(b1 >= L0K),
                   two_sided_bound=b2, two_sided_rstar=r2, two_sided_covers=bool(b2 >= L0K))
    elif experiment == 'shared_vs_conditional':
        d = D.generate(1, n, rep)
        L0 = D.true_size(d)
        grid = list(range(1, n + 1)) if mode == 'all' else paper_grid(n)
        L0K = max([k for k in grid if k <= L0], default=0)
        rec.update(L0=L0, L0K=L0K)
        for name, ev in (('shared', S.Shared1D(d['y'], d['z'], d['draws'])),
                         ('conditional', S.Cond1D(d['y'], d['z'], len(d['draws']), COND_SEED + 1000 * rep))):
            c1 = time.process_time()
            if mode == 'all':
                b, rstar = exact_bound(ev, n)
                P = ev.cache
            else:
                P = {k: float(ev.max_p(k)) for k in sorted(grid, reverse=True)}
                b, rstar = envelope_bound(P, grid, ALPHA)
            rec.update({f'{name}_grid_bound': b, f'{name}_rstar': rstar, f'{name}_covers': bool(b >= L0K),
                        f'{name}_bound': b, f'{name}_full_bound': b if mode == 'all' else None,
                        f'{name}_P': ';'.join(f'{k}:{P[k]!r}' for k in sorted(P)),
                        f'{name}_cpu': time.process_time() - c1, f'{name}_unresolved': int(ev.unresolved)})
    else:
        raise ValueError(experiment)
    if experiment in ('dispersed', 'two_sided'):
        rec.update(one_sided_full_bound=rec['one_sided_bound'] if mode == 'all' else None,
                   two_sided_full_bound=rec['two_sided_bound'] if mode == 'all' else None)
    rec.update({'seed_' + k: v for k, v in d['seeds'].items()})
    rec.update(cpu_seconds=time.process_time() - c0, wall_seconds=time.perf_counter() - t0)
    return rec


def record_path(out, task):
    experiment, dim, variant, n, rep = task
    return out / 'records' / variant / f'd{dim}_n{n}_rep{rep:03d}.json'


def run_if_missing(task, out, grid='all', reuse_from=None):
    path = record_path(out, task)
    if path.exists():
        validate_record(json.loads(path.read_text()), task[0], grid)
        return None
    reuse_record = None
    if reuse_from is not None and task[0] == 'stress':
        reuse_record = load_reuse_records(str(reuse_from)).get(tuple(task))
        if reuse_record is None:
            raise ValueError(f'Requested reuse archive lacks {task}')
    rec = run_task(task, grid=grid, reuse_record=reuse_record)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(rec) + '\n')
    os.replace(tmp, path)
    return task


def collect(out, experiment, grid='all'):
    import pandas as pd
    from zeroreach.utils import environment, write_json
    rows = [json.loads(p.read_text()) for p in sorted((out / 'records').glob('*/*.json'))]
    if not rows:
        return
    for rec in rows:
        validate_record(rec, experiment, grid)
    df = pd.DataFrame(rows).sort_values(['variant', 'dim', 'n', 'rep']).reset_index(drop=True)
    df.to_csv(out / 'details.csv', index=False)
    write_json(out / 'run_meta.json', dict(
        experiment=experiment, M=500, alpha=ALPHA, tasks=len(df), run_config=run_config(experiment, grid),
        compatibility_aliases='*_grid_bound equals *_bound for the configured candidate set',
        seeds='designs.seed_plan(rep, size_index): data 1+rep, assignment 10001+rep, Monte Carlo 20001+1000 rep, '
              'calibration 20260619 + 10^6 size_index + 600000 + rep; conditional tests 20260929 + 1000 rep',
        environment=environment()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--experiment', required=True,
                    choices=['rates', 'stress', 'dispersed', 'two_sided', 'shared_vs_conditional', 'effect_size'])
    ap.add_argument('--grid', choices=['all', 'paper'], default='all',
                    help='all integer sizes (default), or reproduce the original sparse grids')
    ap.add_argument('--reuse-from', type=Path,
                    help='reuse matching archived raw descending/bisection records for stress only')
    ap.add_argument('--workers', type=int, help='parallel processes (default: the available CPUs); each '
                    'process uses NUMBA_NUM_THREADS threads (default: CPUs / processes)')
    ap.add_argument('--reps-max', type=int, help='only repetitions 0..R-1 of every cell')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--smoke', action='store_true', help='repetition 0 of the smallest n of every cell')
    a = ap.parse_args()
    from zeroreach.utils import check_blas, cpu_quota
    cpus = cpu_quota() or os.cpu_count() or 1
    a.workers = a.workers or cpus
    os.environ.setdefault('NUMBA_NUM_THREADS', str(max(1, cpus // a.workers)))   # read by every process
    check_blas()
    todo = tasks(a.experiment)
    if a.smoke:
        nmin = min(t[3] for t in todo)
        todo = [t for t in todo if t[4] == 0 and t[3] == nmin]
        a.out = a.out or REPO / 'results/smoke/supplementary' / a.grid / a.experiment
    if a.reps_max is not None:
        todo = [t for t in todo if t[4] < a.reps_max]
    output_root = REPO / 'results' if a.grid == 'all' else REPO / 'results/paper_grid_reproduction'
    out = a.out or output_root / 'supplementary' / a.experiment
    (out / 'records').mkdir(parents=True, exist_ok=True)
    for task in todo:
        path = record_path(out, task)
        if path.exists():
            validate_record(json.loads(path.read_text()), a.experiment, a.grid)
    todo = [t for t in todo if not record_path(out, t).exists()]
    todo.sort(key=lambda t: -t[3] * t[3] * t[1])        # largest tasks first
    print(f'{len(todo)} tasks, {a.workers} processes', flush=True)
    done = 0
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for _ in pool.imap_unordered(partial(run_if_missing, out=out, grid=a.grid, reuse_from=a.reuse_from), todo, chunksize=1):
            done += 1
            if done % 25 == 0 or done == len(todo):
                print(f'{time.strftime("%H:%M:%S")} {done}/{len(todo)}', flush=True)
    collect(out, a.experiment, grid=a.grid)
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
