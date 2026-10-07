"""Simulations of Section 6.3 (rate of convergence) and Appendix H.

Experiments (all one-dimensional unless stated; M = 500 shared draws, alpha = 0.05):
  rates                  K0 and the shared descending bound over all integer sizes, raw outcome; central
                         zero-effect region with a constant effect or an effect growing with the distance to
                         the region (one and two covariates), and a strip region (two covariates);
                         n = 500, ..., 16000, 100 repetitions (Section 6.3, Table 5).
  stress                 scenarios S0-S5: 5%-grid bounds for the raw, independently adjusted and in-sample
                         adjusted outcomes, descending and bisection bounds over all integer sizes, and the
                         two-sided statistic for S4; n = 200, 1000, 4000, 500 repetitions (Appendix H.1).
  dispersed              dispersed and cancelling effects, 1% grid, centred and two-sided statistics;
                         n = 1000, 4000, 100 repetitions (Appendix H.5).
  two_sided              constant, negative and asymmetric effects outside the central block, 5% grid,
                         centred and two-sided statistics; n = 200, 1000, 4000, 500 repetitions (H.6).
  shared_vs_conditional  shared and conditional tests on the same datasets, 5% grid; n = 1000, 4000,
                         200 repetitions (Appendix H.7).
  effect_size            descending bound over all integer sizes for effects 0.1 to 2 outside the central
                         region, one and two covariates; n = 4000, 200 repetitions (Appendix H.8).

usage:
  python runners/run_supplementary.py --experiment rates [--workers 8] [--reps-max R]
  python runners/run_supplementary.py --experiment stress --smoke
Each task writes one JSON record under results/supplementary/<experiment>/records and is skipped if the
record exists; when all tasks are done they are collected into details.csv. Use NUMBA_NUM_THREADS=1 with
one process per core.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
import json
import multiprocessing as mp
import sys
import time
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ALPHA = 0.05
NS_RATE = [500, 1000, 2000, 4000, 8000, 16000]
DELTAS = ['0.1', '0.2', '0.3', '0.5', '1', '2']
COND_SEED = 20260929                                  # base of the per-region streams of conditional tests


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


def run_task(task):
    from zeroreach import designs as D
    from zeroreach import search as S
    from zeroreach.grids import envelope_bound, paper_grid
    experiment, dim, variant, n, rep = task
    t0, c0 = time.perf_counter(), time.process_time()
    rec = dict(experiment=experiment, dim=dim, variant=variant, n=n, rep=rep)
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
        grid = paper_grid(n)
        L0K = max([k for k in grid if k <= L0], default=0)
        rec.update(L0=L0, L0K=L0K)
        for name, y in (('raw', d['y']), ('indep', d['adj_indep']), ('insample', d['adj_insample'])):
            ev = S.Shared1D(y, d['z'], d['draws'])
            P = {k: float(ev.max_p(k)) for k in grid}
            b, rstar = envelope_bound(P, grid, ALPHA)
            rec.update({f'{name}_grid_bound': b, f'{name}_rstar': rstar, f'{name}_covers': bool(b >= L0K),
                        f'{name}_unresolved': int(ev.unresolved)})
            if name == 'raw':
                kd = S.descending(ev, n, ALPHA)
                ev2 = S.Shared1D(y, d['z'], d['draws'])
                kb, _ = S.bisection(ev2, n, ALPHA)
                rec.update(raw_desc_bound=int(kd), raw_desc_covers=bool(kd >= L0), raw_bis_bound=int(kb),
                           raw_bis_covers=bool(kb >= L0),
                           raw_bis_violations=len(S.monotonicity_violations(ev2.cache)))
        if variant == 'S4':
            b2, r2 = envelope_bound(S.numpy_grid_pvalues(d['y'], d['z'], d['draws'], grid, True), grid, ALPHA)
            rec.update(raw_two_sided_grid_bound=b2, raw_two_sided_rstar=r2, raw_two_sided_covers=bool(b2 >= L0K))
    elif experiment == 'dispersed':
        d = D.generate(1, n, rep, zero='none', effect=variant)
        step = max(1, round(0.01 * n))
        grid = list(range(step, n + 1, step))
        b1, r1 = envelope_bound(S.numpy_grid_pvalues(d['y'], d['z'], d['draws'], grid, False), grid, ALPHA)
        b2, r2 = envelope_bound(S.numpy_grid_pvalues(d['y'], d['z'], d['draws'], grid, True), grid, ALPHA)
        rec.update(L0=D.true_size(d), zero_fraction=float((d['tau'] == 0).mean()), one_sided_bound=b1,
                   one_sided_rstar=r1, two_sided_bound=b2, two_sided_rstar=r2)
    elif experiment == 'two_sided':
        d = D.generate(1, n, rep, zero='centre10', effect=variant)
        L0 = D.true_size(d)
        grid = paper_grid(n)
        L0K = max([k for k in grid if k <= L0], default=0)
        b1, r1 = envelope_bound(S.numpy_grid_pvalues(d['y'], d['z'], d['draws'], grid, False), grid, ALPHA)
        b2, r2 = envelope_bound(S.numpy_grid_pvalues(d['y'], d['z'], d['draws'], grid, True), grid, ALPHA)
        rec.update(L0=L0, L0K=L0K, one_sided_bound=b1, one_sided_rstar=r1, one_sided_covers=bool(b1 >= L0K),
                   two_sided_bound=b2, two_sided_rstar=r2, two_sided_covers=bool(b2 >= L0K))
    elif experiment == 'shared_vs_conditional':
        d = D.generate(1, n, rep)
        L0 = D.true_size(d)
        grid = paper_grid(n)
        L0K = max([k for k in grid if k <= L0], default=0)
        rec.update(L0=L0, L0K=L0K)
        for name, ev in (('shared', S.Shared1D(d['y'], d['z'], d['draws'])),
                         ('conditional', S.Cond1D(d['y'], d['z'], len(d['draws']), COND_SEED + 1000 * rep))):
            c1 = time.process_time()
            P = {k: float(ev.max_p(k)) for k in sorted(grid, reverse=True)}
            b, rstar = envelope_bound(P, grid, ALPHA)
            rec.update({f'{name}_grid_bound': b, f'{name}_rstar': rstar, f'{name}_covers': bool(b >= L0K),
                        f'{name}_P': ';'.join(f'{k}:{P[k]!r}' for k in grid),
                        f'{name}_cpu': time.process_time() - c1, f'{name}_unresolved': int(ev.unresolved)})
    else:
        raise ValueError(experiment)
    rec.update(cpu_seconds=time.process_time() - c0, wall_seconds=time.perf_counter() - t0)
    return rec


def record_path(out, task):
    experiment, dim, variant, n, rep = task
    return out / 'records' / variant / f'd{dim}_n{n}_rep{rep:03d}.json'


def run_if_missing(task, out):
    path = record_path(out, task)
    if path.exists():
        return None
    rec = run_task(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(rec) + '\n')
    os.replace(tmp, path)
    return task


def collect(out, experiment):
    import pandas as pd
    from zeroreach.utils import environment, write_json
    rows = [json.loads(p.read_text()) for p in sorted((out / 'records').glob('*/*.json'))]
    df = pd.DataFrame(rows).sort_values(['variant', 'dim', 'n', 'rep']).reset_index(drop=True)
    df.to_csv(out / 'details.csv', index=False)
    write_json(out / 'run_meta.json', dict(
        experiment=experiment, M=500, alpha=ALPHA, tasks=len(df),
        seeds='designs.seed_plan(rep, size_index): data 1+rep, assignment 10001+rep, Monte Carlo 20001+1000 rep, '
              'calibration 20260619 + 10^6 size_index + 600000 + rep; conditional tests 20260929 + 1000 rep',
        environment=environment()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--experiment', required=True,
                    choices=['rates', 'stress', 'dispersed', 'two_sided', 'shared_vs_conditional', 'effect_size'])
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
        a.out = a.out or REPO / 'results/smoke/supplementary' / a.experiment
    if a.reps_max is not None:
        todo = [t for t in todo if t[4] < a.reps_max]
    out = a.out or REPO / 'results/supplementary' / a.experiment
    (out / 'records').mkdir(parents=True, exist_ok=True)
    todo = [t for t in todo if not record_path(out, t).exists()]
    todo.sort(key=lambda t: -t[3] * t[3] * t[1])        # largest tasks first
    print(f'{len(todo)} tasks, {a.workers} processes', flush=True)
    done = 0
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for _ in pool.imap_unordered(partial(run_if_missing, out=out), todo, chunksize=1):
            done += 1
            if done % 25 == 0 or done == len(todo):
                print(f'{time.strftime("%H:%M:%S")} {done}/{len(todo)}', flush=True)
    collect(out, a.experiment)
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
