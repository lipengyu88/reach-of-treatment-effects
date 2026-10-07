"""Computation times of the methods of Table 1 (Section 6.4) and of the Gaussian warm start as the number
of draws grows (Appendix H.4).

methods    conditional descending search, shared descending search, shared bisection and shared bisection
           with the Gaussian warm start, on the design of Section 6.4 (raw outcome, M = 500), one and two
           covariates, n = 1000, 5000, 10000, 15000, 20000. Shared methods are timed on ten simulated
           datasets (the figure shows the median); the conditional method on the first dataset only, three
           times at n = 1000 (the figure shows the median) and once for larger n (one run takes up to several
           CPU days at n = 20000 and checkpoints every ten minutes).
warm_start one covariate, n = 5000 and 20000, M = 500, 2000, 10000, 50000, bisection with and without the
           warm start, ten datasets.
Timed: two-dimensional neighbour orders, shared draws, cumulative sums and the search. Not timed: data
generation and numba compilation (warmed up first). Times depend on the machine; the records state the
processor and the number of threads (NUMBA_NUM_THREADS).

usage:
  python runners/run_timing.py --experiment methods --dimension 1 --n 5000 --method shared_descending [--reps 10]
  python runners/run_timing.py --experiment warm_start
  python runners/run_timing.py --smoke
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np                                    # noqa: E402

from zeroreach import designs as D                    # noqa: E402
from zeroreach import search as S                     # noqa: E402
from zeroreach.randomization import sample_crd       # noqa: E402

METHODS = ('conditional_descending', 'shared_descending', 'shared_bisection', 'shared_gaussian_bisection')
SIZES = (1000, 5000, 10000, 15000, 20000)
COND_SEED = 20260929


def _cond_descending_ckpt(ev, n, alpha, ckpt, clock):
    """Descending search with a checkpoint every ten minutes (conditional method only)."""
    start = n
    if ckpt is not None and ckpt.exists():
        state = json.loads(ckpt.read_text())
        ev.cache.update({int(k): v for k, v in state['cache'].items()})
        ev.mc_queries.extend(int(k) for k in state['cache'])
        ev.unresolved, ev.structural = state['unresolved'], state['structural']
        clock['prior_cpu'], clock['prior_wall'] = state['cpu'], state['wall']
        clock['segments'] = state.get('segments', 1) + 1
        start = min(int(k) for k in state['cache']) - 1
        for k in sorted(ev.cache, reverse=True):
            if ev.cache[k] > alpha:
                return k
    last = time.perf_counter()
    for k in range(start, 0, -1):
        if ev.max_p(k) > alpha:
            return k
        if ckpt is not None and time.perf_counter() - last > 600:
            state = dict(cache={str(q): v for q, v in ev.cache.items()}, unresolved=ev.unresolved,
                         structural=ev.structural, last_k=k,
                         cpu=clock['prior_cpu'] + time.process_time() - clock['c0'],
                         wall=clock['prior_wall'] + time.perf_counter() - clock['t0'], segments=clock['segments'])
            tmp = ckpt.with_suffix('.tmp')
            tmp.write_text(json.dumps(state))
            tmp.replace(ckpt)
            last = time.perf_counter()
    return 0


def run_method(dimension, method, W, y, z, M, mc_seed, alpha=0.05, ckpt=None, clock=None):
    n = len(y)
    info = {}
    if dimension == 1:
        order = np.lexsort((np.arange(n), W))
        y, z = y[order], z[order]
        orders = None
    else:
        orders = S.knn_orders(W)
    if method == 'conditional_descending':
        ev = S.Cond1D(y, z, M, COND_SEED) if dimension == 1 else S.Cond2D(y, z, orders, M, COND_SEED)
        bound = _cond_descending_ckpt(ev, n, alpha, ckpt, clock)
    else:
        draws = sample_crd(n, int(z.sum()), M, np.random.default_rng(mc_seed))
        if dimension == 1:
            draws = draws[:, order]
        ev = S.Shared1D(y, z, draws) if dimension == 1 else S.Shared2D(y, z, draws, orders)
        if method == 'shared_descending':
            bound = S.descending(ev, n, alpha)
        else:
            bound, info = S.bisection(ev, n, alpha, warm=(method == 'shared_gaussian_bisection'))
    return bound, ev, info


def one(dimension, n, method, M, design, mc_seed, ckpt=None):
    W, y, z = design['W'], design['y'].astype(np.float64), design['z'].astype(np.int8)
    c0, t0 = time.process_time(), time.perf_counter()
    clock = dict(c0=c0, t0=t0, prior_cpu=0.0, prior_wall=0.0, segments=1)
    bound, ev, info = run_method(dimension, method, W, y, z, M, mc_seed, ckpt=ckpt, clock=clock)
    rec = dict(dimension=dimension, n=n, method=method, M=M, bound=int(bound),
               wall_seconds=clock['prior_wall'] + time.perf_counter() - t0,
               cpu_seconds=clock['prior_cpu'] + time.process_time() - c0,
               mc_queries=len(ev.mc_queries), gaussian_queries=len(ev.gauss_queries),
               unresolved_near_ties=int(ev.unresolved), structural_ties=int(ev.structural),
               monotonicity_violations_among_queried=(len(S.monotonicity_violations(ev.cache))
                                                      if 'bisection' in method else None),
               warm_bracket=info.get('bracket'), warm_accepted=info.get('accepted'), segments=clock['segments'])
    if hasattr(ev, 'requested'):
        rec.update(cond2d_regions_requested=ev.requested, cond2d_regions_computed=ev.computed)
    return rec


def warmup(dimension, method):
    d = D.generate(dimension, 60, 0, zero='circle', M=20, size_index=0)
    one(dimension, 60, method, 20, d, D.seed_plan(0, 0)['monte_carlo'])


def _stamp(rec, rep):
    import numba
    from zeroreach.utils import cpu_model
    rec.update(data_rep=rep, cpu_model=cpu_model(), numba_threads=numba.get_num_threads())
    return rec


def _write(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(rec) + '\n')
    tmp.replace(path)


def methods(dimension, n, method, M, reps, out, repeat=1):
    """Time `method` on datasets 0..reps-1, `repeat` times each."""
    warmup(dimension, method)
    for r in range(reps):
        design = None
        for j in range(repeat):
            path = out / 'records' / f'd{dimension}_n{n}_{method}_rep{r}_run{j}.json'
            if path.exists():
                continue
            if design is None:
                design = D.generate(dimension, n, r, zero='circle', M=1, size_index=0)
            ckpt = path.with_name(path.stem + '.ckpt.json') if method == 'conditional_descending' else None
            rec = _stamp(one(dimension, n, method, M, design, D.seed_plan(r, 0)['monte_carlo'], ckpt=ckpt), r)
            rec['run'] = j
            _write(path, rec)
            print(json.dumps({k: rec[k] for k in ('dimension', 'n', 'method', 'data_rep', 'run', 'bound',
                                                   'cpu_seconds')}), flush=True)


def warm_start(out, sizes=(5000, 20000), Ms=(500, 2000, 10000, 50000), reps=10):
    meths = ('shared_bisection', 'shared_gaussian_bisection')
    for m in meths:
        warmup(1, m)
    for M in Ms:
        for n in sizes:
            for r in range(reps):
                design = None
                for method in meths:
                    path = out / 'records' / f'd1_n{n}_M{M}_{method}_rep{r}.json'
                    if path.exists():
                        continue
                    if design is None:
                        design = D.generate(1, n, r, zero='circle', M=1, size_index=0)
                    rec = _stamp(one(1, n, method, M, design, D.seed_plan(r, 0)['monte_carlo']), r)
                    _write(path, rec)
                    print(json.dumps({k: rec[k] for k in ('n', 'M', 'method', 'data_rep', 'bound', 'cpu_seconds',
                                                           'warm_accepted')}), flush=True)


def collect(out, experiment):
    import pandas as pd
    from zeroreach.utils import environment, write_json
    rows = [json.loads(p.read_text()) for p in sorted((out / 'records').glob('*.json')) if '.ckpt' not in p.name]
    df = pd.DataFrame(rows)
    for col in ('warm_bracket',):
        if col in df:
            df[col] = df[col].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else json.dumps(v))
    df.to_csv(out / 'details.csv', index=False)
    write_json(out / 'run_meta.json', dict(experiment=experiment, alpha=0.05, design="designs.generate(zero='circle', "
               "size_index=0), raw outcome; data repetition r uses seed_plan(r, 0)", environment=environment()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--experiment', choices=['methods', 'warm_start'], default='methods')
    ap.add_argument('--dimension', type=int, choices=[1, 2])
    ap.add_argument('--n', type=int)
    ap.add_argument('--method', choices=METHODS)
    ap.add_argument('--M', type=int, default=500)
    ap.add_argument('--reps', type=int, help='datasets (default 10 for shared methods, 1 for the conditional one)')
    ap.add_argument('--repeat', type=int, default=1, help='timing runs per dataset (3 for the conditional method at n = 1000)')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--collect', action='store_true', help='only collect existing records into details.csv')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    from zeroreach.utils import limit_threads
    limit_threads()
    if a.smoke:
        out = a.out or REPO / 'results/smoke/timing'
        for dim in (1, 2):
            for method in METHODS:
                methods(dim, 300, method, 50, 1, out)
        collect(out, 'methods')
        print('DONE', out)
        return
    out = a.out or REPO / 'results/timing' / a.experiment
    if a.collect:
        collect(out, a.experiment)
        return
    if a.experiment == 'warm_start':
        warm_start(out)
    else:
        if None in (a.dimension, a.n, a.method):
            ap.error('--dimension, --n and --method are required')
        reps = a.reps or (1 if a.method == 'conditional_descending' else 10)
        methods(a.dimension, a.n, a.method, a.M, reps, out, a.repeat)
    collect(out, a.experiment)
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
