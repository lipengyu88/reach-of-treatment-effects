"""Summaries of the files in results/, shared by plot_figures.py and make_tables.py."""
from __future__ import annotations

import json
import math
import statistics as st

import numpy as np

from .utils import REPO, wilson

RESULTS = REPO / 'results'
SIZES = [200, 400, 600, 800, 1000, 4000, 7000, 10000, 13000]
NS_RATE = [500, 1000, 2000, 4000, 8000, 16000]
SCEN = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
# theory of Section 5: overshoot of order (log n / n)^{1/(2 gamma)}, gamma = nu + beta
THEORY = {'centre10-const': (1.0, 0.0), 'centre10-grad1': (1.0, 1.0), 'strip20-const': (1.5, 0.0)}
RATE_CELLS = [(1, 'centre10-const'), (1, 'centre10-grad1'), (2, 'centre10-const'), (2, 'centre10-grad1'),
              (2, 'strip20-const')]


def read(path):
    import pandas as pd
    return pd.read_csv(RESULTS / path)


def records(path):
    """Rows of a details.csv as dictionaries (missing values as None)."""
    df = read(path)
    return [{k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in row.items()}
            for row in df.to_dict('records')]


def cov(k, n):
    _, lo, hi = wilson(k, n) if n else (None, np.nan, np.nan)
    return dict(k=int(k), R=int(n), rate=(k / n if n else None), wilson=[float(lo), float(hi)])


def slope(ns, ys):
    ns, ys = np.asarray(ns, float), np.asarray(ys, float)
    ok = ys > 0
    if ok.sum() < 2:
        return None
    return float(np.polyfit(np.log(ns[ok]), np.log(ys[ok]), 1)[0])


# ----------------------------------------------------------------------------------- Section 6.2 and H.2
def calibration(design='centre10'):
    """Mean bound / n, coverage of the grid bound with Wilson intervals, by dimension, n and outcome."""
    recs = records(f'calibration/{design}/details.csv')
    rows = []
    for dim in (1, 2):
        for n in SIZES:
            cell = [r for r in recs if r['dim'] == dim and r['n'] == n]
            for score in ('raw', 'adjusted'):
                b = np.array([r[f'{score}_grid_bound'] / n for r in cell])
                k = sum(bool(r[f'{score}_grid_covers']) for r in cell)
                p, lo, hi = wilson(k, len(cell))
                rows.append(dict(dimension=dim, n=n, score=score, R=len(cell), mean_bound=b.mean(),
                                 mcse=b.std(ddof=1) / math.sqrt(len(b)), coverage=p, wilson_lo=lo, wilson_hi=hi,
                                 covered=k, mean_truth=np.mean([r['K0'] / n for r in cell]),
                                 integer_coverage=float(np.mean([bool(r[f'{score}_integer_covers']) for r in cell]))))
    return rows


def display_rep(dim, n=7000):
    """The replicate shown in the region displays: the first one whose raw and adjusted bounds both equal
    the median bound of that outcome over the 200 replicates."""
    recs = sorted((r for r in records('calibration/centre10/details.csv') if r['dim'] == dim and r['n'] == n),
                  key=lambda r: r['rep'])
    med = {s: st.median(r[f'{s}_grid_bound'] for r in recs) for s in ('raw', 'adjusted')}
    return next(r['rep'] for r in recs if all(r[f'{s}_grid_bound'] == med[s] for s in med))


def two_regions():
    recs = records('calibration/two_regions/details.csv')
    res = {}
    for dim in (1, 2):
        for n in SIZES:
            rr = [r for r in recs if r['dim'] == dim and r['n'] == n]
            row = dict(dim=dim, n=n, R=len(rr), K0_over_n=float(np.mean([r['K0'] for r in rr]) / n))
            for s in ('raw', 'adjusted'):
                row[s] = dict(grid=cov(sum(bool(r[f'{s}_grid_covers']) for r in rr), len(rr)),
                              integer=cov(sum(bool(r[f'{s}_integer_covers']) for r in rr), len(rr)),
                              bound_over_n=float(np.mean([r[f'{s}_grid_bound'] for r in rr]) / n))
            res[f'd{dim}/n{n}'] = row
    return res


# -------------------------------------------------------------------------------------------- Section 6.3
def rates():
    recs = records('supplementary/rates/details.csv')
    res = {}
    for dim, variant in RATE_CELLS:
        rows = []
        for n in NS_RATE:
            rr = [r for r in recs if r['dim'] == dim and r['variant'] == variant and r['n'] == n]
            if not rr:
                continue
            ex = np.array([r['excess'] for r in rr])
            # the rate concerns the overshoot (K^U - K0)_+ / n; in the at most alpha of replicates that miss K0
            # the bound can fall far below K0, which would mask the rate in a plain mean
            pos = np.maximum(ex, 0.0)
            rows.append(dict(n=n, R=len(rr), K0_over_n=float(np.mean([r['K0'] for r in rr]) / n),
                             Khat_over_n=float(np.mean([r['Khat'] for r in rr]) / n),
                             excess_mean=float(ex.mean()), excess_median=float(np.median(ex)),
                             excess_pos_mean=float(pos.mean()),
                             excess_pos_se=float(pos.std(ddof=1) / np.sqrt(len(pos))) if len(pos) > 1 else None,
                             excess_min=float(ex.min()), coverage=cov(sum(bool(r['covers']) for r in rr), len(rr)),
                             unresolved=int(sum(r['unresolved'] for r in rr))))
        nu, beta = THEORY[variant]
        gamma = nu + beta
        full = [r for r in rows if r['R'] >= 100]
        s3 = slope([r['n'] for r in full[-3:]], [r['excess_pos_mean'] for r in full[-3:]]) if len(full) >= 3 else None
        res[f'd{dim}/{variant}'] = dict(dim=dim, variant=variant, nu=nu, beta=beta, gamma=gamma,
                                        theory_slope=-1 / (2 * gamma), rows=rows, slope_last3=s3)
    return res


# -------------------------------------------------------------------------------------------- Section 6.4
def timing_table():
    """Median CPU and wall seconds by dimension, n and method (over datasets for the shared methods, over
    the timing runs for the conditional method)."""
    recs = records('timing/methods/details.csv')
    rows = []
    keys = sorted({(r['dimension'], r['n'], r['method']) for r in recs})
    for dim, n, method in keys:
        rr = [r for r in recs if (r['dimension'], r['n'], r['method']) == (dim, n, method)]
        rows.append(dict(dimension=dim, n=n, method=method, reps=len(rr),
                         cpu_seconds=st.median(r['cpu_seconds'] for r in rr),
                         wall_seconds=st.median(r['wall_seconds'] for r in rr),
                         threads=rr[0]['numba_threads'], cpu_model=rr[0]['cpu_model'], bound=rr[0]['bound'],
                         mc_queries=rr[0]['mc_queries'], unresolved=sum(r['unresolved_near_ties'] for r in rr)))
    return rows


def warm_start():
    recs = records('timing/warm_start/details.csv')
    res = {}
    groups = sorted({(r['n'], r['M'], r['method']) for r in recs})
    for n, M, method in groups:
        rr = {r['data_rep']: r for r in recs if (r['n'], r['M'], r['method']) == (n, M, method)}
        base = {r['data_rep']: r for r in recs if (r['n'], r['M'], r['method']) == (n, M, 'shared_bisection')}
        row = dict(n=n, M=M, method=method, R=len(rr),
                   cpu_median=float(np.median([r['cpu_seconds'] for r in rr.values()])),
                   wall_median=float(np.median([r['wall_seconds'] for r in rr.values()])),
                   mc_queries_median=float(np.median([r['mc_queries'] for r in rr.values()])),
                   gaussian_queries_median=float(np.median([r['gaussian_queries'] for r in rr.values()])),
                   bound_mean=float(np.mean([r['bound'] for r in rr.values()])),
                   same_bound_as_bisection=float(np.mean([r['bound'] == base[k]['bound'] for k, r in rr.items()
                                                          if k in base])) if base else None)
        if method == 'shared_gaussian_bisection':
            row['accepted'] = int(sum(bool(r['warm_accepted']) for r in rr.values()))
        res[f'n{n}/M{M}/{method}'] = row
    return res


# --------------------------------------------------------------------------------------------- Appendix H
def stress():
    recs = records('supplementary/stress/details.csv')
    res = {}
    for s in SCEN:
        for n in (200, 1000, 4000):
            rr = [r for r in recs if r['variant'] == s and r['n'] == n]
            if not rr:
                continue
            row = dict(scenario=s, n=n, R=len(rr), L0_mean=float(np.mean([r['L0'] for r in rr])),
                       L0K_mean=float(np.mean([r['L0K'] for r in rr])))
            for m in ('raw', 'indep', 'insample'):
                row[f'{m}_grid'] = dict(coverage=cov(sum(bool(r[f'{m}_covers']) for r in rr), len(rr)),
                                        bound_over_n=float(np.mean([r[f'{m}_grid_bound'] for r in rr]) / n),
                                        at_n=float(np.mean([r[f'{m}_grid_bound'] == n for r in rr])))
            row['raw_desc'] = dict(coverage=cov(sum(bool(r['raw_desc_covers']) for r in rr), len(rr)),
                                   bound_over_n=float(np.mean([r['raw_desc_bound'] for r in rr]) / n))
            row['raw_bis'] = dict(coverage=cov(sum(bool(r['raw_bis_covers']) for r in rr), len(rr)),
                                  bound_over_n=float(np.mean([r['raw_bis_bound'] for r in rr]) / n),
                                  differs_from_desc=float(np.mean([r['raw_bis_bound'] != r['raw_desc_bound'] for r in rr])),
                                  below_desc=float(np.mean([r['raw_bis_bound'] < r['raw_desc_bound'] for r in rr])),
                                  with_violation=float(np.mean([r['raw_bis_violations'] > 0 for r in rr])))
            if s == 'S4':
                row['raw_two_sided_grid'] = dict(
                    coverage=cov(sum(bool(r['raw_two_sided_covers']) for r in rr), len(rr)),
                    bound_over_n=float(np.mean([r['raw_two_sided_grid_bound'] for r in rr]) / n))
            res[f'{s}/n{n}'] = row
    return res


def dispersed():
    recs = records('supplementary/dispersed/details.csv')
    res = {}
    for v in ('dispersed5', 'cancel5'):
        for n in (1000, 4000):
            rr = [r for r in recs if r['variant'] == v and r['n'] == n]
            row = dict(variant=v, n=n, R=len(rr), L0=sorted({int(r['L0']) for r in rr}),
                       zero_fraction=float(np.mean([r['zero_fraction'] for r in rr])))
            for side in ('one_sided', 'two_sided'):
                b = np.array([r[f'{side}_bound'] for r in rr]) / n
                row[side] = dict(mean=float(b.mean()), median=float(np.median(b)), min=float(b.min()),
                                 max=float(b.max()), at_n=float(np.mean(b == 1.0)))
            res[f'{v}/n{n}'] = row
    return res


def two_sided():
    recs = records('supplementary/two_sided/details.csv')
    res = {}
    for v in ('const', 'neg', 'asym'):
        for n in (200, 1000, 4000):
            rr = [r for r in recs if r['variant'] == v and r['n'] == n]
            row = dict(variant=v, n=n, R=len(rr), L0K=rr[0]['L0K'])
            for side in ('one_sided', 'two_sided'):
                row[side] = dict(coverage=cov(sum(bool(r[f'{side}_covers']) for r in rr), len(rr)),
                                 bound_over_n=float(np.mean([r[f'{side}_bound'] for r in rr]) / n),
                                 at_n=float(np.mean([r[f'{side}_bound'] == n for r in rr])))
            res[f'{v}/n{n}'] = row
    return res


def shared_vs_conditional():
    recs = records('supplementary/shared_vs_conditional/details.csv')
    res = {}
    for n in (1000, 4000):
        rr = [r for r in recs if r['n'] == n]
        d = np.array([r['conditional_grid_bound'] - r['shared_grid_bound'] for r in rr])
        res[f'n{n}'] = dict(
            n=n, R=len(rr), identical=float(np.mean(d == 0)), cond_larger=float(np.mean(d > 0)),
            shared_larger=float(np.mean(d < 0)), max_abs_diff_over_n=float(np.abs(d).max() / n),
            shared=dict(coverage=cov(sum(bool(r['shared_covers']) for r in rr), len(rr)),
                        bound_over_n=float(np.mean([r['shared_grid_bound'] for r in rr]) / n),
                        cpu=float(np.median([r['shared_cpu'] for r in rr]))),
            conditional=dict(coverage=cov(sum(bool(r['conditional_covers']) for r in rr), len(rr)),
                             bound_over_n=float(np.mean([r['conditional_grid_bound'] for r in rr]) / n),
                             cpu=float(np.median([r['conditional_cpu'] for r in rr]))))
    return res


def effect_size():
    recs = records('supplementary/effect_size/details.csv')
    res = {}
    for dim in (1, 2):
        for dtag in ('0.1', '0.2', '0.3', '0.5', '1', '2'):
            rr = [r for r in recs if r['dim'] == dim and r['variant'] == f'centre10-const-d{dtag}']
            ex = np.array([r['excess'] for r in rr])
            pos = np.maximum(ex, 0.0)
            res[f'd{dim}/delta{dtag}'] = dict(
                dim=dim, delta=float(dtag), n=4000, R=len(rr), Khat_over_n=float(np.mean([r['Khat'] for r in rr]) / 4000),
                excess_pos_mean=float(pos.mean()), excess_pos_se=float(pos.std(ddof=1) / np.sqrt(len(pos))),
                coverage=cov(sum(bool(r['covers']) for r in rr), len(rr)))
    return res


# ------------------------------------------------------------------------------------------------ real data
def real(dataset, name='summary.json'):
    return json.loads((RESULTS / 'real_data' / dataset / name).read_text())
