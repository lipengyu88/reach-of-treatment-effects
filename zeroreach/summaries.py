"""Summaries of the files in results/, shared by plot_figures.py and make_tables.py."""
from __future__ import annotations

import json
import math
import statistics as st

import numpy as np

from .utils import REPO, wilson

RESULTS = REPO / 'results'
# This comparison has a different statistical meaning from the archived raw/OLS experiment.
# Plot callers may select a completed external run by changing only CALIBRATION_RESULTS.
CALIBRATION_RESULTS = RESULTS / 'centering_comparison/full_integer'
SIZES = [200, 400, 600, 800, 1000, 4000, 7000, 10000, 13000]
NS_RATE = [500, 1000, 2000, 4000, 8000, 16000]
SCEN = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
# Convergence theory: overshoot of order (log n / n)^{1/(2 gamma)}, gamma = nu + beta, with (nu, beta) for each
# design; nu = 3/2 for the strip is the value suggested by the boundary-cap calculation.
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



def complete_integer_records(path, repetitions, expected_groups):
    """Require the completed full-integer series before aggregating paper tables."""
    recs = records(path)
    groups = {}
    for r in recs:
        if r.get('grid_mode') != 'all' or r.get('search') != 'exact_descending':
            raise ValueError(f'{path}: full integer descending-search records are required')
        if r.get('M') != 500 or r.get('alpha') != 0.05:
            raise ValueError(f'{path}: incompatible Monte Carlo configuration')
        key = (r.get('variant', r.get('design')), r['dim'], r['n'])
        groups.setdefault(key, []).append(r['rep'])
    if set(groups) != set(expected_groups) or any(sorted(reps) != list(range(repetitions))
                                                 for reps in groups.values()):
        raise ValueError(f'{path}: missing, duplicated or unexpected repetition identities')
    return recs


def cov(k, n):
    _, lo, hi = wilson(k, n) if n else (None, np.nan, np.nan)
    return dict(k=int(k), R=int(n), rate=(k / n if n else None), wilson=[float(lo), float(hi)])


def slope(ns, ys):
    ns, ys = np.asarray(ns, float), np.asarray(ys, float)
    ok = ys > 0
    if ok.sum() < 2:
        return None
    return float(np.polyfit(np.log(ns[ok]), np.log(ys[ok]), 1)[0])


# Centring comparison and separate two-region diagnostic
def calibration_records(design='centre10', expected_reps=200):
    """Read only explicitly identified uncentered-versus-centered records."""
    import pandas as pd
    from runners.run_calibration import run_config, validate_record
    columns = set(run_config(design, 'all')) | {'dim', 'n', 'rep', 'K0', 'input_sha256'}
    columns.update(f'{score}_{field}' for score in ('raw', 'adjusted')
                   for field in ('bound', 'grid_bound', 'covers', 'grid_covers', 'integer_covers'))
    # Full descending paths remain in separate original run bundles; compact plot inputs omit them.
    frame = pd.read_csv(CALIBRATION_RESULTS / f'calibration/{design}/details.csv',
                        usecols=lambda name: name in columns)
    recs = frame.to_dict('records')
    for rec in recs:
        validate_record(rec, design, 'all')
    actual = [(r['dim'], r['n'], r['rep']) for r in recs]
    expected = {(dim, n, rep) for dim in (1, 2) for n in SIZES for rep in range(expected_reps)}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Calibration figures require one complete record for every dimension, size and '
                         f'repetition 0..{expected_reps - 1}; partial or duplicate results are not publishable')
    return recs


def calibration(design='centre10', expected_reps=200):
    """Mean bound / n and full-integer coverage with Wilson intervals by dimension, n and statistic."""
    recs = calibration_records(design, expected_reps)
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


def display_rep(dim, n=7000, expected_reps=200):
    """Use the preselected replicate 0, independently of either group's resulting bound.

    Exact integer medians need not be attained (nor jointly attained), so matching both medians is
    not a defined selection rule for every experiment. The runner has always stored rep-0 displays.
    """
    if not any(r['dim'] == dim and r['n'] == n and r['rep'] == 0
               for r in calibration_records(expected_reps=expected_reps)):
        raise ValueError(f'Missing preselected display record: dim={dim}, n={n}, rep=0')
    return 0


def two_regions():
    recs = complete_integer_records('calibration/two_regions/details.csv', 200,
                                    [('two_regions', d, n) for d in (1, 2) for n in SIZES])
    # This historical table reports the centred statistic on the original outcome.
    # A revised raw/centred comparison uses a different meaning of the raw field.
    if any(r.get('schema_version', 0) == 3 or r.get('raw_statistic') == 'uncentered_assignment_covariance'
           for r in recs):
        raise ValueError('The two-region table requires the historical centred-raw results, '
                         'not the uncentred raw branch of the revised calibration comparison')
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


# Convergence rates
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


# Computational time
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


# Supplementary diagnostics
def stress():
    recs = complete_integer_records('supplementary/stress/details.csv', 500,
                                    [(s, 1, n) for s in SCEN for n in (200, 1000, 4000)])
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
    recs = complete_integer_records('supplementary/dispersed/details.csv', 100,
                                    [(s, 1, n) for s in ('dispersed5', 'cancel5') for n in (1000, 4000)])
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
    recs = complete_integer_records('supplementary/two_sided/details.csv', 500,
                                    [(s, 1, n) for s in ('const', 'neg', 'asym') for n in (200, 1000, 4000)])
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
    recs = complete_integer_records('supplementary/shared_vs_conditional/details.csv', 200,
                                    [('centre10-const', 1, n) for n in (1000, 4000)])
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
    result = json.loads((RESULTS / 'real_data' / dataset / name).read_text())
    if dataset == 'ggl':
        if not result.get('search', '').startswith('all integer sizes'):
            raise ValueError('GGL presentation requires the completed full-integer search')
        rows = [result]
    else:
        if result.get('grid_mode') != 'all':
            raise ValueError(f'{dataset}: full-integer results required')
        rows = list(result['outcomes'].values()) if 'outcomes' in result else [result[s] for s in ('raw', 'change')]
        for row in rows:
            n = result.get('n', row.get('n'))
            if result.get('grid', row.get('grid')) != list(range(1, n + 1)):
                raise ValueError(f'{dataset}: incomplete integer candidate domain')
    for row in rows:
        if row['bound'] != row['certificate'] or row['first_rejected'] != row['bound'] + 1:
            raise ValueError(f'{dataset}: inconsistent bound/certificate summary')
    return result
