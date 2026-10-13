"""The three experiments of Sections 1, 6.7 and 6.8.

ggl   Gerber-Green-Larimer social-pressure experiment: households of the Neighbors and Control groups ordered
      by the mean age of their members (ties by cell, then household), the recorded design (permutation
      of the assignment within cells), M = 5000 draws with seed 20260526. This entry point retains only
      the historical 32-length analysis behind --grid paper; use run_ggl_integer.py for the paper result.
      --robustness adds the historical five tie-breaking rules and three seeds.
actg  ACTG 175, arms 0 and 1, ordered by baseline CD4 count (ties by patient id), randomization within the
      three strata, M = 5000 draws with seed 20260520, all integer sizes; raw 20-week CD4 count and its change
      from baseline. --stability repeats it for M in {500, 1000, 2000, 5000} and seeds 20260520-22;
      --two-covariates adds age as a second covariate (Section 6.8).
noy   Noy-Zhang writing experiment: participants of the linear and convex incentive groups ordered by the
      grade of their first task, complete randomization within the analysed sample, M = 5000 draws with
      seed 20260918, all integer sizes; grade of the second task and minus the time spent on it. --stability
      adds three seeds at M = 5000, three at M = 50000 and one at M = 200000 (Appendix F.3).

The data are not included (data/README.md); shared outputs are aggregates only. The default all-integer
destination is results/real_data/<dataset>; --grid paper uses results/paper_grid_reproduction/real_data/<dataset>.
--grid paper reproduces the previous sparse-grid analyses; --grid all (default) uses exact descending
search, with no monotonicity assumption. GGL's full-integer search uses its separate runner.
usage: python runners/run_real_data.py --dataset actg [--data-dir data/raw] [--stability] [--two-covariates]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

from zeroreach import realdata as RD                  # noqa: E402
from zeroreach import search as S                     # noqa: E402
from zeroreach import kernels as K                    # noqa: E402
from zeroreach.display import hist_pieces, mask_runs  # noqa: E402
from zeroreach.grids import ggl_grid, invert_path, paper_grid   # noqa: E402
from zeroreach.randomization import permutations_within_cells, sample_crd, sample_within_strata  # noqa: E402
from zeroreach.utils import environment, write_json   # noqa: E402

ALPHA = 0.05


def diff_in_means(y, z, sel):
    y, z = y[sel], z[sel]
    if (z == 1).sum() < 2 or (z == 0).sum() < 2:
        return dict(n=int(sel.sum()), diff=None, se=None)
    d = y[z == 1].mean() - y[z == 0].mean()
    se = np.sqrt(y[z == 1].var(ddof=1) / (z == 1).sum() + y[z == 0].var(ddof=1) / (z == 0).sum())
    return dict(n=int(sel.sum()), diff=float(d), se=float(se))


def write_pieces(path, W, mask, edges):
    pd.DataFrame(hist_pieces(W, mask, edges), columns=['left', 'width', 'height', 'covered']).to_csv(path, index=False)


# ------------------------------------------------------------------------------------------------- GGL
GGL_M, GGL_SEED = 5000, 20260526
GGL_SEEDS = (20260526, 20260527, 20260528)


def _ggl_scan(sub, zperm, order, grid, nchunk):
    n = len(sub)
    y, zobs = sub.yint.to_numpy()[order], sub.z.to_numpy()[order]
    if not np.array_equal(zperm[0, order], zobs):
        raise ValueError('the observed assignment must be row 0 of the permutations')
    total = RD.int_window_counts(y, zobs, np.ascontiguousarray(zperm[1:, order]), grid, nchunk)
    M = zperm.shape[0] - 1
    P = np.asarray([(total[g, :n - grid[g] + 1].max() + 1.0) / (M + 1.0) for g in range(len(grid))])
    inv = invert_path(grid, P, ALPHA, n)
    bound = inv['grid_bound']
    mask = np.zeros(n, bool)
    windows = 0
    if bound:
        p_at = (total[int(np.flatnonzero(grid == bound)[0]), :n - bound + 1] + 1.0) / (M + 1.0)
        for s in np.flatnonzero(p_at > ALPHA):
            mask[s:s + bound] = True
        windows = int((p_at > ALPHA).sum())
    return dict(P=P, bound=bound, first_rejected=inv['first_rejected'], certificate=inv['certificate'],
                windows=windows, mask=mask)


def _md5_key(cells, hhs, k):
    return np.array([int.from_bytes(hashlib.md5(f'{int(c)}|{int(h)}|ggl_tie_{k}'.encode()).digest()[:8], 'big')
                     for c, h in zip(cells, hhs)], dtype=np.uint64)


def run_ggl(data_dir, out, robustness, nchunk):
    t0 = time.time()
    sub, L = RD.load_ggl(Path(data_dir) / 'ggl' / 'ggl_individuals.dta')
    n = len(sub)
    grid = np.asarray(ggl_grid(n), np.int64)
    cell, hh, age = sub.cell_id.to_numpy(), sub.hh_id.to_numpy(), sub.age_key.to_numpy()
    zperm = permutations_within_cells(cell, sub.z.to_numpy(), GGL_M, GGL_SEED)
    order = np.lexsort((hh, cell, age))                     # age, then cell, then household
    res = _ggl_scan(sub, zperm, order, grid, nchunk)
    W = age[order] / L
    rate0, z0 = (sub.vsum / sub.members).to_numpy(), sub.z.to_numpy()
    rate, z = rate0[order], z0[order]
    itt = {name: dict(n=int(sel.sum()), diff=float(rate[sel & (z == 1)].mean() - rate[sel & (z == 0)].mean()))
           for name, sel in (('inside', res['mask']), ('outside', ~res['mask']))}
    itt['all'] = dict(n=n, diff=float(rate0[z0 == 1].mean() - rate0[z0 == 0].mean()))
    summary = dict(dataset='GGL', covariate='household mean age in 2006 (years)',
                   ties='age, then cell id, then household id', n=n, n1=int(sub.z.sum()),
                   cells=int(sub.cell_id.nunique()), M=GGL_M, seed=GGL_SEED, alpha=ALPHA, grid=grid.tolist(),
                   P=res['P'].tolist(), bound=res['bound'], bound_fraction=res['bound'] / n,
                   first_rejected=res['first_rejected'], certificate=res['certificate'],
                   nonrejected_windows=res['windows'], covered_units=int(res['mask'].sum()),
                   covered_age_range=[float(W[res['mask']].min()), float(W[res['mask']].max())] if res['bound'] else None,
                   turnout_difference=itt, seconds=round(time.time() - t0, 1))
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'summary.json', summary)
    pd.DataFrame(dict(r=grid, P=res['P'])).to_csv(out / 'pvalue_path.csv', index=False)
    write_pieces(out / 'display_pieces.csv', W, res['mask'],
                 np.arange(np.floor(W.min()), np.floor(W.max()) + 2))
    print(json.dumps({k: summary[k] for k in ('n', 'bound', 'first_rejected', 'certificate', 'covered_units')}), flush=True)
    if robustness:
        h1, h2 = _md5_key(cell, hh, 1), _md5_key(cell, hh, 2)
        size = sub.members.to_numpy()
        rules = {'A_primary': (hh, cell, age), 'B_reverse': (-hh, -cell, age), 'C_household_size': (hh, cell, size, age),
                 'D_hash1': (h1, age), 'E_hash2': (h2, age)}
        rows = []
        for seed in GGL_SEEDS:
            zp = zperm if seed == GGL_SEED else permutations_within_cells(cell, sub.z.to_numpy(), GGL_M, seed)
            for rule, keys in rules.items():
                o = np.lexsort(keys)
                r = _ggl_scan(sub, zp, o, grid, nchunk)
                Wm = (age[o] / L)[r['mask']]
                rows.append(dict(seed=seed, tie_rule=rule, bound=r['bound'], bound_fraction=r['bound'] / n,
                                 first_rejected=r['first_rejected'], certificate=r['certificate'], windows=r['windows'],
                                 covered_units=int(r['mask'].sum()),
                                 covered_age_min=float(Wm.min()) if len(Wm) else None,
                                 covered_age_max=float(Wm.max()) if len(Wm) else None))
                print(json.dumps(rows[-1]), flush=True)
        pd.DataFrame(rows).to_csv(out / 'robustness.csv', index=False)


# ------------------------------------------------------------------------------------------------ ACTG
ACTG_M, ACTG_SEED = 5000, 20260520
ACTG_STABILITY = [(M, seed) for M in (500, 1000, 2000, 5000)
                  for seed in (20260520, 20260521, 20260522)]


def _candidate_sizes(n, grid_mode):
    if grid_mode == 'all':
        return list(range(1, n + 1))
    if grid_mode == 'paper':
        return paper_grid(n)
    raise ValueError(f'unknown grid mode: {grid_mode}')


def _search_sizes(ev, n, grid_mode):
    """Invert all candidates exactly; unvisited sizes below the bound cannot alter the maximum."""
    grid = _candidate_sizes(n, grid_mode)
    if grid_mode == 'all':
        bound = int(S.descending(ev, n, ALPHA))
        P = {int(k): float(v) for k, v in sorted(ev.cache.items())}
        first = bound + 1 if bound < n else None
        inv = dict(grid_bound=bound, first_rejected=first, certificate=bound)
    else:
        P = {k: float(ev.max_p(k)) for k in reversed(grid)}
        inv = invert_path(grid, [P[k] for k in grid], ALPHA, n)
    return inv, P


def _write_stability(out, rows):
    """Checkpoint each completed configuration without exposing partially written CSV files."""
    path = out / 'stability.csv'
    tmp = path.with_suffix('.csv.tmp')
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def _actg_one(sub, z, strat, M, seed, primary, grid_mode='all'):
    n = len(sub)
    draws = sample_within_strata(strat, z, M, np.random.default_rng(seed))
    out = {}
    for score, y in (('raw', sub.cd420.to_numpy(float)), ('change', (sub.cd420 - sub.cd40).to_numpy(float))):
        ev = S.Shared1D(y, z, draws)
        inv, P = _search_sizes(ev, n, grid_mode)
        rec = dict(bound=inv['grid_bound'], first_rejected=inv['first_rejected'], certificate=inv['certificate'],
                   P_at_first_rejected=P[inv['first_rejected']] if inv['first_rejected'] else None,
                   search='descending' if grid_mode == 'all' else 'paper_grid',
                   grid_mode=grid_mode, evaluated_sizes=sorted(P),
                   unresolved_near_ties=int(ev.unresolved), P=P)
        if primary:
            b = inv['grid_bound']
            p = ev.local_p(b) if b else np.empty(0)
            mask = np.zeros(n, bool)
            for s in np.flatnonzero(p > ALPHA):
                mask[s:s + b] = True
            rec.update(windows=int((p > ALPHA).sum()), covered_units=int(mask.sum()),
                       covered_cd40_range=[float(sub.cd40[mask].min()), float(sub.cd40[mask].max())] if b else None,
                       inside=diff_in_means(y, z, mask), outside=diff_in_means(y, z, ~mask),
                       overall=diff_in_means(y, z, np.ones(n, bool)), mask=mask)
        out[score] = rec
    return out, draws


def _rank_by(primary, pid):
    order = np.lexsort((pid, primary))
    r = np.empty(len(primary), np.int64)
    r[order] = np.arange(1, len(primary) + 1)
    return r


def run_actg(data_dir, out, stability, two_covariates, grid_mode='all', primary=True, stability_configs=None):
    sub, z, strat = RD.load_actg(Path(data_dir) / 'actg175' / 'actg175.csv')
    n = len(sub)
    out.mkdir(parents=True, exist_ok=True)
    res, draws = None, None
    if primary:
        res, draws = _actg_one(sub, z, strat, ACTG_M, ACTG_SEED, primary=True, grid_mode=grid_mode)
        W = sub.cd40.to_numpy(float)
        summary = dict(dataset='ACTG175', covariate='baseline CD4 count (cd40)', ties='patient id', arms='0 versus 1',
                       design='complete randomization within the three strata, observed treated count per stratum',
                       n=n, n1=int(z.sum()), M=ACTG_M, seed=ACTG_SEED, alpha=ALPHA,
                       grid_mode=grid_mode, grid=_candidate_sizes(n, grid_mode), outcomes={})
        for score, r in res.items():
            write_pieces(out / f'display_pieces_{score}.csv', W, r['mask'], np.arange(0, 1201, 20))
            pd.DataFrame(mask_runs(r['mask']), columns=['start', 'end']).to_csv(out / f'covered_runs_{score}.csv', index=False)
            summary['outcomes'][score] = {k: v for k, v in r.items() if k not in ('mask', 'P')}
            summary['outcomes'][score]['P'] = {str(k): v for k, v in r['P'].items()}
        write_json(out / 'summary.json', summary)
        print(json.dumps({s: {k: r[k] for k in ('bound', 'first_rejected', 'windows', 'covered_units')}
                          for s, r in res.items()}), flush=True)
    if stability:
        rows = []
        for M, seed in (ACTG_STABILITY if stability_configs is None else stability_configs):
            if res is not None and (M, seed) == (ACTG_M, ACTG_SEED):
                r = res
            else:
                r, _ = _actg_one(sub, z, strat, M, seed, primary=False, grid_mode=grid_mode)
            for score in ('raw', 'change'):
                rows.append(dict(M=M, seed=seed, outcome=score, bound=r[score]['bound'], grid_mode=grid_mode,
                                 first_rejected=r[score]['first_rejected'], certificate=r[score]['certificate']))
                print(json.dumps(rows[-1]), flush=True)
            _write_stability(out, rows)
    if two_covariates:
        pid = sub.pidnum.to_numpy()
        g1 = _rank_by(sub.cd40.to_numpy(float), pid)
        g2 = _rank_by(sub.age.to_numpy(float), pid)
        orders = K.knn_orders(g1, g2)
        if draws is None:
            draws = sample_within_strata(strat, z, ACTG_M, np.random.default_rng(ACTG_SEED))
        grid = _candidate_sizes(n, grid_mode)
        two = dict(covariates=['cd40', 'age'], ties='ranks by (covariate, patient id); distance ties by the cd40 rank',
                   n=n, n1=int(z.sum()), M=ACTG_M, seed=ACTG_SEED, alpha=ALPHA, grid_mode=grid_mode, grid=grid)
        masks = {}
        for score, y in (('raw', sub.cd420.to_numpy(float)), ('change', (sub.cd420 - sub.cd40).to_numpy(float))):
            eg = S.Shared2D(y, z, draws, orders)
            inv, P2 = _search_sizes(eg, n, grid_mode)
            b2, r2 = inv['grid_bound'], inv['first_rejected']
            kd = b2
            unresolved = int(eg.unresolved)
            if grid_mode == 'paper':
                # Retain the previous comparison, while all-mode needs only one exact search.
                ed = S.Shared2D(y, z, draws, orders)
                kd = int(S.descending(ed, n, ALPHA))
                unresolved += int(ed.unresolved)
            mask = np.zeros(n, bool)
            nwin = 0
            if b2 > 0:
                p = eg.local_p(b2)
                keep = np.flatnonzero(p > ALPHA)
                nwin = len(keep)
                for c in keep:
                    mask[orders[c, :b2]] = True
            two[score] = dict(bound=int(b2), grid_bound=int(b2), first_rejected=r2, descending_bound=kd,
                              certificate=inv['certificate'], search='descending' if grid_mode == 'all' else 'paper_grid',
                              evaluated_sizes=sorted(P2), P={str(k): P2[k] for k in sorted(P2)},
                              nonrejected_regions=int(nwin), covered_units=int(mask.sum()),
                              covered_cd40_range=[float(sub.cd40[mask].min()), float(sub.cd40[mask].max())] if mask.any() else None,
                              covered_age_range=[float(sub.age[mask].min()), float(sub.age[mask].max())] if mask.any() else None,
                              inside=diff_in_means(y, z, mask), outside=diff_in_means(y, z, ~mask),
                              unresolved_near_ties=unresolved)
            masks[score] = mask
            print(score, json.dumps({k: two[score][k] for k in ('grid_bound', 'first_rejected', 'descending_bound',
                                                                  'nonrejected_regions', 'covered_units')}), flush=True)
            write_json(out / 'two_covariates.json', two)
        # unit-level points for the scatter display stay in the git-ignored folder local/ (never shared)
        (out / 'local').mkdir(exist_ok=True)
        np.savez_compressed(out / 'local' / 'two_covariates_points.npz', ranks=np.column_stack((g1, g2)) / n,
                            raw=masks['raw'], change=masks['change'])


# ------------------------------------------------------------------------------------------------- Noy
NOY_M, NOY_SEED = 5000, 20260918
NOY_STABILITY = [(5000, 20260919), (5000, 20260920), (5000, 20260921),
                 (50000, 20260918), (50000, 20260919), (50000, 20260920), (200000, 20260918)]


def _noy_one(frame, score, M, seed, primary, grid_mode='all'):
    from zeroreach.realdata import scale_exact
    y, scale = scale_exact(frame[f'{score}_exact'])
    z = frame.treatment.to_numpy(dtype=np.int8)
    n = len(frame)
    draws = sample_crd(n, int(z.sum()), M, np.random.default_rng(seed))
    ev = S.Shared1D(y, z, draws)
    inv, P = _search_sizes(ev, n, grid_mode)
    rec = dict(n=n, n1=int(z.sum()), M=M, seed=seed, score_scale=scale, bound=inv['grid_bound'],
               first_rejected=inv['first_rejected'], certificate=inv['certificate'],
               P_at_bound=P.get(inv['grid_bound']), unresolved_near_ties=int(ev.unresolved),
               grid_mode=grid_mode, grid=_candidate_sizes(n, grid_mode),
               search='descending' if grid_mode == 'all' else 'paper_grid', evaluated_sizes=sorted(P))
    if primary:
        b = inv['grid_bound']
        p = ev.local_p(b) if b else np.empty(0)
        mask = np.zeros(n, bool)
        for s in np.flatnonzero(p > ALPHA):
            mask[s:s + b] = True
        W = np.asarray([float(x) for x in frame.W_exact])
        yf = np.asarray([float(x) for x in frame[f'{score}_exact']])
        rec.update(P={str(k): P[k] for k in sorted(P)}, windows=int((p > ALPHA).sum()), covered_units=int(mask.sum()),
                   covered_W_range=[float(W[mask].min()), float(W[mask].max())] if b else None,
                   inside=diff_in_means(yf, z, mask), outside=diff_in_means(yf, z, ~mask),
                   overall=diff_in_means(yf, z, np.ones(n, bool)), mask=mask, W=W)
    return rec


def run_noy(data_dir, out, stability, grid_mode='all', primary=True, stability_configs=None):
    cohorts = RD.load_noy(Path(data_dir) / 'noy')
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(dataset='NoyZhang', covariate='grade of the first task (exact mean of the grades)',
                   ties='participant id', assumption='complete randomization within the analysed sample',
                   alpha=ALPHA, grid_mode=grid_mode, outcomes={})
    if primary:
        for score in ('quality', 'time'):
            r = _noy_one(cohorts[score], score, NOY_M, NOY_SEED, primary=True, grid_mode=grid_mode)
            write_pieces(out / f'display_pieces_{score}.csv', r['W'], r['mask'], np.arange(5, 44, 2) / 6)
            summary['outcomes'][score] = {k: v for k, v in r.items() if k not in ('mask', 'W')}
            write_json(out / 'summary.json', summary)
            print(score, json.dumps({k: r[k] for k in ('n', 'bound', 'first_rejected', 'windows', 'covered_units')}), flush=True)
    if stability:
        rows = []
        for M, seed in (NOY_STABILITY if stability_configs is None else stability_configs):
            for score in ('quality', 'time'):
                r = _noy_one(cohorts[score], score, M, seed, primary=False, grid_mode=grid_mode)
                rows.append({k: r[k] for k in ('M', 'seed', 'n', 'bound', 'first_rejected', 'certificate', 'P_at_bound')}
                            | dict(outcome=score, grid_mode=grid_mode))
                print(json.dumps(rows[-1]), flush=True)
                _write_stability(out, rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', required=True, choices=['ggl', 'actg', 'noy'])
    ap.add_argument('--data-dir', type=Path, default=REPO / 'data' / 'raw')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--grid', choices=['all', 'paper'], default='all',
                    help='all: exact descending over 1,...,n (default); paper: reproduce the previous sparse grid')
    ap.add_argument('--robustness', action='store_true', help='GGL: tie-breaking rules and seeds')
    ap.add_argument('--stability', action='store_true', help='ACTG and Noy: Monte Carlo sizes and seeds')
    ap.add_argument('--two-covariates', action='store_true', help='ACTG: baseline CD4 count and age')
    phase = ap.add_mutually_exclusive_group()
    phase.add_argument('--stability-only', action='store_true', help='skip primary analysis; run stability configurations')
    phase.add_argument('--two-covariates-only', action='store_true', help='ACTG: skip one-dimensional analysis')
    ap.add_argument('--stability-config', action='append', metavar='M:SEED',
                    help='run only these original stability configurations (repeatable; implies --stability)')
    ap.add_argument('--nchunk', type=int, default=16, help='GGL: number of parallel chunks of draws')
    a = ap.parse_args()
    if a.dataset == 'ggl' and a.grid == 'all':
        ap.error('GGL full-integer search requires its dedicated runner; use --grid paper only to reproduce the old grid')
    if a.dataset == 'ggl' and (a.stability_only or a.two_covariates_only or a.stability_config):
        ap.error('phase and stability-configuration options apply only to ACTG/Noy')
    if a.dataset != 'actg' and (a.two_covariates or a.two_covariates_only):
        ap.error('--two-covariates and --two-covariates-only apply only to ACTG')
    configs = None
    if a.stability_config:
        try:
            configs = [tuple(map(int, item.split(':'))) for item in a.stability_config]
        except ValueError:
            ap.error('--stability-config must have the format M:SEED')
        allowed = ACTG_STABILITY if a.dataset == 'actg' else NOY_STABILITY
        if any(item not in allowed for item in configs) or len(set(configs)) != len(configs):
            ap.error('--stability-config must select distinct configurations from the original experiment')
    a.stability = a.stability or a.stability_only or bool(configs)
    a.two_covariates = a.two_covariates or a.two_covariates_only
    if a.two_covariates_only and a.stability:
        ap.error('--two-covariates-only cannot be combined with stability analyses')
    from zeroreach.utils import limit_threads
    limit_threads()
    output_root = REPO / 'results' if a.grid == 'all' else REPO / 'results/paper_grid_reproduction'
    out = a.out or output_root / 'real_data' / a.dataset
    if any((out / name).exists() for name in ('summary.json', 'two_covariates.json', 'stability.csv', 'run_meta.json')):
        ap.error(f'output already contains results: {out}; choose a new --out directory')
    metadata = dict(dataset=a.dataset, grid_mode=a.grid, robustness=a.robustness, stability=a.stability,
                    stability_configs=configs, two_covariates=a.two_covariates,
                    primary=not (a.stability_only or a.two_covariates_only), environment=environment(), status='running')
    write_json(out / 'run_meta.json', metadata)
    if a.dataset == 'ggl':
        run_ggl(a.data_dir, out, a.robustness, a.nchunk)
    elif a.dataset == 'actg':
        run_actg(a.data_dir, out, a.stability, a.two_covariates, a.grid, metadata['primary'], configs)
    else:
        run_noy(a.data_dir, out, a.stability, a.grid, metadata['primary'], configs)
    metadata['status'] = 'complete'
    write_json(out / 'run_meta.json', metadata)
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
