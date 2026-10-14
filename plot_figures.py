"""Regenerate statistical plots from retained results; no randomization experiment is rerun.

usage: python plot_figures.py [--out figures] [--only NAME ...] [--calibration-results DIR]
The boundary-cap schematic is authored separately and kept as figures/knn_boundary_cap.pdf.
The ACTG 175 two-dimensional scatter requires local provider-derived arrays, supplied via
--actg-scatter-input or the default local path. Otherwise the supplied PDF is preserved.
See README.md for the provider-data rerun command and REPRODUCIBILITY.md for validation scope.
"""
import argparse
import math
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np                                     # noqa: E402
import matplotlib                                      # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402
from matplotlib.patches import Circle, Patch  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

from zeroreach import summaries as SU                  # noqa: E402
from zeroreach.style import (AQUA, BLUE, GRID, INK, INK2, NEUTRAL, ORANGE, RC, SYM_K, YELLOW,  # noqa: E402
                             annotate, bound_lines, draw_pieces)

RES = REPO / 'results'
SIZES = SU.SIZES
K0_FRACTION = 0.10
LEG_REGION = Patch(facecolor=BLUE, label='units in non-rejected regions (union)')
LEG_OTHER = Patch(facecolor=NEUTRAL, label='other units')
ACTG_SCATTER_INPUT = None


def pieces(path):
    import pandas as pd
    df = pd.read_csv(path)
    return list(zip(df.left, df.width, df.height, df.covered.astype(bool)))


# ------------------------------------------------------------------------------------------ Figures 2-3
CALIBRATION_RATIO_SIZES = tuple(n for n in SIZES if n <= 7000)
CALIBRATION_COVERAGE_SIZES = tuple(n for n in SIZES if 1000 <= n <= 13000)


def _axes_style(ax, panel):
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xscale('linear')
    ax.minorticks_off()
    if panel == 'ratio':
        ax.set_xticks([200, 1000, 4000, 7000], ['.2', '1', '4', '7'])
        ax.set_xticks([400, 600, 800], minor=True)
        ax.set_xlim(0, 7280)
    elif panel == 'coverage':
        ax.set_xticks([1000, 4000, 7000, 10000, 13000], ['1', '4', '7', '10', '13'])
        ax.set_xlim(520, 13480)
    else:
        raise ValueError('Unknown calibration panel: ' + panel)
    ax.set_xlabel('sample size $n$ (thousands)')


def fig_calibration(out):
    """Figures 2 and 3: panel-specific sample ranges on linear axes.

    Bound ratios show n<=7000; coverage shows 1000<=n<=13000. All nine
    sample sizes remain in the archived results and experiment configuration.
    """
    rows = SU.calibration('centre10')
    lab = {'raw': 'raw (uncentred)', 'adjusted': 'adjusted (centred)'}
    col = {'raw': BLUE, 'adjusted': ORANGE}
    mk = {'raw': 'o', 'adjusted': 's'}
    for dim in (1, 2):
        sel_ratio = [r for r in rows if r['dimension'] == dim and r['n'] in CALIBRATION_RATIO_SIZES]
        sel_coverage = [r for r in rows if r['dimension'] == dim and r['n'] in CALIBRATION_COVERAGE_SIZES]
        fig, ax = plt.subplots(figsize=(3.3, 2.6))
        for s in ('raw', 'adjusted'):
            r = sorted((x for x in sel_ratio if x['score'] == s), key=lambda x: x['n'])
            ax.plot([x['n'] for x in r], [x['mean_bound'] for x in r],
                    marker=mk[s], ms=3.2, color=col[s], label=lab[s])
        sym = 'L' if dim == 1 else 'K'
        ax.axhline(K0_FRACTION, ls='--', color=INK, lw=1.0, label=f'true ${sym}_0/n={K0_FRACTION:.2f}$')
        ax.set_ylabel(r'mean $\widehat L_{n,0}^{\,U}/n$' if dim == 1 else r'mean $\widehat K_{0}^{\,U}/n$')
        top = max(x['mean_bound'] for x in sel_ratio)
        ax.set_ylim(0, max(0.3, math.ceil((top + 0.02) * 20) / 20))
        _axes_style(ax, 'ratio')
        ax.legend(frameon=False, loc='upper right')
        fig.savefig(out / f'fig_d{dim}_calibration_ratio.pdf', metadata={'CreationDate': None, 'ModDate': None})
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(3.3, 2.6))
        # A small fixed offset in points separates the methods without growing with n.
        off = {'raw': -0.65, 'adjusted': 0.65}
        for s in ('raw', 'adjusted'):
            r = sorted((x for x in sel_coverage if x['score'] == s), key=lambda x: x['n'])
            x = np.array([q['n'] for q in r])
            y = np.array([q['coverage'] for q in r])
            err = np.array([[q['coverage'] - q['wilson_lo'] for q in r], [q['wilson_hi'] - q['coverage'] for q in r]])
            transform = ax.transData + ScaledTranslation(off[s] / 72, 0, fig.dpi_scale_trans)
            eb = ax.errorbar(x, y, yerr=err, marker=mk[s], ms=3.2, color=col[s],
                            capsize=1.2, lw=1.4, label=lab[s], transform=transform)
            for art in [eb[0], *eb[1], *eb[2]]:
                art.set_clip_on(False)
        ax.axhline(0.95, ls='--', color=INK, lw=1.0, label='nominal 0.95')
        ax.set_ylim(0.6, 1.0)
        ax.set_ylabel('empirical coverage of $L_0$' if dim == 1 else 'empirical coverage of $K_0$')
        _axes_style(ax, 'coverage')
        ax.legend(frameon=False, loc='lower right')
        fig.savefig(out / f'fig_d{dim}_calibration_coverage.pdf', metadata={'CreationDate': None, 'ModDate': None})
        plt.close(fig)


# ---------------------------------------------------------------------------------------------- Figure 4
def fig_rates(out):
    """Figure 4: mean overshoot against n with the theoretical slopes."""
    rates = SU.rates()
    style = {'centre10-const': (BLUE, 'o', 'central zero region, constant effect'),
             'centre10-grad1': (ORANGE, 's', 'central zero region, effect grows with distance'),
             'strip20-const': (AQUA, '^', 'strip zero region (two covariates), constant effect')}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    for ax, dim in zip(axes, (1, 2)):
        for key, res in rates.items():
            if res['dim'] != dim or not res['rows']:
                continue
            c, mk, lab = style[res['variant']]
            ns = np.array([r['n'] for r in res['rows']])
            ys = np.array([r['excess_pos_mean'] for r in res['rows']])
            se = np.array([r['excess_pos_se'] or 0 for r in res['rows']])
            ax.errorbar(ns, ys, yerr=2 * se, color=c, marker=mk, ms=4, lw=1.3, capsize=2, label=lab)
            j = int(np.flatnonzero(ns == 4000)[0]) if 4000 in ns else len(ns) - 1
            xx = np.array([ns[0], ns[-1]], float)
            ax.plot(xx, ys[j] * (xx / ns[j]) ** res['theory_slope'], color=c, ls=':', lw=1.0)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(SU.NS_RATE)
        ax.set_xticklabels([f'{n / 1000:g}' for n in SU.NS_RATE])
        ax.minorticks_off()
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel('sample size $n$ (thousands, log scale)')
        ax.set_title(f'({"ab"[dim - 1]}) {"one" if dim == 1 else "two"} covariate{"s" if dim == 2 else ""}', loc='left')
    axes[0].set_ylabel(r'mean overshoot $(\widehat{K}_0^{\,U}-K_0)_+/n$')
    handles = [Line2D([], [], color=style[v][0], marker=style[v][1], ms=4, lw=1.3, label=style[v][2]) for v in style]
    handles.append(Line2D([], [], color=INK2, ls=':', lw=1.0, label=r'theoretical slope $-1/(2\gamma)$'))
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out / 'fig_rate_of_convergence.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- Figure 5
TSIZES = [1000, 5000, 10000, 15000, 20000]
METHOD_STYLE = [
    ('conditional_descending', 'conditional, descending search', BLUE, 'o', '-'),
    ('shared_descending', 'shared, descending search', ORANGE, 's', '-'),
    ('shared_bisection', 'shared, bisection', AQUA, '^', '--'),
    ('shared_gaussian_bisection', 'shared, bisection + Gaussian warm-start', YELLOW, 'D', ':'),
]


def fig_timing(out):
    """Figure 5: CPU minutes of the four searches (medians over datasets or timing runs)."""
    rows = SU.timing_table()
    for dim in (1, 2):
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        for key, label, colour, marker, ls in METHOD_STYLE:
            done = sorted((r['n'], r['cpu_seconds'] / 60) for r in rows if r['dimension'] == dim and r['method'] == key)
            if done:
                ax.plot([x for x, _ in done], [y for _, y in done], ls=ls, marker=marker, ms=6.5,
                        color=colour, label=label, lw=2.2)
        ax.set_yscale('log')
        ax.set_xticks(TSIZES)
        ax.set_xticklabels([str(n) for n in TSIZES])
        ax.set_xlim(0, 21000)
        ax.set_xlabel('sample size $n$', fontsize=11)
        ax.set_ylabel('computational time (CPU minutes)', fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(which='major', axis='both', color='#b8b7b2', linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(True)
        ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9, handlelength=3)
        fig.savefig(out / f'fig_d{dim}_computation_time.pdf', metadata={'CreationDate': None, 'ModDate': None})
        plt.close(fig)


# ------------------------------------------------------------------------------------------ Figures 6-7
def fig_regions(out):
    """Figures 6 and 7: units in candidate regions of the reported size that are not rejected, n = 7000."""
    from runners import run_calibration as C
    records = SU.calibration_records()
    for dim in (1, 2):
        r = SU.display_rep(dim)
        d = np.load(SU.CALIBRATION_RESULTS / 'calibration' / 'centre10' / f'display_d{dim}_n7000_rep{r}.npz')
        row = next(x for x in records if (x['dim'], x['n'], x['rep']) == (dim, 7000, r))
        expected = dict(schema_version=C.SCHEMA_VERSION, raw_statistic=C.RAW_STATISTIC,
                        adjusted_statistic=C.ADJUSTED_STATISTIC, grid_mode='all',
                        input_sha256=row['input_sha256'], display_selection='preselected_rep_0',
                        n=7000, raw_bound=row['raw_grid_bound'], adjusted_bound=row['adjusted_grid_bound'],
                        raw_grid_bound=row['raw_grid_bound'], adjusted_grid_bound=row['adjusted_grid_bound'])
        for key, value in expected.items():
            if key not in d or d[key].item() != value:
                raise ValueError(f'Incompatible region display: {key}; regenerate it with the revised runner')
        W, region, n = d['W'], d['region'], int(d['n'])
        names = [str(x) for x in d['region_names']]
        pcts = [f'{np.sum(region == name) / n:.0%}' for name in names]
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9 if dim == 1 else 3.6))
        for ax, score, title in zip(axes, ('raw', 'adjusted'), ('(a) raw (uncentred)', '(b) adjusted (centred)')):
            mask = d[score + '_mask'].astype(bool)
            b = int(d[score + '_grid_bound'])
            if dim == 1:
                from zeroreach.display import hist_pieces
                draw_pieces(ax, hist_pieces(W, mask, np.linspace(0, 1, 41)))
                ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
                top = ax.get_ylim()[1]
                for side, pct in zip(names, pcts):
                    idx = np.flatnonzero(region == side)
                    lo = (W[idx[0]] + W[idx[0] - 1]) / 2 if idx[0] > 0 else W[0]
                    hi = (W[idx[-1]] + W[idx[-1] + 1]) / 2 if idx[-1] < n - 1 else W[-1]
                    for x in (lo, hi):
                        ax.axvline(x, ymax=0.72, ls='--', color=INK, lw=1.0)
                    ax.text((lo + hi) / 2, 0.735 * top, pct, ha='center', va='bottom', fontsize=8, color=INK)
                ax.set_xlabel('pre-treatment covariate $W$')
                annotate(ax, bound_lines(b, n), loc='upper right')
            else:
                v = d['ranks'] / n
                ax.scatter(v[~mask, 0], v[~mask, 1], s=1.2, color=NEUTRAL, lw=0, rasterized=True)
                ax.scatter(v[mask, 0], v[mask, 1], s=1.6, color=BLUE, lw=0, rasterized=True)
                for side, pct, c in zip(names, pcts, d['centres']):
                    dist = np.sqrt(((v - v[c]) ** 2).sum(1))
                    inside = region == side
                    rad = (dist[inside].max() + dist[~inside].min()) / 2
                    ax.add_patch(Circle(tuple(v[c]), rad, fill=False, ls='--', color=INK, lw=1.0))
                    ax.text(v[c, 0], v[c, 1] + rad + 0.015, pct, ha='center', va='bottom', fontsize=8, color=INK,
                            bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.9))
                ax.set_aspect('equal')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel('normalized rank of $W_1$')
                ax.set_ylabel('normalized rank of $W_2$')
                annotate(ax, bound_lines(b, n, sym=SYM_K), loc='upper left')
            ax.set_title(title, loc='left')
        label = f'true zero-effect region ({pcts[0]} of units)'
        leg_true = Line2D([], [], color=INK, ls='--', lw=1.0, label=label)
        fig.legend(handles=[LEG_REGION, LEG_OTHER, leg_true], loc='lower center', ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, -0.06), fontsize=7.5)
        if dim == 1:
            fig.text(0.5, -0.105,
                     'Histogram bins with at least 75% of units in the union are shown entirely blue.',
                     ha='center', va='top', fontsize=7, color=INK2)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out / f'fig_d{dim}_regions_n7000.pdf', metadata={'CreationDate': None, 'ModDate': None})
        plt.close(fig)


# ---------------------------------------------------------------------------------------- Figures 1, 8-10
def fig_real(out):
    """Figures 1, 8 and 9 from the stored aggregates; Figure 10 only if unit-level files exist locally."""
    base = RES / 'real_data'
    actg = SU.real('actg')
    n = actg['n']
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, key, title in zip(axes, ('raw', 'change'), ('(a) 20-week CD4 count', '(b) change in CD4 count from baseline')):
        draw_pieces(ax, pieces(base / 'actg' / f'display_pieces_{key}.csv'))
        ax.set_xlabel('baseline CD4 count (cells/mm$^3$)')
        ax.set_title(title, loc='left')
        annotate(ax, bound_lines(actg['outcomes'][key]['bound'], n), loc='upper right')
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.3)
    axes[0].set_ylabel('number of patients')
    axes[1].set_ylabel('')
    fig.legend(handles=[LEG_REGION, LEG_OTHER], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / 'fig_real_actg175_regions.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    gs = SU.real('ggl')
    gp = pieces(base / 'ggl' / 'display_pieces.csv')
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    draw_pieces(ax, gp)
    ax.set_xlabel('household mean age in 2006 (years)')
    ax.set_ylabel('number of households')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    annotate(ax, bound_lines(gs['bound'], gs['n']), loc='upper right')
    ins = ax.inset_axes([0.62, 0.27, 0.36, 0.46])
    draw_pieces(ins, gp)
    ins.set_xlim(31, 42)
    ins.set_ylim(0, 4700)
    ins.set_ylabel('')
    ins.set_xticks([32, 34, 36, 38, 40, 42])
    ins.tick_params(labelsize=6.5)
    ins.set_title('zoom: ages 31–42', fontsize=7, pad=2)
    ax.axvspan(31, 42, color=GRID, alpha=0.7, zorder=0, lw=0)
    fig.legend(handles=[LEG_REGION, LEG_OTHER], loc='lower center', ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out / 'fig_real_ggl_regions.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    noy = SU.real('noy')['outcomes']
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, key, title in zip(axes, ('quality', 'time'), ('(a) writing quality', '(b) completion time')):
        draw_pieces(ax, pieces(base / 'noy' / f'display_pieces_{key}.csv'))
        ax.set_xlabel('baseline writing ability (task-1 grade)')
        ax.set_title(title, loc='left')
        annotate(ax, bound_lines(noy[key]['bound'], noy[key]['n']))
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.3)
    axes[0].set_ylabel('number of participants')
    axes[1].set_ylabel('')
    fig.legend(handles=[LEG_REGION, LEG_OTHER], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / 'fig_real_noy_regions.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    local = ACTG_SCATTER_INPUT or base / 'actg' / 'local' / 'two_covariates_points.npz'
    if not local.exists():
        shipped = REPO / 'figures' / 'fig_real_actg175_2d_regions.pdf'
        if not shipped.exists():
            raise FileNotFoundError('ACTG scatter requires either the shipped PDF or local provider-data arrays')
        target = out / shipped.name
        if target.resolve() != shipped.resolve():
            shutil.copyfile(shipped, target)
        print('ACTG two-dimensional scatter not regenerated: local unit-level arrays are absent; '
              'the shipped PDF is retained. Obtain provider data, rerun ACTG into a fresh output directory, '
              'then provide its local NPZ with --actg-scatter-input (see README.md).')
        return
    pts = np.load(local)
    two = SU.real('actg', 'two_covariates.json')
    xy = pts['ranks']
    if xy.shape != (two['n'], 2) or not np.isfinite(xy).all():
        raise ValueError('ACTG scatter requires a finite n-by-2 array of normalized ranks')
    if np.any((xy < 0) | (xy > 1)):
        raise ValueError('ACTG scatter ranks must lie between zero and one')
    for score in ('raw', 'change'):
        mask = pts[score]
        if (mask.shape != (two['n'],) or not np.isin(mask, [False, True]).all()
                or int(mask.sum()) != two[score]['covered_units']):
            raise ValueError('ACTG scatter arrays do not match the current result summary')
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
    for ax, (score, title) in zip(axes, (('raw', '(a) 20-week CD4 count'), ('change', '(b) change in CD4 count from baseline'))):
        mask = pts[score].astype(bool)
        ax.scatter(xy[~mask, 0], xy[~mask, 1], s=3, color=NEUTRAL, lw=0, rasterized=True)
        ax.scatter(xy[mask, 0], xy[mask, 1], s=3.5, color=BLUE, lw=0, rasterized=True)
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('normalized rank of baseline CD4')
        ax.set_ylabel('normalized rank of age')
        ax.set_title(title, loc='left')
        annotate(ax, bound_lines(two[score]['bound'], len(mask), sym=SYM_K), loc='upper left')
    fig.legend(handles=[LEG_REGION, LEG_OTHER], loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / 'fig_real_actg175_2d_regions.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


# --------------------------------------------------------------------------------------- Figures 12-14
SCEN_SHORT = {'S0': 'normal', 'S1': '$t_3$ errors', 'S2': 'heterosk.', 'S3': 'binary', 'S4': 'mixed sign',
              'S5': 'stratified'}
N_STYLE = {200: (AQUA, 'o'), 1000: (BLUE, 's'), 4000: (ORANGE, '^')}


def fig_stress(out):
    """Coverage of the actual zero-effect interval and mean bound/n over all integer lengths."""
    e2 = SU.stress()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    x = np.arange(len(SU.SCEN))
    for j, n in enumerate((200, 1000, 4000)):
        c, mk = N_STYLE[n]
        dx = (j - 1) * 0.22
        cov = [e2[f'{s}/n{n}']['raw_grid']['coverage'] for s in SU.SCEN]
        rate = np.array([v['rate'] for v in cov])
        lo = rate - np.array([v['wilson'][0] for v in cov])
        hi = np.array([v['wilson'][1] for v in cov]) - rate
        axes[0].errorbar(x + dx, rate, yerr=[lo, hi], fmt=mk, color=c, ms=4, capsize=2, lw=1.1)
        b = [e2[f'{s}/n{n}']['raw_grid']['bound_over_n'] for s in SU.SCEN]
        axes[1].plot(x + dx, b, mk, color=c, ms=4.5)
    axes[0].axhline(0.95, color=INK2, ls='--', lw=0.9, label='nominal coverage: 0.95')
    axes[0].set_ylim(0.9, 1.005)
    axes[0].set_ylabel(r'coverage of $L_0$')
    axes[0].legend(loc='lower left', frameon=False, fontsize=7.0)
    axes[1].axhline(0.10, color=INK2, ls='--', lw=0.9, label='planted fraction: 0.10')
    axes[1].legend(loc='upper left', frameon=False, fontsize=7.0)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel(r'mean $\widehat{L}_{n,0}^{\,U}/n$')
    for ax, t in zip(axes, ('(a) coverage (Wilson 95% intervals)', '(b) reported length bound')):
        ax.set_xticks(x)
        ax.set_xticklabels([SCEN_SHORT[s] for s in SU.SCEN], rotation=30, ha='right')
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(t, loc='left')
    handles = [Line2D([], [], color=N_STYLE[n][0], marker=N_STYLE[n][1], ls='', ms=4.5, label=f'$n={n:,}$')
               for n in (200, 1000, 4000)]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / 'fig_coverage_stress.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def fig_warm(out):
    """Figure 13: bisection with and without the Gaussian warm start as M grows."""
    e5 = SU.warm_start()
    Ms = [500, 2000, 10000, 50000]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for n, c in ((5000, BLUE), (20000, ORANGE)):
        for method, ls, mk in (('shared_bisection', '-', 'o'), ('shared_gaussian_bisection', '--', 's')):
            cpu = [e5[f'n{n}/M{M}/{method}']['cpu_median'] for M in Ms]
            axes[0].plot(Ms, cpu, ls=ls, marker=mk, color=c, ms=4, lw=1.3)
        acc = [e5[f'n{n}/M{M}/shared_gaussian_bisection']['accepted'] for M in Ms]
        axes[1].plot(Ms, acc, ls='--', marker='s', color=c, ms=4, lw=1.3)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xticks(Ms)
        ax.set_xticklabels(['500', '2,000', '10,000', '50,000'])
        ax.minorticks_off()
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel('number of shared draws $M$ (log scale)')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('median CPU seconds (log scale)')
    axes[0].set_title('(a) computation time', loc='left')
    axes[1].set_ylim(-0.3, 10.5)
    axes[1].set_ylabel('accepted warm starts (of 10)')
    axes[1].set_title('(b) warm-start bracket accepted', loc='left')
    handles = [Line2D([], [], color=BLUE, lw=1.3, label='$n=5{,}000$'),
               Line2D([], [], color=ORANGE, lw=1.3, label='$n=20{,}000$'),
               Line2D([], [], color=INK2, ls='-', marker='o', ms=4, lw=1.3, label='bisection'),
               Line2D([], [], color=INK2, ls='--', marker='s', ms=4, lw=1.3, label='bisection with warm start')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / 'fig_warm_start_vs_M.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def fig_effect_size(out):
    """Figure 14: mean overshoot against the effect outside the zero-effect region."""
    e9 = SU.effect_size()
    deltas = [0.1, 0.2, 0.3, 0.5, 1, 2]
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.9))
    for dim, c, mk in ((1, BLUE, 'o'), (2, ORANGE, 's')):
        ys = np.array([e9[f'd{dim}/delta{d:g}']['excess_pos_mean'] for d in deltas])
        se = np.array([e9[f'd{dim}/delta{d:g}']['excess_pos_se'] for d in deltas])
        ax.errorbar(deltas, ys, yerr=2 * se, color=c, marker=mk, ms=4, lw=1.3, capsize=2,
                    label=f'{"one covariate" if dim == 1 else "two covariates"}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(deltas)
    ax.set_xticklabels([f'{d:g}' for d in deltas])
    ax.minorticks_off()
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(r'effect $\delta$ outside the zero-effect region')
    ax.set_ylabel(r'mean overshoot $(\widehat{K}_0^{\,U}-K_0)_+/n$')
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / 'fig_effect_size.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


FIGURES = {'calibration': fig_calibration, 'rates': fig_rates, 'timing': fig_timing,
           'regions': fig_regions, 'real': fig_real, 'stress': fig_stress, 'warm': fig_warm,
           'effect_size': fig_effect_size}


def main():
    global ACTG_SCATTER_INPUT
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', type=Path, default=REPO / 'figures')
    ap.add_argument('--only', nargs='+', choices=sorted(FIGURES))
    ap.add_argument('--calibration-results', type=Path, default=SU.CALIBRATION_RESULTS,
                    help='Result root containing calibration/centre10 for the uncentred/centred comparison')
    ap.add_argument('--actg-scatter-input', type=Path,
                    help='Local two_covariates_points.npz from a matching ACTG replication; not distributed')
    a = ap.parse_args()
    if a.actg_scatter_input is not None and not a.actg_scatter_input.is_file():
        ap.error('--actg-scatter-input must name an existing local NPZ file')
    ACTG_SCATTER_INPUT = a.actg_scatter_input
    SU.CALIBRATION_RESULTS = a.calibration_results
    a.out.mkdir(parents=True, exist_ok=True)
    for name, f in FIGURES.items():
        if a.only and name not in a.only:
            continue
        with plt.rc_context(RC):
            f(a.out)
        print('drawn:', name, flush=True)


if __name__ == '__main__':
    main()
