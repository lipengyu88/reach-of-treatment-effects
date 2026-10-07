"""Write the tables of the paper (Appendices F.2 and H) from the files in results/, as LaTeX (.tex) and
with unrounded numbers (.csv).

usage: python make_tables.py [--out tables]
Table 1 (computational costs) is analytical and has no data.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd                                    # noqa: E402

from zeroreach import summaries as SU                  # noqa: E402

SCEN_LONG = {'S0': 'normal errors', 'S1': '$t_3$ errors', 'S2': 'heteroskedastic errors', 'S3': 'binary outcome',
             'S4': 'opposite-sign effects', 'S5': 'stratified design'}


def pct(c):
    return f"{100 * c['rate']:.1f}"


def write(out, name, lines, rows):
    (out / f'{name}.tex').write_text('\n'.join(lines) + '\n')
    pd.DataFrame(rows).to_csv(out / f'{name}.csv', index=False)


def table_actg(out):
    """Table F.2 (arXiv Table 2): ACTG 175, raw and baseline-adjusted outcomes."""
    a = SU.real('actg')
    n = a['n']
    L = [r'\begin{tabular}{lllccc}', r'\toprule',
         r'Outcome & Assignment draws & Adjustment & $\widehat L_{n,0}^{\,U}$ & $\widehat L_{n,0}^{\,U}/n$ & $r^{\ast}$ \\',
         r'\midrule']
    rows = []
    for key, label, adj in (('raw', r'Raw 20-week CD4 \par $Y_i=\mathrm{cd420}_i$', 'no outcome adjustment'),
                            ('change', r'CD4 change \par $Y_i=\mathrm{cd420}_i-\mathrm{cd40}_i$', 'subtract baseline CD4 count')):
        o = a['outcomes'][key]
        L.append(f"{label} & within-stratum two-arm permutation & {adj} & ${o['bound']}$ & "
                 f"${100 * o['bound'] / n:.2f}\\%$ & ${o['first_rejected']}$ \\\\")
        rows.append(dict(outcome=key, n=n, bound=o['bound'], bound_over_n=o['bound'] / n, first_rejected=o['first_rejected'],
                         certificate=o['certificate'], P_at_first_rejected=o['P_at_first_rejected'],
                         alpha_minus_P_at_first_rejected=0.05 - o['P_at_first_rejected']))
    L += [r'\bottomrule', r'\end{tabular}']
    write(out, 'actg_twoarm', L, rows)


def table_stress(out):
    """Appendix H.1 (arXiv Table 3): coverage (%) by scenario, n and method."""
    S = SU.stress()
    L = [r'\begin{tabular}{llccccc}', r'\toprule',
         r'& & \multicolumn{3}{c}{$5\%$ grid} & \multicolumn{2}{c}{All integer lengths} \\',
         r'\cmidrule(lr){3-5}\cmidrule(lr){6-7}',
         r'Scenario & $n$ & raw & \makecell{independent\\OLS} & \makecell{in-sample\\OLS} & descending & bisection \\',
         r'\midrule']
    rows = []
    for s in SU.SCEN:
        for n in (200, 1000, 4000):
            r = S[f'{s}/n{n}']
            L.append(f"{SCEN_LONG[s] if n == 200 else ''} & {n:,} & {pct(r['raw_grid']['coverage'])} & "
                     f"{pct(r['indep_grid']['coverage'])} & {pct(r['insample_grid']['coverage'])} & "
                     f"{pct(r['raw_desc']['coverage'])} & {pct(r['raw_bis']['coverage'])} \\\\")
            rows.append(dict(scenario=s, n=n, R=r['R'], raw_grid=r['raw_grid']['coverage']['rate'],
                             indep_grid=r['indep_grid']['coverage']['rate'], insample_grid=r['insample_grid']['coverage']['rate'],
                             raw_descending=r['raw_desc']['coverage']['rate'], raw_bisection=r['raw_bis']['coverage']['rate'],
                             raw_grid_bound_over_n=r['raw_grid']['bound_over_n'],
                             bisection_below_descending=r['raw_bis']['below_desc'],
                             two_sided_grid=r.get('raw_two_sided_grid', {}).get('coverage', {}).get('rate')))
    L += [r'\bottomrule', r'\end{tabular}']
    write(out, 'stress', L, rows)


def table_two_regions(out):
    """Appendix H.2 (arXiv Table 4): two disjoint zero-effect regions of 0.2n and 0.1n units."""
    S = SU.two_regions()
    L = [r'\begin{tabular}{rcccc}', r'\toprule',
         r'& \multicolumn{2}{c}{One covariate} & \multicolumn{2}{c}{Two covariates} \\', r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
         r'$n$ & coverage & $\widehat L_{n,0}^{\,U}/n$ & coverage & $\widehat K_0^{\,U}/n$ \\', r'\midrule']
    rows = []
    for n in SU.SIZES:
        a, b = S[f'd1/n{n}'], S[f'd2/n{n}']
        L.append(f"{n:,} & {pct(a['raw']['grid'])} & {a['raw']['bound_over_n']:.3f} & {pct(b['raw']['grid'])} & "
                 f"{b['raw']['bound_over_n']:.3f} \\\\")
        rows.append(dict(n=n, d1_coverage=a['raw']['grid']['rate'], d1_bound_over_n=a['raw']['bound_over_n'],
                         d2_coverage=b['raw']['grid']['rate'], d2_bound_over_n=b['raw']['bound_over_n']))
    L += [r'\bottomrule', r'\end{tabular}']
    write(out, 'two_regions', L, rows)


def table_rates(out):
    """Section 6.3 (arXiv Table 5): theoretical and fitted slopes."""
    S = SU.rates()
    names = {'d1/centre10-const': ('one', 'central, constant'), 'd1/centre10-grad1': ('one', 'central, grows with distance'),
             'd2/centre10-const': ('two', 'central, constant'), 'd2/centre10-grad1': ('two', 'central, grows with distance'),
             'd2/strip20-const': ('two', 'strip, constant')}
    L = [r'\begin{tabular}{llccc}', r'\toprule',
         r'Covariates & \makecell{Zero-effect region\\and effect} & $\gamma$ & \makecell{Theoretical slope\\$-1/(2\gamma)$} & '
         r'\makecell{Fitted slope\\$n=4000$--$16000$} \\', r'\midrule']
    rows = []
    for k, (d, lab) in names.items():
        r = S[k]
        L.append(f"{d} & {lab} & {r['gamma']:g} & ${r['theory_slope']:.2f}$ & ${r['slope_last3']:.2f}$ \\\\")
        rows.append(dict(cell=k, gamma=r['gamma'], theory_slope=r['theory_slope'], fitted_slope=r['slope_last3'],
                         **{f'overshoot_n{x["n"]}': x['excess_pos_mean'] for x in r['rows']},
                         **{f'coverage_n{x["n"]}': x['coverage']['rate'] for x in r['rows']}))
    L += [r'\bottomrule', r'\end{tabular}']
    write(out, 'rates', L, rows)


def table_warm_start(out):
    """Appendix H.4 (arXiv Table 6): CPU seconds, Monte Carlo lengths and accepted warm starts by n and M."""
    S = SU.warm_start()
    L = [r'\begin{tabular}{rrcccc}', r'\toprule',
         r'$n$ & $M$ & \makecell{Bisection\\CPU s} & \makecell{With warm-start\\CPU s} & '
         r'\makecell{Monte Carlo lengths\\bisection / warm-start} & \makecell{Warm-start\\accepted} \\', r'\midrule']
    rows = []
    for n in (5000, 20000):
        for M in (500, 2000, 10000, 50000):
            b = S[f'n{n}/M{M}/shared_bisection']
            w = S[f'n{n}/M{M}/shared_gaussian_bisection']
            L.append(f"{n:,} & {M:,} & {b['cpu_median']:.1f} & {w['cpu_median']:.1f} & "
                     f"{b['mc_queries_median']:g} / {w['mc_queries_median']:g} & {w['accepted']}/10 \\\\")
            rows.append(dict(n=n, M=M, bisection_cpu=b['cpu_median'], warm_cpu=w['cpu_median'],
                             bisection_mc_lengths=b['mc_queries_median'], warm_mc_lengths=w['mc_queries_median'],
                             warm_accepted=w['accepted'], same_bound=w['same_bound_as_bisection']))
    L += [r'\bottomrule', r'\end{tabular}']
    write(out, 'warm_start', L, rows)


def table_dispersed(out):
    """Appendix H.5 (arXiv Table 7): mean bound / n for dispersed and cancelling effects."""
    S = SU.dispersed()
    L = [r'\begin{tabular}{lrccc}', r'\toprule', r'Effect pattern & $n$ & $L_0/n$ & centred statistic & $|T_B|$ \\',
         r'\midrule']
    rows = []
    for v, lab in (('dispersed5', 'dispersed (half zero)'), ('cancel5', 'cancelling ($\\pm\\delta$)')):
        for n in (1000, 4000):
            r = S[f'{v}/n{n}']
            L.append(f"{lab if n == 1000 else ''} & {n:,} & {r['L0'][0] / n:.4f} & {r['one_sided']['mean']:.3f} & "
                     f"{r['two_sided']['mean']:.3f} \\\\")
            rows.append(dict(effect=v, n=n, L0_over_n=r['L0'][0] / n, centred_mean_bound_over_n=r['one_sided']['mean'],
                             two_sided_mean_bound_over_n=r['two_sided']['mean']))
    L += [r'\bottomrule', r'\end{tabular}']
    write(out, 'dispersed', L, rows)


def table_two_sided(out):
    """Appendix H.6 (arXiv Table 8): centred and two-sided statistics under three effect patterns."""
    S = SU.two_sided()
    lab = {'const': 'positive, $+1$', 'neg': 'negative, $-1$', 'asym': '$+1$ below, $-0.5$ above'}
    L = [r'\begin{tabular}{lrcccc}', r'\toprule',
         r'& & \multicolumn{2}{c}{Centred statistic} & \multicolumn{2}{c}{$|T_B|$} \\', r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
         r'Effect outside the block & $n$ & coverage & bound$/n$ & coverage & bound$/n$ \\', r'\midrule']
    rows = []
    for v in ('const', 'neg', 'asym'):
        for n in (200, 1000, 4000):
            r = S[f'{v}/n{n}']
            L.append(f"{lab[v] if n == 200 else ''} & {n:,} & {pct(r['one_sided']['coverage'])} & {r['one_sided']['bound_over_n']:.3f} & "
                     f"{pct(r['two_sided']['coverage'])} & {r['two_sided']['bound_over_n']:.3f} \\\\")
            rows.append(dict(effect=v, n=n, centred_coverage=r['one_sided']['coverage']['rate'],
                             centred_bound_over_n=r['one_sided']['bound_over_n'],
                             two_sided_coverage=r['two_sided']['coverage']['rate'],
                             two_sided_bound_over_n=r['two_sided']['bound_over_n']))
    L += [r'\bottomrule', r'\end{tabular}']
    write(out, 'two_sided', L, rows)


def table_shared_vs_conditional(out):
    """Appendix H.7 (arXiv Table 9): shared and conditional tests on the same datasets."""
    S = SU.shared_vs_conditional()
    L = [r'\begin{tabular}{rlcccc}', r'\toprule', r'$n$ & Test & Coverage (\%) & Mean bound$/n$ & CPU s & Same (\%) \\',
         r'\midrule']
    rows = []
    for n in (1000, 4000):
        r = S[f'n{n}']
        L.append(f"{n:,} & shared & {pct(r['shared']['coverage'])} & {r['shared']['bound_over_n']:.4f} & "
                 f"{r['shared']['cpu']:.2f} & {100 * r['identical']:.1f} \\\\")
        L.append(f" & conditional & {pct(r['conditional']['coverage'])} & {r['conditional']['bound_over_n']:.4f} & "
                 f"{r['conditional']['cpu']:.1f} & \\\\")
        rows.append(dict(n=n, shared_coverage=r['shared']['coverage']['rate'], shared_bound_over_n=r['shared']['bound_over_n'],
                         shared_cpu=r['shared']['cpu'], conditional_coverage=r['conditional']['coverage']['rate'],
                         conditional_bound_over_n=r['conditional']['bound_over_n'], conditional_cpu=r['conditional']['cpu'],
                         same=r['identical'], conditional_larger=r['cond_larger'], shared_larger=r['shared_larger']))
    L += [r'\bottomrule', r'\end{tabular}']
    write(out, 'shared_vs_conditional', L, rows)


def table_effect_size(out):
    """Appendix H.8 (arXiv Table 10): mean overshoot and coverage by effect size."""
    S = SU.effect_size()
    L = [r'\begin{tabular}{rcccc}', r'\toprule',
         r'& \multicolumn{2}{c}{One covariate} & \multicolumn{2}{c}{Two covariates} \\', r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
         r'$\delta$ & overshoot & coverage & overshoot & coverage \\', r'\midrule']
    rows = []
    for d in ('0.1', '0.2', '0.3', '0.5', '1', '2'):
        a, b = S[f'd1/delta{d}'], S[f'd2/delta{d}']
        L.append(f"{d} & {a['excess_pos_mean']:.3f} & {pct(a['coverage'])} & {b['excess_pos_mean']:.3f} & {pct(b['coverage'])} \\\\")
        rows.append(dict(delta=float(d), d1_overshoot=a['excess_pos_mean'], d1_coverage=a['coverage']['rate'],
                         d2_overshoot=b['excess_pos_mean'], d2_coverage=b['coverage']['rate']))
    L += [r'\bottomrule', r'\end{tabular}']
    write(out, 'effect_size', L, rows)


def table_calibration(out):
    """Numbers behind Figures 2 and 3 (Section 6.2)."""
    rows = SU.calibration('centre10')
    pd.DataFrame(rows).to_csv(out / 'calibration.csv', index=False)


def table_timing(out):
    """Numbers behind Figure 5 (Section 6.4): median CPU seconds."""
    pd.DataFrame(SU.timing_table()).to_csv(out / 'timing.csv', index=False)


TABLES = [table_actg, table_stress, table_two_regions, table_rates, table_warm_start, table_dispersed, table_two_sided,
          table_shared_vs_conditional, table_effect_size, table_calibration, table_timing]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', type=Path, default=REPO / 'tables')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for t in TABLES:
        t(a.out)
    print('written:', sorted(p.name for p in a.out.iterdir()))


if __name__ == '__main__':
    main()
