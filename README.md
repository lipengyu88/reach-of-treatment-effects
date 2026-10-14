# The reach of treatment effects: randomization inference for zero-effect regions

Code and reproducibility materials for the paper by Pengyu Li and Yao Zhang. This revision contains
the corrected **uncentred versus within-region-centred comparison** and the completed
**all-integer-size analyses**. These results are included in both manuscript versions. The artifact
mapping below follows arXiv revision `d8b6b46` and Biometrika revision `0033d00` (14 October 2026);
subsequent editorial changes may change the numbering. Use the asset names to identify the outputs.

The method bounds the size of the largest candidate covariate region in which every unit has zero
treatment effect. Regions are consecutive blocks in a one-dimensional ordering, or nearest-neighbour
sets in normalized ranks in two dimensions. The displayed non-rejected regions are descriptive,
not confidence sets for their locations.

## Statistical definitions and search

For a region B, the main simulation compares

```text
raw (uncentred):     sum_{i in B} (Z_i - pi) Y_i
adjusted (centred):  sum_{i in B} (Z_i - pi) (Y_i - mean_B(Y))
```

Both statistics use the same original outcomes, assignments and Monte Carlo draws. No OLS fit enters
this comparison. The API and the real-data analyses use the centred statistic. In the supplementary
stress and two-region diagnostics, the legacy field `raw` means the centred statistic on the original
outcome. Independent OLS residuals are labelled `indep` in stress and `adjusted` in the two-region
records; stress also includes an `insample` OLS branch. The main comparison carries explicit statistic
labels; the separate two-region runner and aggregation checks reject mixing the two conventions.
In-sample OLS in the stress experiment is a diagnostic and does not inherit the local-null validity
guarantee of the main statistic.

The default candidate domain is **1, ..., n**. Exact descending search stops at the largest
non-rejected size: all greater integer sizes have been checked, and smaller sizes cannot change the
maximum. The evaluated path therefore need not contain n entries. The returned `bound` and
`certificate` coincide for this search. Bisection and its Gaussian warm start also query integer sizes,
but require the paper's monotonicity condition; they can return a different bound from descending
search. They are retained for the computational comparisons, not used for the new primary bounds.
An explicitly requested `grid='paper'` or `--grid paper` reproduces the historical sparse-grid option;
it is not the default or the basis of the revised results.

## Installation and quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-frozen.txt
python -m pytest -q
```

`requirements.txt` lists minimum versions; `requirements-frozen.txt` records the environment of the
paper. Run commands from the repository root. The kernels use numba; `NUMBA_NUM_THREADS` controls the
thread count, subject to the available CPU quota. The simulation workers restrict inner thread pools.

```python
import numpy as np
import zeroreach

rng = np.random.default_rng(0)
n = 1000
W = rng.uniform(size=n)
Z = np.zeros(n, dtype=int)
Z[rng.choice(n, n // 2, replace=False)] = 1
Y = 2 * W + rng.normal(scale=0.5, size=n) + Z * (np.abs(W - 0.5) >= 0.05)
result = zeroreach.upper_bound(W, Y, Z, alpha=0.05, M=1000, seed=1)
print(result['bound'], result['certificate'])
```

The default is `grid='all'`. A two-column W selects nearest-neighbour regions; `draws=` accepts
assignments sampled from another design, and `strata=` requests randomization within strata.
`result['region']` marks the union at the reported size in the original unit order.

## Repository layout

```text
zeroreach/       shared/conditional tests, search, randomization, designs, data readers, summaries
runners/        main centering comparison, separate two-region diagnostic, additional simulations,
                timings, real-data runs and the specialized full-integer GGL search
results/        compact per-repetition simulation records, configuration, real-data summaries,
                stability checks and display aggregates
figures/        manuscript figure assets
tables/         numerical tables in CSV and LaTeX
plot_figures.py regenerate figures from retained results
make_tables.py  regenerate tables from retained results
data/           provider instructions, checksums and acquisition script; raw data are ignored
tests/          statistical oracles, search/configuration checks and representative result checks
REPRODUCIBILITY.md  scope of regeneration, reruns and validation; implementation details
```

Main calibration inputs are in `results/centering_comparison/full_integer/calibration/centre10/`.
The two-region diagnostic is in `results/calibration/two_regions/`. Completed runs are kept distinct
from the `results/smoke/` and `results/paper_grid_reproduction/` outputs.

## Regenerate figures and tables without rerunning experiments

```bash
python plot_figures.py
python make_tables.py
```

The included compact simulation records retain one row per repetition, with the bounds, coverage,
truth, seeds and configuration needed for aggregation. Large evaluated-p-value paths and checkpoint
files are retained separately by the authors, rather than included in this bundle. The source hashes
and compact-file mapping are recorded in `results/presentation_manifest.json`. Full paths can be
regenerated by the experiment runners. A compact summary rebuild and a from-scratch randomization
run are different operations.

All statistical plots except the ACTG two-covariate scatter can be redrawn from the included aggregate
or synthetic inputs. The scatter uses individual-level covariates: its final PDF is included, but
redrawing it requires the provider data and a local two-covariate run. Without the ignored local
arrays, the figure script explicitly preserves that supplied PDF.
It does not reconstruct individual observations from aggregates. The boundary-geometry schematic is
an author-drawn asset, not an output of the numerical experiments. After obtaining and checking the
provider data, rebuild the scatter in a new ignored output directory:

```bash
python runners/run_real_data.py --dataset actg --two-covariates-only --out results/real_data/actg/local/scatter_rerun
python plot_figures.py --only real --actg-scatter-input results/real_data/actg/local/scatter_rerun/local/two_covariates_points.npz
```

The plot checks the supplied arrays against the retained result summary. The table below lists analysis
options; the complete commands further below give fresh output directories for real-data reruns.
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) distinguishes regenerating outputs from rerunning experiments
and explains the implementation details needed to interpret these operations.

| arXiv item | Biometrika item | Content | Figure/table files | From-scratch runner |
|---|---|---|---|---|
| Figure 1 | Figure 1 | AI-assisted writing | `fig_real_noy_regions.pdf` | `run_real_data.py --dataset noy --stability` |
| Table 1 | Table 1 | Computational costs | Analytical table in manuscript | Not a numerical experiment |
| Figure 2 | Figure S6 | Non-rejected regions, both dimensions | `fig_d{1,2}_regions_n7000.pdf` | `run_calibration.py` |
| Figure 3 | Figure 2 | Centring comparison, coverage and bound | `fig_d{1,2}_calibration_{coverage,ratio}.pdf` | `run_calibration.py` |
| Figure 4 | Figure S7 | Computational time | `fig_d{1,2}_computation_time.pdf` | `run_timing.py --experiment methods ...` |
| Figures 5 and 8 | Figure 3 | Gerber–Green–Larimer | `fig_real_ggl_regions.pdf` | `run_ggl_integer.py` (below) |
| Figures 6 and 9; Table 2 | Figure S8; Table S1 | ACTG one covariate | `fig_real_actg175_regions.pdf`, `actg_twoarm.*` | `run_real_data.py --dataset actg --stability` |
| Figures 7 and 10 | Figure S9 | ACTG two covariates | `fig_real_actg175_2d_regions.pdf` | `run_real_data.py --dataset actg --two-covariates` |
| Figure 11; Table 5 | Figure S1; Table S4 | Convergence rates | `fig_rate_of_convergence.pdf`, `rates.*` | `run_supplementary.py --experiment rates` |
| Figure 12 | Figure S2 | Boundary geometry | `knn_boundary_cap.pdf` | Author-drawn schematic |
| Figure 13; Table 3 | Figure S3; Table S2 | Stress designs | `fig_coverage_stress.pdf`, `stress.*` | `run_supplementary.py --experiment stress` |
| Table 4 | Table S3 | Two zero-effect regions | `two_regions.*` | `run_two_regions.py` |
| Figure 14; Table 6 | Figure S4; Table S5 | Gaussian warm start and draw count | `fig_warm_start_vs_M.pdf`, `warm_start.*` | `run_timing.py --experiment warm_start` |
| Table 7 | Table S6 | Dispersed/cancelling effects | `dispersed.*` | `run_supplementary.py --experiment dispersed` |
| Table 8 | Table S7 | Two-sided statistic | `two_sided.*` | `run_supplementary.py --experiment two_sided` |
| Table 9 | Table S8 | Shared versus conditional tests | `shared_vs_conditional.*` | `run_supplementary.py --experiment shared_vs_conditional` |
| Figure 15; Table 10 | Figure S5; Table S9 | Effect size | `fig_effect_size.pdf`, `effect_size.*` | `run_supplementary.py --experiment effect_size` |

The cited arXiv revision retains two copies of the GGL and ACTG displays. Each pair uses the same
asset and numerical result; it is not a separate experiment. Biometrika includes one copy of each.

Real-data headline bounds from the included summaries are:

| Analysis | Sample size | Upper bound |
|---|---:|---:|
| GGL, household mean age | 119999 | 4286 (3.6%) |
| ACTG, one covariate: 20-week count / change | 1054 | 134 / 149 (12.7% / 14.1%) |
| ACTG, two covariates: 20-week count / change | 1054 | 228 / 294 (21.6% / 27.9%) |
| Writing quality | 352 | 90 (25.6%) |
| Completion time | 353 | 65 (18.4%) |

The GGL analysis uses the recorded within-cell assignment law. ACTG assumes complete randomization
within its three stated strata; the writing analysis assumes complete randomization within each
analysed sample. Reported location displays and post-selection mean differences remain descriptive.

## Run experiments from scratch

These commands can require substantial CPU time, especially full-integer GGL and conditional
randomization. Use `--smoke` on the simulation and timing runners for small checks.
Existing record identities and statistic configurations are checked before reuse; use a new `--out`
for a different configuration. Large simulation records are ignored by Git.

```bash
# Main comparison: same Y, Z and randomization draws; only centring changes.
python runners/run_calibration.py --design centre10
# Separate supplementary diagnostic; the paper table reports centred original outcomes.
python runners/run_two_regions.py

python runners/run_supplementary.py --experiment stress
python runners/run_supplementary.py --experiment dispersed
python runners/run_supplementary.py --experiment two_sided
python runners/run_supplementary.py --experiment shared_vs_conditional
# These already used all integer sizes and were retained in this revision.
python runners/run_supplementary.py --experiment rates
python runners/run_supplementary.py --experiment effect_size

python runners/run_timing.py --experiment methods --dimension 1 --n 5000 --method shared_descending
python runners/run_timing.py --experiment methods --dimension 1 --n 1000 --method conditional_descending --repeat 3
python runners/run_timing.py --experiment methods --dimension 2 --n 5000 --method shared_bisection
python runners/run_timing.py --experiment methods --dimension 2 --n 5000 --method shared_gaussian_bisection
python runners/run_timing.py --collect --experiment methods
python runners/run_timing.py --experiment warm_start

# Obtain provider data first, using data/README.md.
python runners/run_real_data.py --dataset actg --stability --two-covariates --out results/real_data/actg/local/rerun
python runners/run_real_data.py --dataset noy --stability --out results/real_data/noy/local/rerun
```

The real-data commands write to fresh ignored directories and reject existing outputs. They preserve
the supplied summaries; compare rerun summaries before replacing any archived result. Use another
new `--out` if the example directory already exists.

For the full timing series, use both dimensions and n = 1000, 5000, 10000, 15000, 20000 for all four
methods. Shared methods use ten datasets per setting. Conditional descending uses one dataset per
setting, timed three times at n = 1000 and once at larger n. The original CPU times depend on the
reported processor and thread settings; a new machine should not be expected to reproduce them.

### Full-integer GGL search

The specialized runner has three stages. The following is a single-range reproduction of the primary
result; it is an expensive full experiment, not a smoke command.

```bash
python runners/run_ggl_integer.py prepare --data-dir data/raw --shared results/real_data/ggl/local/shared
python runners/run_ggl_integer.py scan --shared results/real_data/ggl/local/shared --out results/real_data/ggl/local/parts/all --lo 1 --hi 119999
python runners/run_ggl_integer.py merge --shared results/real_data/ggl/local/shared --parts results/real_data/ggl/local/parts/all --out results/real_data/ggl/local/rerun
```

For parallel execution, partition [1, 119999] into disjoint contiguous ranges, write each scan to a
separate part directory, and pass every part to `merge --parts`. Merge rejects gaps, overlaps,
incomplete parts and incompatible run identities. A range can stop once its largest non-rejected
size is found; the merged maximum is still exact without a p-value monotonicity assumption.
For the fifteen sensitivity configurations, use seeds 20260526, 20260527 and 20260528 with each of
`A_primary`, `B_reverse`, `C_household_size`, `D_hash1`, `E_hash2`, choosing a separate shared/parts/output
location for each. Same-seed assignments can be reused with `prepare --reuse-shared` after validation.
The old `run_real_data.py --dataset ggl` entry point supports only explicit `--grid paper`; it is not
the primary full-integer analysis.

## Reproducibility details

- Simulation repetition r uses data seed 1+r, assignment seed 10001+r and Monte Carlo seed
  20001+1000r. Independent calibration and conditional tests use their separately recorded streams.
- Primary real-data seeds are 20260526 (GGL), 20260520 (ACTG) and 20260918 (writing), with M = 5000.
  ACTG and writing stability files retain the additional seeds and draw counts stated in the paper.
- The kernels use compensated sums and explicit numerical tie handling. Tests compare them with
  independently calculated statistics and check search identities. Identical seeds alone do not
  guarantee bitwise equality across software/BLAS environments; retain the supplied numerical rules.
- Main centring requires no fitted linear model. Supplementary OLS diagnostics can depend on the
  linear-algebra environment. The runners check for an incompatible BLAS result before expensive work;
  `OPENBLAS_CORETYPE=SKYLAKEX` is an available workaround on the affected machines.
- The fixed sample sizes and 200 main repetitions were retained; the region displays use the
  preselected repetition 0 at n = 7000. Plotted coverage intervals are 95% Wilson intervals.
- `data/raw/` and `results/real_data/*/local/` are ignored. Never add provider data, GGL prepared
  assignment arrays or ACTG individual-level scatter arrays to the repository.

## Data, citation and license

[data/README.md](data/README.md) gives the provider terms and checksums for GGL (Yale ISPS D001),
ACTG 175 (UCI, DOI 10.24432/C5ZG8F) and the Noy–Zhang writing study (OSF xd7qw). Provider data are not
redistributed. The JTPA/RAND notes in the manuscript are historical design diagnostics with no reported
numerical bounds and are not included as revised full-integer results.

```bibtex
@article{li2026reach,
  title = {The Reach of Treatment Effects: Randomization Inference for Zero-Effect Regions},
  author = {Li, Pengyu and Zhang, Yao},
  year = {2026}
}
```

The code is released under the MIT License; see [LICENSE](LICENSE).
