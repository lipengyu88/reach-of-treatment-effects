# The Reach of Treatment Effects: Randomization Inference for Zero-Effect Regions

Python implementation and reproducibility materials for the paper by Pengyu Li and Yao Zhang.

**Paper:** [The Reach of Treatment Effects: Randomization Inference for Zero-Effect Regions](https://arxiv.org/abs/2610.20759), arXiv:2610.20759.

The experiments compare uncentred and within-region-centred statistics and search all integer region
sizes. The table below maps the results to the arXiv paper and JRSSB manuscript
(16 October 2026).

The method bounds the size of the largest candidate covariate region in which every unit has zero
treatment effect. Regions are consecutive blocks in a one-dimensional ordering, or nearest-neighbour
sets in normalized ranks in two dimensions. Plots show the locations of non-rejected regions
as descriptive displays.

## Statistical definitions and search

For a region B, the main simulation compares

```text
raw (uncentred):     sum_{i in B} (Z_i - pi) Y_i
adjusted (centred):  sum_{i in B} (Z_i - pi) (Y_i - mean_B(Y))
```

Both statistics use the same original outcomes, assignments and Monte Carlo draws; the comparison
changes only within-region centring. The API and real-data analyses use the centred statistic.
In the additional stress and two-region diagnostics, `raw` denotes the centred statistic on the
original outcome. Independent OLS residuals are labelled `indep` in stress and `adjusted` in the
two-region records. The `insample` OLS branch in the stress experiment is exploratory. The main
analyses use original outcomes; the independent-residual diagnostics use separately fitted models.
Run metadata and aggregation checks identify the statistic used in each experiment.

The default candidate domain is **1, ..., n**. Exact descending search stops at the largest
non-rejected size after checking every greater integer size. The returned `bound` and `certificate`
coincide for this search. Primary analyses use exact descending search. The timing comparisons also
include bisection and a Gaussian warm start, which query integer sizes under the paper's monotonicity
condition. Set `grid='paper'` or `--grid paper` to select the optional sparse candidate grid.

## Installation and quick start

Use Python 3.11 with the frozen environment below.

```bash
python3.11 -m venv .venv
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
REPRODUCIBILITY.md  reproduction commands, result formats and implementation notes
```

Main calibration inputs are in `results/centering_comparison/full_integer/calibration/centre10/`.
The two-region diagnostic is in `results/calibration/two_regions/`.

## Regenerate figures and tables without rerunning experiments

```bash
python plot_figures.py
python make_tables.py
```

The included compact simulation records retain one row per repetition, with the bounds, coverage,
truth, seeds and configuration needed for aggregation. Source hashes and the compact-file mapping
are recorded in `results/presentation_manifest.json`. The main calibration, two-region,
shared-versus-conditional and real-data runners produce evaluated p-value paths; the other runners
save experiment-specific summaries. The authors retain full run archives, including large paths
and checkpoint files, separately from the compact results distributed here.

The plot command regenerates 15 statistical PDFs from the included aggregate or synthetic inputs.
The ACTG two-covariate scatter is supplied as a PDF and can be redrawn using the provider data and
a local two-covariate run. The figure script preserves the supplied PDF when local plotting arrays
are absent. The boundary-geometry schematic is supplied as an author-drawn PDF. To rebuild the
ACTG scatter after obtaining and checking the provider data:

```bash
python runners/run_real_data.py --dataset actg --two-covariates-only --out results/real_data/actg/local/scatter_rerun
python plot_figures.py --only real --actg-scatter-input results/real_data/actg/local/scatter_rerun/local/two_covariates_points.npz
```

The plot checks the supplied arrays against the retained result summary. The table below lists analysis
options; the complete commands further below give fresh output directories for real-data reruns.
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) gives further reproduction commands and implementation notes.

| arXiv item | JRSSB item | Content | Figure/table files | From-scratch runner |
|---|---|---|---|---|
| Figure 1 | Figure 1 | AI-assisted writing | `fig_real_noy_regions.pdf` | `run_real_data.py --dataset noy --stability` |
| Table 1 | Table S1 | Computational costs | Analytical table in manuscript | Analytical derivation |
| Figure 2 | Figure S3 | Non-rejected regions, both dimensions | `fig_d{1,2}_regions_n7000.pdf` | `run_calibration.py` |
| Figure 3 | Figure 2 | Centring comparison, coverage and bound | `fig_d{1,2}_calibration_{coverage,ratio}.pdf` | `run_calibration.py` |
| Figure 4 | Figure S4 | Computational time | `fig_d{1,2}_computation_time.pdf` | `run_timing.py --experiment methods ...` |
| Figure 5 | Figure 3 | Gerber–Green–Larimer | `fig_real_ggl_regions.pdf` | `run_ggl_integer.py` (below) |
| Figure 6 | Figure S5 | ACTG one covariate | `fig_real_actg175_regions.pdf` | `run_real_data.py --dataset actg --stability` |
| Figure 7 | Figure S6 | ACTG two covariates | `fig_real_actg175_2d_regions.pdf` | `run_real_data.py --dataset actg --two-covariates` |
| Figure 8 | Figure S1 | Boundary geometry | `knn_boundary_cap.pdf` | Author-drawn schematic |
| Figure 9 | Figure S2 | Convergence rates | `fig_rate_of_convergence.pdf` | `run_supplementary.py --experiment rates` |

Additional diagnostics and numerical tables are retained with the code, including stress designs,
two zero-effect regions, Gaussian warm starts, dispersed or cancelling effects, two-sided statistics,
shared versus conditional tests, and effect-size comparisons. These provide complementary analyses.
The `actg_twoarm.*` and `rates.*` tables provide numerical summaries for the corresponding analyses.

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

Use `--smoke` on the simulation and timing runners for a small initial run. Full-integer GGL and
conditional randomization are the most computationally intensive experiments.
Choose a new run directory so that the supplied results remain available for comparison. All outputs
under `results/reproductions/` are ignored by Git. Set `run_out` once in the shell used for the commands
below; use a different directory for each new configuration or independent timing run.

```bash
run_out="results/reproductions/run_01"

# Main comparison: same Y, Z and randomization draws; only centring changes.
python runners/run_calibration.py --design centre10 --out "$run_out/calibration/centre10"
# Two-region diagnostic with centred original outcomes.
python runners/run_two_regions.py --out "$run_out/calibration/two_regions"

python runners/run_supplementary.py --experiment stress --out "$run_out/supplementary/stress"
python runners/run_supplementary.py --experiment dispersed --out "$run_out/supplementary/dispersed"
python runners/run_supplementary.py --experiment two_sided --out "$run_out/supplementary/two_sided"
python runners/run_supplementary.py --experiment shared_vs_conditional --out "$run_out/supplementary/shared_vs_conditional"
# Convergence rates and effect-size comparisons.
python runners/run_supplementary.py --experiment rates --out "$run_out/supplementary/rates"
python runners/run_supplementary.py --experiment effect_size --out "$run_out/supplementary/effect_size"

python runners/run_timing.py --experiment methods --dimension 1 --n 5000 --method shared_descending --out "$run_out/timing/methods"
python runners/run_timing.py --experiment methods --dimension 1 --n 1000 --method conditional_descending --repeat 3 --out "$run_out/timing/methods"
python runners/run_timing.py --experiment methods --dimension 2 --n 5000 --method shared_bisection --out "$run_out/timing/methods"
python runners/run_timing.py --experiment methods --dimension 2 --n 5000 --method shared_gaussian_bisection --out "$run_out/timing/methods"
python runners/run_timing.py --collect --experiment methods --out "$run_out/timing/methods"
python runners/run_timing.py --experiment warm_start --out "$run_out/timing/warm_start"

# Obtain provider data first, using data/README.md.
python runners/run_real_data.py --dataset actg --stability --two-covariates --out "$run_out/real_data/actg"
python runners/run_real_data.py --dataset noy --stability --out "$run_out/real_data/noy"
```

The simulation and timing runners resume from task files in `<out>/records/`. A fresh checkout starts
new runs; its included `details.csv` files provide comparison results. The main calibration,
two-region and supplementary runners validate record configurations before skipping completed tasks.
The timing runner skips existing filenames, so use a new directory after changing the draw count,
hardware or thread settings. Reuse a run directory to continue the same configuration. Use
`--collect --out <run-directory>` to rebuild summaries from that directory's `records/` files.
Start each ACTG or writing analysis with a fresh `--out` directory.

The timing commands above illustrate individual settings. For the full series, use both dimensions
and n = 1000, 5000, 10000, 15000, 20000 for all four
methods. Shared methods use ten datasets per setting. Conditional descending uses one dataset per
setting, timed three times at n = 1000 and once at larger n. Record the processor and thread settings
alongside the measured CPU times.

### Full-integer GGL search

The specialized runner has three stages. The following runs the full primary experiment as a single
range over all 119,999 candidate sizes:

```bash
run_out="results/reproductions/run_01"
python runners/run_ggl_integer.py prepare --data-dir data/raw --shared "$run_out/real_data/ggl/shared"
python runners/run_ggl_integer.py scan --shared "$run_out/real_data/ggl/shared" --out "$run_out/real_data/ggl/parts/all" --lo 1 --hi 119999
python runners/run_ggl_integer.py merge --shared "$run_out/real_data/ggl/shared" --parts "$run_out/real_data/ggl/parts/all" --out "$run_out/real_data/ggl/summary"
```

For parallel execution, partition [1, 119999] into disjoint contiguous ranges, write each scan to a
separate part directory, and pass every part to `merge --parts`. Merge rejects gaps, overlaps,
incomplete parts and incompatible run identities. A range can stop once its largest non-rejected
size is found; the merged maximum is still exact without a p-value monotonicity assumption.
For the fifteen sensitivity configurations, use seeds 20260526, 20260527 and 20260528 with each of
`A_primary`, `B_reverse`, `C_household_size`, `D_hash1`, `E_hash2`, choosing a separate shared/parts/output
location for each. Same-seed assignments can be reused with `prepare --reuse-shared` after validation.
Use `run_ggl_integer.py` for the primary full-integer analysis. The optional sparse-grid analysis is
available through `run_real_data.py --dataset ggl --grid paper`.

## Reproducibility details

- Simulation repetition r uses data seed 1+r, assignment seed 10001+r and Monte Carlo seed
  20001+1000r. Independent calibration and conditional tests use their separately recorded streams.
- Primary real-data seeds are 20260526 (GGL), 20260520 (ACTG) and 20260918 (writing), with M = 5000.
  ACTG and writing stability files retain the additional seeds and draw counts stated in the paper.
- The kernels use compensated sums and explicit numerical tie handling. Tests compare them with
  independently calculated statistics and check search identities. Use the frozen environment and
  supplied numerical rules for result comparisons.
- Supplementary OLS diagnostics use BLAS calculations, which the runners check before execution.
  Use a BLAS build compatible with the host CPU. The original Intel servers used
  `OPENBLAS_CORETYPE=SKYLAKEX`, which requires AVX-512 support.
- The main simulation uses 200 repetitions at each sample size; the region displays use the
  preselected repetition 0 at n = 7000. Plotted coverage intervals are 95% Wilson intervals.
- Keep provider data, GGL prepared assignments and ACTG individual-level plotting arrays in the
  ignored `data/raw/` and `results/real_data/*/local/` directories.

## Data, citation and license

[data/README.md](data/README.md) gives the provider terms and checksums for GGL (Yale ISPS D001),
ACTG 175 (UCI, DOI 10.24432/C5ZG8F) and the Noy–Zhang writing study (OSF xd7qw).
Obtain the original data from these providers using the instructions in that file.

```bibtex
@article{li2026reach,
  title = {The Reach of Treatment Effects: Randomization Inference for Zero-Effect Regions},
  author = {Li, Pengyu and Zhang, Yao},
  year = {2026},
  eprint = {2610.20759},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2610.20759}
}
```

The code is released under the MIT License; see [LICENSE](LICENSE).
