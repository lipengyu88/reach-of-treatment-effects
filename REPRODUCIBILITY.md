# Reproducing the reported experiments

This document describes how to rebuild figures and tables, run experiments, and check the
implementation. See [README.md](README.md) for installation, the manuscript-to-file map, and full
experiment commands. Use asset names and result directories to identify outputs across manuscript versions.

## Choose the scope

| Task | Inputs and command | Output |
|---|---|---|
| Rebuild figures and numerical tables | Included compact results; `plot_figures.py` and `make_tables.py` | Figures and numerical summaries from the retained records. |
| Run an experiment from scratch | Recorded design and seeds; the appropriate runner; provider files for real data | Randomization calculations for comparison with the retained results. |
| Check the implementation | `python -m pytest -q` | Direct-statistic checks, search/configuration checks, and selected small replications. |

Run commands from the repository root using Python 3.11 after installing `requirements-frozen.txt` as described in the
README. To rebuild into a new directory without replacing the supplied figures and tables:

```bash
repro_out="$(mktemp -d)"
python plot_figures.py --out "$repro_out/figures"
python make_tables.py --out "$repro_out/tables"
```

Using the included results, the plot command regenerates 15 statistical PDF assets. Two further
assets are supplied separately:

- `fig_real_actg175_2d_regions.pdf` is supplied, and copied or preserved when the individual-level
  plotting arrays are absent. To regenerate it, obtain the ACTG provider data and follow the
  `--two-covariates-only` / `--actg-scatter-input` commands in the README. The arrays are local only.
- `figures/knn_boundary_cap.pdf` is the supplied author-drawn geometric schematic.

## What is retained

| Analysis | Included result directory | Runner |
|---|---|---|
| Main uncentred/centred comparison | `results/centering_comparison/full_integer/calibration/centre10/` | [run_calibration.py](runners/run_calibration.py) |
| Separate two-region diagnostic | `results/calibration/two_regions/` | [run_two_regions.py](runners/run_two_regions.py) |
| Additional simulation designs | `results/supplementary/<experiment>/` | [run_supplementary.py](runners/run_supplementary.py) |
| Method timings and Gaussian warm start | `results/timing/{methods,warm_start}/` | [run_timing.py](runners/run_timing.py) |
| ACTG and writing data | `results/real_data/{actg,noy}/` | [run_real_data.py](runners/run_real_data.py) |
| Full-integer GGL analysis | `results/real_data/ggl/` | [run_ggl_integer.py](runners/run_ggl_integer.py) |

Simulation `details.csv` files retain per-repetition scalar results, including the bounds and
configuration needed for aggregation; `run_meta.json` files describe the runs. Real-data files retain
summary statistics, stability or robustness calculations, and aggregate display pieces. The
[presentation manifest](results/presentation_manifest.json) records source hashes, compact-file
transformations, and figure/table inputs. Its `work/...` source paths identify author-side archives;
the included compact files are the inputs used to rebuild the figures and tables.

To generate full evaluated p-value paths, use the main calibration, two-region,
shared-versus-conditional, and real-data runners. These paths and large checkpoints are kept in
the original run bundles, while the repository contains compact summaries. Other experiments
record their own experiment-specific summaries. Obtain provider data and prepare GGL assignments
and ACTG point-level arrays locally using [data/README.md](data/README.md), which provides acquisition
instructions and file checksums. Keep these files in the documented ignored directories.

## Statistics and candidate sizes

The main comparison uses the same original outcomes, assignments, and Monte Carlo draws:

```text
raw:       sum_{i in B} (Z_i - pi) Y_i
adjusted:  sum_{i in B} (Z_i - pi) (Y_i - mean_B(Y))
```

The comparison metadata records `ols_adjustment: false` and explicit statistic names. The API and
real-data analyses use the centred statistic. ACTG's `change` outcome is the 20-week CD4 count minus
the baseline count.

Field names in the separate supplementary diagnostics have different meanings: `raw` there
means the centred statistic on original outcomes; `indep` in stress and `adjusted` in the two-region
records refer to independent OLS residuals. Stress also has an in-sample OLS diagnostic. Each run's
configuration identifies its statistic and adjustment.

The primary analyses use the full integer candidate domain `1, ..., n`. Descending search stops
at the largest non-rejected size after checking every larger integer size, so the evaluated path
contains the sizes queried up to that stopping point. This search is valid without a p-value
monotonicity assumption. Bisection in the timing comparisons queries integer sizes under the
paper's monotonicity condition. The `--grid paper` option selects the historical sparse-grid setting.

## Running and comparing new experiments

Use the [from-scratch commands](README.md#run-experiments-from-scratch) for the full configurations,
including the timing series and staged GGL search. Use fresh output directories for new runs;
keep the distributed results as the comparison target. The from-scratch commands write to ignored
`results/reproductions/` directories. The ACTG scatter example uses the separate ignored
`results/real_data/actg/local/` directory. Smoke runs use smaller configurations for a quick check;
the from-scratch commands specify the paper's repetition counts and sample sizes.

Record the code revision, provider checksums, command, output directory, software environment,
CPU/thread settings, and seeds. Compare the run configuration and statistic identities before
comparing bounds or aggregating results. Preserve the recorded environment and numerical tie rules
for exact comparisons, including the linear-algebra environment used by supplementary OLS diagnostics.

### GGL descriptive difference and standard error

After obtaining and verifying the provider file as described in [data/README.md](data/README.md),
calculate the difference in means and its standard error from the full analysed household sample:

```bash
python - <<'PY'
import numpy as np
from zeroreach.realdata import load_ggl
from runners.run_real_data import diff_in_means

households, _ = load_ggl('data/raw/ggl/ggl_individuals.dta')
y = (households.vsum / households.members).to_numpy()
z = households.z.to_numpy()
summary = diff_in_means(y, z, np.ones(len(y), dtype=bool))
print(f"Difference: {100 * summary['diff']:.2f} percentage points; "
      f"SE: {100 * summary['se']:.2f} percentage points")
PY
```

With the provider checksum specified in this repository, the output is a difference of
8.48 percentage points and a standard error of 0.34 percentage points.

## Tests

The test suite checks direct statistic calculations on small examples, search and configuration
behaviour, and selected retained-result replications. See [test_results.py](tests/test_results.py)
for the replication cases.

## Computation and data preparation

- **Timing.** The one-dimensional shared evaluator uses compensated prefix sums. The two-dimensional
  evaluator stores neighbour orders and updates compensated running sums as the queried size changes. See
  [search.py](zeroreach/search.py) (`Shared1D`, `Shared2D`) and [kernels.py](zeroreach/kernels.py).
  The [timing runner](runners/run_timing.py) includes neighbour-order construction, assignment draws,
  evaluator setup, and search. Data generation and numba warm-up are outside the timed section.
  Record hardware and thread settings alongside CPU times.
- **Writing data.** Date checks use the provider's `startdate` fields normalized to calendar days.
  The participant filter excludes a participant whose earliest recorded session as a grader predates
  that participant's task-survey date. A separate grade-level filter removes a grade when its session
  date is later than the grader's own task-survey date, where that date is available.
  See [realdata.py](zeroreach/realdata.py) (`load_noy`, `_grade_means`) and
  [data/README.md](data/README.md) for the remaining exclusions and outcome construction.
