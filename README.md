# The reach of treatment effects: randomization inference for zero-effect regions

This repository contains the code, the simulation and real-data results, and the figures and tables of
the paper *The reach of treatment effects: randomization inference for zero-effect regions* by Pengyu Li
and Yao Zhang.

Given a pre-treatment covariate, the method asks how large a group of units can be on which treatment
has no effect at all. Local Fisher randomization tests are run on candidate regions of the covariate
(blocks of consecutive units in one dimension, nearest-neighbour regions in normalized ranks in two) and
inverted into an upper confidence bound for the size of the largest zero-effect region. Shared
randomization draws, the centred linear statistic and cumulative sums reduce the cost of all local tests
from O(Mn^3) to O(Mn^2); the descending search needs no further assumption, and bisection with a Gaussian
warm start is faster under a monotonicity assumption.

## Repository layout

```text
.
├── zeroreach/                  the method (importable package)
│   ├── api.py                  upper_bound(): the whole procedure for one dataset
│   ├── kernels.py              compiled kernels (numba): shared and conditional tests, 1D blocks, 2D regions
│   ├── search.py               evaluators and the searches of Table 1 (descending, bisection, warm start)
│   ├── grids.py                searched grids and the inversion with the tail envelope
│   ├── randomization.py        complete randomization, randomization within strata or cells
│   ├── designs.py              simulation designs of Section 6 and Appendix H
│   ├── realdata.py             readers for the three experiments of Sections 1, 6.7 and 6.8
│   ├── display.py              aggregates behind the region displays
│   ├── summaries.py, style.py  summaries of results/ and figure style
│   └── utils.py
├── runners/
│   ├── run_calibration.py      Section 6.2, Section 6.5 and Appendix H.2
│   ├── run_supplementary.py    Section 6.3 and Appendix H
│   ├── run_timing.py           Section 6.4 and Appendix H.4
│   └── run_real_data.py        Sections 1, 6.7 and 6.8
├── results/                    outputs of the runners (included)
├── figures/                    the figures of the paper (included)
├── tables/                     the tables of the paper, as LaTeX and CSV (included)
├── plot_figures.py             draws every figure from results/
├── make_tables.py              writes every table from results/
├── data/                       README.md (sources, terms, checksums) and download.py; the data are not included
├── tests/                      pytest
├── requirements.txt            minimum versions
├── requirements-frozen.txt     exact versions used for the paper
├── CITATION.cff
└── LICENSE
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt         # or: pip install -r requirements-frozen.txt (versions of the paper)
pytest -q                               # about 20 seconds the first time, while numba compiles the kernels
```

Run all commands from the repository root. The compiled kernels use all available cores (in containers,
no more than the CPU quota); set `NUMBA_NUM_THREADS` to change this.

## Quick start

```python
import numpy as np
import zeroreach

rng = np.random.default_rng(0)
n = 1000
W = rng.uniform(size=n)                                   # pre-treatment covariate
Z = np.zeros(n, dtype=int)
Z[rng.choice(n, n // 2, replace=False)] = 1               # completely randomized assignment
tau = np.where(np.abs(W - 0.5) < 0.05, 0.0, 1.0)          # no effect for units with W near 0.5
Y = 2 * W + rng.normal(scale=0.5, size=n) + tau * Z

res = zeroreach.upper_bound(W, Y, Z, alpha=0.05, M=1000, seed=1)
print(res['bound'], res['first_rejected'], res['certificate'])   # 150 200 199
```

The largest zero-effect block here has 93 units. On the 5% grid the bound is 150 units; the next searched
length, 200, is rejected, so 199 bounds the size over all integer lengths. `res['region']` marks the units
covered by blocks of length 150 that are not rejected; these are descriptive, not a confidence set for
the location of a zero-effect region. A two-column `W` gives nearest-neighbour regions; `grid='all'`
searches every size with the descending search; `draws=` takes randomization draws from another design.

## Reproduce the figures and tables from the included results

```bash
python plot_figures.py      # figures/ (a few seconds)
python make_tables.py       # tables/
```

With the versions in `requirements-frozen.txt` the figures are identical to those in the paper; with
other matplotlib versions the layout can differ by a few pixels.

## Figures and tables of the paper

Numbers follow the arXiv version; the Biometrika version moves some of them to the Supplementary
Material (prefix S). Status: *covered* means that the command regenerates the result from the
repository alone; *needs data* that it needs the public data of `data/README.md` first.

| arXiv | Biometrika | Content | Files | Command | Status |
|---|---|---|---|---|---|
| Fig. 1 | Fig. 1 | Noy-Zhang writing experiment | `fig_real_noy_regions.pdf` | `run_real_data.py --dataset noy --stability` | needs data (redrawn from results/) |
| Table 1 | Table 1 | Computational costs | — | analytical | — |
| Fig. 2 | Fig. 2 | Calibration, one covariate | `fig_d1_calibration_{ratio,coverage}.pdf` | `run_calibration.py --design centre10` | covered |
| Fig. 3 | Fig. S5 | Calibration, two covariates | `fig_d2_calibration_{ratio,coverage}.pdf` | same | covered |
| Fig. 4, Table 5 | Fig. S6, Table S4 | Rate of convergence | `fig_rate_of_convergence.pdf`, `rates.tex` | `run_supplementary.py --experiment rates` | covered |
| Fig. 5 | Fig. S7 | Computation time | `fig_d{1,2}_computation_time.pdf` | `run_timing.py --experiment methods ...` | covered (machine dependent) |
| Figs 6-7 | Figs S8-S9 | Region displays, n = 7000 | `fig_d{1,2}_regions_n7000.pdf` | `run_calibration.py --design centre10` | covered |
| Fig. 8 | Fig. 3 | Gerber-Green-Larimer | `fig_real_ggl_regions.pdf` | `run_real_data.py --dataset ggl --robustness` | needs data (redrawn from results/) |
| Fig. 9, Table 2 | Fig. S10, Table S1 | ACTG 175 | `fig_real_actg175_regions.pdf`, `actg_twoarm.tex` | `run_real_data.py --dataset actg --stability --two-covariates` | needs data (redrawn from results/) |
| Fig. 10 | Fig. S11 | ACTG 175, two covariates | `fig_real_actg175_2d_regions.pdf` | same | needs data |
| Fig. 11 | Fig. S1 | Boundary geometry (schematic) | `knn_boundary_cap.pdf` | drawn by the authors, not generated from data | — |
| Fig. 12, Table 3 | Fig. S2, Table S2 | Validity stress tests | `fig_coverage_stress.pdf`, `stress.tex` | `run_supplementary.py --experiment stress` | covered |
| Table 4 | Table S3 | Two zero-effect regions | `two_regions.tex` | `run_calibration.py --design two_regions` | covered |
| Fig. 13, Table 6 | Fig. S3, Table S5 | Warm start and M | `fig_warm_start_vs_M.pdf`, `warm_start.tex` | `run_timing.py --experiment warm_start` | covered (machine dependent) |
| Table 7 | Table S6 | Dispersed and cancelling effects | `dispersed.tex` | `run_supplementary.py --experiment dispersed` | covered |
| Table 8 | Table S7 | Two-sided statistic | `two_sided.tex` | `run_supplementary.py --experiment two_sided` | covered |
| Table 9 | Table S8 | Shared and conditional tests | `shared_vs_conditional.tex` | `run_supplementary.py --experiment shared_vs_conditional` | covered |
| Fig. 14, Table 10 | Fig. S4, Table S9 | Effect size | `fig_effect_size.pdf`, `effect_size.tex` | `run_supplementary.py --experiment effect_size` | covered |

The real-data figures are redrawn from the aggregates in `results/real_data/` (counts behind the
histograms, bounds, p-value paths), so `plot_figures.py` needs no data; recomputing those aggregates
needs the data. Figure 10 shows unit-level points and is redrawn only after the ACTG 175 analysis has
been run locally. The appendix notes on the JTPA and RAND Health Insurance experiments report no numbers
and are not part of this repository.

## Running the experiments

Every runner has a `--smoke` option that runs a few small tasks in seconds and writes to `results/smoke/`.
The simulation runners write one record per task under `<out>/records/` (skipped if it exists, so a run
can be resumed or split across machines) and collect them into `details.csv` (and `summary.csv`) at the
end. CPU hours below are those of our runs.

```bash
# Section 6.2, 6.5 and Appendix H.2: 9 sample sizes x 2 dimensions x 200 repetitions per design
python runners/run_calibration.py --design centre10        # about 10 CPU hours
python runners/run_calibration.py --design two_regions     # about 9 CPU hours

# Section 6.3 and Appendix H (one process per core, NUMBA_NUM_THREADS=1 per process is set automatically)
python runners/run_supplementary.py --experiment rates                  # about 76 CPU hours
python runners/run_supplementary.py --experiment stress                 # about 13 CPU hours
python runners/run_supplementary.py --experiment effect_size            # about 12 CPU hours
python runners/run_supplementary.py --experiment shared_vs_conditional  # about 3 CPU hours
python runners/run_supplementary.py --experiment two_sided              # under 1 CPU hour
python runners/run_supplementary.py --experiment dispersed              # under 1 CPU hour

# Section 6.4: one search, one dimension and one n per call; shared searches on 10 datasets
python runners/run_timing.py --experiment methods --dimension 1 --n 5000 --method shared_descending
python runners/run_timing.py --experiment methods --dimension 1 --n 1000 --method conditional_descending --repeat 3
python runners/run_timing.py --experiment warm_start
python runners/run_timing.py --collect --experiment methods             # collect the records into details.csv

# Sections 1, 6.7 and 6.8 (data in data/raw/, see data/README.md)
python runners/run_real_data.py --dataset ggl --robustness              # a few minutes with 32 threads
python runners/run_real_data.py --dataset actg --stability --two-covariates
python runners/run_real_data.py --dataset noy --stability
```

The conditional descending search is slow by design (Table 1): one run took 306 CPU hours in one
dimension and 1,097 CPU hours (46 CPU days) in two dimensions at n = 20,000; it saves a checkpoint every
ten minutes and resumes from it. The timings of the paper are CPU times on AMD EPYC 7J13 processors with
24 threads (one dimension) and on Intel Xeon Platinum 8358 processors (two dimensions; 50 threads for the
shared searches, 16 to 50 for the conditional search, whose longest runs were resumed from checkpoints).

## Paper names and code names

| Paper | Code |
|---|---|
| shared tests, descending search / bisection / bisection with Gaussian warm start | `search.Shared1D`, `search.Shared2D` with `search.descending`, `search.bisection(warm=False/True)`; `shared_descending`, `shared_bisection`, `shared_gaussian_bisection` |
| conditional tests (computed from scratch), descending search | `search.Cond1D`, `search.Cond2D`; `conditional_descending` |
| centred statistic T_B; two-sided statistic \|T_B\| | `kernels` (all evaluators); `search.numpy_grid_pvalues(two_sided=True)` |
| 5%-of-n grid; tail envelope; r* and the certificate r* - 1 | `grids.paper_grid`, `grids.invert_path` (`grid_bound`, `first_rejected`, `certificate`) |
| raw outcome; adjustment with an independent calibration sample; in-sample adjustment | `y` (`raw`); `adj_indep` (`adjusted`); `adj_insample` |
| zero-effect region of the calibration: the 10% of units nearest the centre | `designs.generate(zero='centre10')` |
| two zero-effect regions (20% and 10%); strip; disc used for timing | `zero='two_regions'`, `'strip20'`, `'circle'` |
| constant effect; effect growing with the distance | `effect='const'`, `'grad1'` |
| normal, t3, heteroskedastic, binary, opposite signs, stratified (Appendix H.1) | `scenario='S0'`, ..., `'S5'` |

## Reproducibility notes

- **Seeds.** Repetition r of a simulation uses the data seed 1 + r, the assignment seed 10001 + r, the
  Monte Carlo seed 20001 + 1000 r and, for the independent calibration sample, 20260619 + 10^6 s +
  600000 + r, where s indexes the sample size (`designs.seed_plan`). Conditional tests use per-region
  streams based on 20260929 + 1000 r. The real-data analyses use M = 5000 draws with seeds 20260526
  (Gerber-Green-Larimer), 20260520 (ACTG 175) and 20260918 (Noy-Zhang); their checks use the seeds and
  Monte Carlo sizes stated in the paper.
- **Exactness.** The kernels use compensated sums and resolve near ties by comparing the assignments
  exactly, so results do not depend on the number of threads. We reran repetitions 0-9 of every
  simulation cell and all real-data analyses with this code, and they reproduced the stored results
  exactly; `tests/test_results.py` reruns a few tasks.
- **Least squares.** The adjusted outcome uses a small least-squares fit; different BLAS libraries can
  change the coefficients in the last bit. If a machine returns wrong float64 results with OpenBLAS (the
  runners check this before starting), set `OPENBLAS_CORETYPE=SKYLAKEX`.
- **Timing** depends on the processor, the number of threads and the load of the machine; the shape of
  the curves and the ordering of the methods are what the paper reports.

## Data

See [data/README.md](data/README.md): the Gerber-Green-Larimer experiment (Yale ISPS Data Archive, study
D001), ACTG 175 (UCI Machine Learning Repository, doi:10.24432/C5ZG8F) and the Noy-Zhang writing
experiment (OSF xd7qw). No data are redistributed here.

## Citation

If you use this code, please cite the paper (the reference will be completed on publication):

```bibtex
@article{li2026reach,
  title  = {The reach of treatment effects: randomization inference for zero-effect regions},
  author = {Li, Pengyu and Zhang, Yao},
  year   = {2026}
}
```

## License

The code is released under the MIT License; see [LICENSE](LICENSE).
