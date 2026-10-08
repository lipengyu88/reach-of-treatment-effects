# Data

The three experiments of Sections 1, 6.7 and 6.8 use public data that are **not** redistributed here. Put the
files under `data/raw/` as below; `runners/run_real_data.py` reads them from there (or from `--data-dir`).
Everything stored under `results/real_data/` is aggregated (bounds, p-value paths, counts behind the
displays); no unit-level value is part of this repository.

```text
data/raw/
├── ggl/ggl_individuals.dta
├── actg175/actg175.csv
└── noy/
    ├── fullsurvey.csv
    ├── grading_long_complete.csv
    ├── grading.xlsx
    └── grading_retry.xlsx
```

`python data/download.py` downloads the ACTG 175 and Noy–Zhang files and checks every file against the
SHA-256 sums below; `python data/download.py --check` only checks.

| File | Source | SHA-256 |
|---|---|---|
| `ggl/ggl_individuals.dta` | Gerber, Green and Larimer (2008), replication data, Yale ISPS Data Archive, study D001 (ISPS08-001). The archive asks users to accept its terms of use. Download the individual-level Stata file `GerberGreenLarimer_APSR_2008_social_pressure.dta` (Dataverse file `https://dataverse.yale.edu/api/access/datafile/42238?format=original`) and save it under this name. | `d8dd711c5216443d21e60050d768a7269b3ea9d656dbedd6802a738c2d0b080f` |
| `actg175/actg175.csv` | AIDS Clinical Trials Group Study 175, UCI Machine Learning Repository, [doi:10.24432/C5ZG8F](https://doi.org/10.24432/C5ZG8F); file `https://archive.ics.uci.edu/static/public/890/data.csv` | `cf58638c7687764ccb32d037938142518293d505eec349ccec406d4d42ba9532` |
| `noy/fullsurvey.csv` | Noy and Zhang (2023), replication package, [OSF xd7qw](https://osf.io/xd7qw/); `https://osf.io/download/dch8w/` | `06f430ba9a9194039530721d68ad37f95552977a1ae1868c9228d1ce6334db0f` |
| `noy/grading_long_complete.csv` | same; `https://osf.io/download/hdszk/` | `92771e43f2116df81a8681d281a94e94dec8ca4a2d0a04ff0a1ecfe3bcdc2779` |
| `noy/grading.xlsx` | same; `https://osf.io/download/m35kf/` | `76e66cb815a3e9b7e333ff50f8f49fd661857bc4614fae62fb987d06590edd1e` |
| `noy/grading_retry.xlsx` | same; `https://osf.io/download/3qcfb/` | `09009bc52b50d76e4585a8072a63115c819dbd21ab5da553c7f5bd01b2898e27` |

Follow each provider's terms of use. The analysed samples are built by `zeroreach/realdata.py`:

- **Gerber–Green–Larimer.** Households of the Control and Neighbors groups (119,999 households, 20,000
  of them treated, 10,000 cells). The outcome is the household turnout rate in 2006 and the covariate the
  mean age of the members in 2006; both are kept as exact integers after multiplying by the least common
  multiple of the household sizes.
- **ACTG 175.** Arms 0 and 1 (1,054 patients, 522 treated), ordered by baseline CD4 count `cd40` with
  ties broken by `pidnum`; outcomes `cd420` and `cd420 - cd40`; randomization within `strat`.
- **Noy–Zhang.** The cleaning steps of the authors' code (`2 - clean.do`): finished and consenting
  participants, the six manual exclusions, both quality-check waves, and participants graded before
  they were tasked. Grades are exact means of the eligible original grades. The analysed groups are the
  linear and convex incentive groups (352 participants for the grade of the second task, 353 for its time).
