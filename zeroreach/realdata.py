"""Readers for the three experiments of Sections 6.7-6.9 and the exact count kernel used for the
Gerber-Green-Larimer analysis. The data are not part of this repository; see data/README.md.

  load_ggl(path)     household-level Neighbors versus Control data from the individual file of ISPS D001;
  load_actg(path)    ACTG 175 arms 0 and 1, ordered by baseline CD4 count (ties by patient id);
  load_noy(raw_dir)  the Noy-Zhang writing experiment, rebuilt from the public replication files.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np
from numba import njit, prange

# ------------------------------------------------------------------------------ Gerber-Green-Larimer
GGL_AGE_YEAR = 2006


def load_ggl(path):
    """Households of the Control (arm 0) and Neighbors (arm 3) groups, with exact integer keys.

    With L the least common multiple of the household sizes, yint = L * (household turnout rate in 2006),
    age_key = L * (mean age in 2006 of the members) and turnout_key = L * (mean number of the 2000 and 2002
    elections voted in). Returns (households, L)."""
    import pandas as pd
    d = pd.read_stata(path, convert_categoricals=False)
    for c in ('hh_id', 'cluster', 'treatment', 'voted', 'g2000', 'g2002', 'yob'):
        if d[c].isna().any():
            raise ValueError('missing values in ' + c)
    d['age'] = GGL_AGE_YEAR - d['yob'].astype(np.int64)
    hh = d.groupby('hh_id').agg(cell_id=('cluster', 'first'), cell_n=('cluster', 'nunique'),
                                arm=('treatment', 'first'), arm_n=('treatment', 'nunique'),
                                members=('hh_id', 'size'), vsum=('voted', 'sum'),
                                g0=('g2000', 'sum'), g2=('g2002', 'sum'), agesum=('age', 'sum'),
                                agemin=('age', 'min'), agemax=('age', 'max')).reset_index()
    if not ((hh.cell_n == 1).all() and (hh.arm_n == 1).all()):
        raise ValueError('every household must lie in one cell and one arm')
    sub = hh[hh.arm.isin([0, 3])].copy().reset_index(drop=True)
    sub['z'] = (sub.arm == 3).astype(np.int8)
    L = math.lcm(*(int(s) for s in sub.members.unique()))
    sub['turnout_key'] = ((sub.g0 + sub.g2) * (L // sub.members)).astype(np.int64)
    sub['age_key'] = (sub.agesum * (L // sub.members)).astype(np.int64)
    sub['yint'] = (sub.vsum * (L // sub.members)).astype(np.int64)
    return sub, L


@njit(parallel=True, cache=True)
def int_window_counts(y, zobs, draws, grid, nchunk):
    """Exceedance counts of the centred statistic for every block of every grid size, in exact int64
    arithmetic (integer-valued y). Memory is O(n) per thread, so n can be large."""
    Mloc, n = draws.shape
    G = grid.size
    Py = np.zeros(n + 1, np.int64)
    Po = np.zeros(n + 1, np.int64)
    Pz = np.zeros(n + 1, np.int64)
    for i in range(n):
        Py[i + 1] = Py[i] + y[i]
        Pz[i + 1] = Pz[i] + zobs[i]
        Po[i + 1] = Po[i] + zobs[i] * y[i]
    cnt = np.zeros((nchunk, G, n), np.int32)
    for ch in prange(nchunk):
        Fz = np.zeros(n + 1, np.int64)
        Fy = np.zeros(n + 1, np.int64)
        for m in range(ch * Mloc // nchunk, (ch + 1) * Mloc // nchunk):
            for i in range(n):
                Fz[i + 1] = Fz[i] + draws[m, i]
                Fy[i + 1] = Fy[i] + draws[m, i] * y[i]
            for g in range(G):
                r = grid[g]
                for s in range(n - r + 1):
                    e = s + r
                    ysum = Py[e] - Py[s]
                    tobs = r * (Po[e] - Po[s]) - ysum * (Pz[e] - Pz[s])
                    t = r * (Fy[e] - Fy[s]) - ysum * (Fz[e] - Fz[s])
                    if t >= tobs:
                        cnt[ch, g, s] += 1
    total = np.zeros((G, n), np.int64)
    for g in prange(G):
        for s in range(n):
            acc = 0
            for ch in range(nchunk):
                acc += cnt[ch, g, s]
            total[g, s] = acc
    return total


# --------------------------------------------------------------------------------------------- ACTG 175
def load_actg(path):
    """Arms 0 and 1 of ACTG 175 ordered by baseline CD4 count, ties by patient id. Returns the frame, the
    treatment indicator (arm 1) and the randomization strata."""
    import pandas as pd
    raw = pd.read_csv(path)
    sub = raw.loc[raw.trt.isin([0, 1]), ['trt', 'pidnum', 'cd40', 'cd420', 'strat', 'age']].copy()
    sub = sub.sort_values(['cd40', 'pidnum'], kind='mergesort').reset_index(drop=True)
    z = (sub.trt == 1).to_numpy().astype(np.int8)
    return sub, z, sub.strat.to_numpy()


# ---------------------------------------------------------------------------------------- Noy-Zhang
# participants excluded by the authors' cleaning code ("2 - clean.do"), identified by their hashed ids
NOY_MANUAL_EXCLUSIONS = {
    '484d7e16cdcac3774e045744722ecd4610f01d40069b05c9affb00499e8f3731',
    '727c882101c4438387011b92398c999d46e76b0f3c59bb45b17e6ea1c64e4561',
    '8b0833b266be60de6278a28d4ae8052236ecac85b248710ac6efaf57420ee11a',
    '174207d4312d46b6b908d28dc34c9f6dbe62d22844b6b9b8b875ac4de272b275',
    'f9a1e36ff890a83c3dc1634403f02e0d250e66cf08218f197b7127f2993892ec',
    'bee7a8f1c3bdc0c2a6e6303ab33a8253a69894569927a633222bd5e3a6d244ff',
}
NOY_TEST_GRADER = '043a718774c572bd8a25adbeb1bfcd5c0256ae11cecf9f9c3f925d0e52beaf89'


def _read_qcheck(path):
    """Quality-check sheet of a grading workbook (first sheet), read without an Excel dependency."""
    import pandas as pd
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    with zipfile.ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            for si in ET.fromstring(archive.read('xl/sharedStrings.xml')):
                strings.append(''.join(t.text or '' for t in si.iter(ns + 't')))
        sheets = sorted(x for x in archive.namelist() if x.startswith('xl/worksheets/sheet') and x.endswith('.xml'))
        rows = []
        for row in ET.fromstring(archive.read(sheets[0])).iter(ns + 'row'):
            cells = {}
            for cell in row.iter(ns + 'c'):
                col = ''.join(c for c in cell.get('r', '') if c.isalpha())
                v = cell.find(ns + 'v')
                if v is not None:
                    cells[col] = strings[int(v.text)] if cell.get('t') == 's' else v.text
                elif cell.find(ns + 'is') is not None:
                    cells[col] = ''.join(t.text or '' for t in cell.iter(ns + 't'))
            rows.append(cells)
    headings = {col: str(value).strip().lower() for col, value in rows[0].items()}
    result = pd.DataFrame([{headings.get(k, k): v for k, v in row.items()} for row in rows[1:]])
    result['qcheck'] = pd.to_numeric(result.qcheck, errors='raise').replace({3: 1})
    return result.loc[~result.prolific_pid.isin(['', '1', '2', '3', '4']), ['prolific_pid', 'qcheck']].dropna(
        subset=['qcheck'])


def _rational(value):
    import pandas as pd
    if pd.isna(value) or str(value).strip() in ('', '.'):
        return None
    return Fraction(str(value).strip())


def scale_exact(values):
    """Multiply exact rational scores by their common denominator: the integers keep every p-value."""
    fractions = [Fraction(x) for x in values]
    denominator = math.lcm(*(x.denominator for x in fractions))
    integers = [x.numerator * (denominator // x.denominator) for x in fractions]
    if any(int(float(x)) != x for x in integers):
        raise ValueError('scaled score exceeds the exact range of float64')
    return np.asarray(integers, dtype=np.float64), denominator


def _grade_means(grading, task_dates):
    """Mean of the eligible original (non-retry) integer grades of each essay."""
    import pandas as pd
    g = grading.copy()
    g['dategraded'] = pd.to_datetime(g.startdate, errors='raise').dt.normalize()
    ineligible = g.dategraded.gt(g.grader_id.map(task_dates))
    keep = g.retry.ne(1) & g.grader_id.ne(NOY_TEST_GRADER) & ~ineligible & g.overall.notna()
    g = g.loc[keep].copy()
    grades = pd.to_numeric(g.overall, errors='raise')
    if not grades.between(1, 7).all() or not np.equal(grades, np.floor(grades)).all():
        raise ValueError('grades must be integers 1 to 7')
    g['overall'] = grades.astype(np.int64)
    sums = g.groupby(['prolific_pid', 'essaynum']).overall.agg(['sum', 'count'])
    return {(pid, essay): Fraction(int(row['sum']), int(row['count'])) for (pid, essay), row in sums.iterrows()}


def load_noy(raw_dir, expected_n=453):
    """Participants of the linear and convex incentive groups (Section 6.9), one cohort per outcome.

    W is the exact mean grade of the first task; 'quality' is the exact mean grade of the second task and
    'time' minus the time spent on the second task. Participants are ordered by (W, participant id).
    Returns {'quality': frame, 'time': frame}; each frame has columns W_exact, <outcome>_exact, treatment."""
    import pandas as pd
    raw = Path(raw_dir)
    files = [raw / 'fullsurvey.csv', raw / 'grading_long_complete.csv', raw / 'grading_retry.xlsx', raw / 'grading.xlsx']
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError('missing Noy-Zhang files: ' + ', '.join(missing))
    f = pd.read_csv(files[0], encoding='ISO-8859-1', dtype=str, low_memory=False)
    f.columns = f.columns.str.lower()
    for col in ('finished', 'consent', 'consent_exact', 'task_realism', 'treatment', 'a_first'):
        f[col] = pd.to_numeric(f[col], errors='raise')
    f = f.loc[f.finished.eq(1) & f.consent.ne(2) & f.consent_exact.ne(2)].copy()
    f['datetasked'] = pd.to_datetime(f.startdate, errors='raise').dt.normalize()
    task_dates = f.groupby('prolific_pid').datetasked.min()
    f = f.sort_values(['prolific_pid', 'startdate'], kind='mergesort').drop_duplicates('prolific_pid')
    f = f.loc[~f.prolific_pid.isin(NOY_MANUAL_EXCLUSIONS)].copy()
    # quality checks of both grading waves; the first wave wins for a participant graded twice
    qs = [q.groupby('prolific_pid').qcheck.agg(minq='min', maxq='max').reset_index()
          for q in (_read_qcheck(files[3]), _read_qcheck(files[2]))]
    q = pd.concat(qs).drop_duplicates('prolific_pid', keep='first')
    f = f.merge(q, on='prolific_pid', how='left', validate='one_to_one')
    f = f.loc[~(f.maxq.gt(1) | f.minq.lt(1))].copy()
    g = pd.read_csv(files[1], low_memory=False)
    g.columns = g.columns.str.lower()
    gdates = pd.to_datetime(g.startdate, errors='raise').dt.normalize().groupby(g.grader_id).min()
    f = f.loc[~f.datetasked.gt(f.prolific_pid.map(gdates))].copy()
    if expected_n is not None and len(f) != expected_n:
        raise ValueError(f'full-sample count {len(f)} differs from {expected_n}')
    means = _grade_means(g, task_dates)
    f['W_exact'] = [means.get((r.prolific_pid, 'A' if r.a_first == 1 else 'B')) for r in f.itertuples()]
    f['quality_exact'] = [means.get((r.prolific_pid, 'B' if r.a_first == 1 else 'A')) for r in f.itertuples()]
    times = []
    for r in f.itertuples():
        parts = [_rational(getattr(r, f'time2_{j}')) for j in (4, 5, 6, 7)]
        times.append(None if any(x is None for x in parts) else -sum(parts, Fraction(0)))
    f['time_exact'] = times
    main = f.loc[f.incentive_arm.isin(['linear', 'convex'])].copy()
    cohorts = {}
    for score in ('quality', 'time'):
        cohort = main.loc[main.W_exact.notna() & main[f'{score}_exact'].notna()].copy()
        order = sorted(range(len(cohort)), key=lambda i: (cohort.iloc[i].W_exact, cohort.iloc[i].prolific_pid))
        cohorts[score] = cohort.iloc[order].reset_index(drop=True)
    return cohorts
