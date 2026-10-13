"""Published result records have the current semantics and match representative small reruns."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RES = REPO / 'results'
MAIN_CALIBRATION = 'centering_comparison/full_integer/calibration/centre10'
ROWS = {MAIN_CALIBRATION: 3600, 'calibration/two_regions': 3600,
        'supplementary/rates': 3000, 'supplementary/stress': 9000,
        'supplementary/dispersed': 400, 'supplementary/two_sided': 4500,
        'supplementary/shared_vs_conditional': 400, 'supplementary/effect_size': 2400,
        'timing/warm_start': 160}


@pytest.mark.parametrize('folder', sorted(ROWS))
def test_results_complete(folder):
    df = pd.read_csv(RES / folder / 'details.csv')
    assert len(df) == ROWS[folder]
    assert (RES / folder / 'run_meta.json').exists()
    identity = [k for k in ('dim', 'variant', 'n', 'rep') if k in df]
    if 'rep' in identity:
        assert not df.duplicated(identity).any()


@pytest.mark.parametrize('dim', [1, 2])
def test_main_calibration_reproduced(dim, tmp_path):
    from runners import run_calibration as C
    task = (dim, 200, 0)
    rec = C.run_task(task, 'centre10', tmp_path)
    df = pd.read_csv(RES / MAIN_CALIBRATION / 'details.csv')
    stored = df[(df.dim == dim) & (df.n == 200) & (df.rep == 0)].iloc[0].to_dict()
    C.validate_record(stored, 'centre10', 'all', task)
    for field in ('K0', 'input_sha256', 'raw_bound', 'adjusted_bound',
                  'raw_certificate', 'adjusted_certificate', 'raw_covers', 'adjusted_covers'):
        assert rec[field] == stored[field], field
    assert (rec['raw_bound'], rec['adjusted_bound']) == {1: (106, 43), 2: (158, 46)}[dim]


@pytest.mark.parametrize('dim', [1, 2])
def test_two_regions_centred_original_outcome_reproduced(dim, tmp_path):
    from runners import run_two_regions as T
    rec = T.run_task((dim, 200, 0), 'two_regions', tmp_path)
    df = pd.read_csv(RES / 'calibration/two_regions/details.csv')
    stored = df[(df.dim == dim) & (df.n == 200) & (df.rep == 0)].iloc[0]
    assert rec['raw_statistic'] == 'within_region_centered_covariance'
    for field in ('K0', 'raw_bound', 'adjusted_bound', 'raw_covers', 'adjusted_covers'):
        assert rec[field] == stored[field], field
    assert rec['K0'] == 40


@pytest.mark.parametrize('task,fields', [
    (('stress', 1, 'S3', 200, 4), ['L0', 'raw_bound', 'indep_bound', 'insample_bound',
                                  'raw_desc_bound', 'raw_bis_bound']),
    (('rates', 1, 'centre10-grad1', 500, 0), ['K0', 'Khat', 'covers', 'mc_sizes']),
    (('two_sided', 1, 'asym', 200, 1), ['L0', 'one_sided_bound', 'two_sided_bound']),
])
def test_supplementary_reproduced(task, fields):
    from runners import run_supplementary as R
    rec = R.run_task(task)
    df = pd.read_csv(RES / 'supplementary' / task[0] / 'details.csv')
    row = df[(df.variant == task[2]) & (df.n == task[3]) & (df.rep == task[4])].iloc[0]
    for field in fields:
        assert rec[field] == row[field], field


def test_real_data_summaries_match_the_paper():
    def summary(name):
        return json.loads((RES / 'real_data' / name / 'summary.json').read_text())
    ggl, actg, noy = summary('ggl'), summary('actg'), summary('noy')
    assert (ggl['n'], ggl['bound'], ggl['first_rejected'], ggl['covered_units']) == (119999, 4286, 4287, 4286)
    assert [actg['outcomes'][s]['bound'] for s in ('raw', 'change')] == [134, 149]
    assert [noy['outcomes'][s]['bound'] for s in ('quality', 'time')] == [90, 65]
    two = json.loads((RES / 'real_data/actg/two_covariates.json').read_text())
    assert [two[s]['bound'] for s in ('raw', 'change')] == [228, 294]
    for result in [ggl] + list(actg['outcomes'].values()) + list(noy['outcomes'].values()) + [two[s] for s in ('raw', 'change')]:
        assert result['certificate'] == result['bound']
        assert result['first_rejected'] == result['bound'] + 1
    robustness = pd.read_csv(RES / 'real_data/ggl/robustness.csv')
    assert len(robustness) == 15
    assert robustness['bound'].min() == 3085 and robustness['bound'].max() == 4306
