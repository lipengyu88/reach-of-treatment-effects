"""The files in results/ are complete and are reproduced by the runners (a few small tasks)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'runners'))
RES = REPO / 'results'

ROWS = {'calibration/centre10': 3600, 'calibration/two_regions': 3600, 'supplementary/rates': 3000,
        'supplementary/stress': 9000, 'supplementary/dispersed': 400, 'supplementary/two_sided': 4500,
        'supplementary/shared_vs_conditional': 400, 'supplementary/effect_size': 2400, 'timing/warm_start': 160}


@pytest.mark.parametrize('folder', sorted(ROWS))
def test_results_complete(folder):
    df = pd.read_csv(RES / folder / 'details.csv')
    assert len(df) == ROWS[folder]
    assert (RES / folder / 'run_meta.json').exists()


@pytest.mark.parametrize('task', [(1, 200, 0), (2, 200, 1), (1, 1000, 2)])
def test_calibration_reproduced(task):
    import run_calibration as RC
    rec = RC.run_task(task, 'centre10', RES)
    df = pd.read_csv(RES / 'calibration/centre10/details.csv')
    row = df[(df.dim == task[0]) & (df.n == task[1]) & (df.rep == task[2])].iloc[0].to_dict()
    for key, value in rec.items():
        assert row[key] == value, key


@pytest.mark.parametrize('task', [('stress', 1, 'S3', 200, 4), ('rates', 1, 'centre10-grad1', 500, 0),
                                  ('two_sided', 1, 'asym', 200, 1), ('shared_vs_conditional', 1, 'centre10-const', 1000, 0)])
def test_supplementary_reproduced(task):
    import run_supplementary as RS
    rec = RS.run_task(task)
    df = pd.read_csv(RES / 'supplementary' / task[0] / 'details.csv')
    row = df[(df.variant == task[2]) & (df.n == task[3]) & (df.rep == task[4])].iloc[0].to_dict()
    for key, value in rec.items():
        if key in ('cpu_seconds', 'wall_seconds', 'search_cpu', 'shared_cpu', 'conditional_cpu'):
            continue
        if value is None:
            assert row[key] is None or (isinstance(row[key], float) and np.isnan(row[key])), key
        else:
            assert row[key] == value, key


def test_real_data_summaries_match_the_paper():
    ggl = json.loads((RES / 'real_data/ggl/summary.json').read_text())
    actg = json.loads((RES / 'real_data/actg/summary.json').read_text())
    noy = json.loads((RES / 'real_data/noy/summary.json').read_text())
    assert (ggl['n'], ggl['bound'], ggl['first_rejected'], ggl['covered_units']) == (119999, 3388, 4686, 4876)
    assert [actg['outcomes'][s]['bound'] for s in ('raw', 'change')] == [106, 106]
    assert [noy['outcomes'][s]['bound'] for s in ('quality', 'time')] == [90, 54]
