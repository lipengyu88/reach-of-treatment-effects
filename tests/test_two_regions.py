"""The appendix's centred/OLS branches cannot be confused with the main centering comparison."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runners import run_calibration as C, run_two_regions as T
from zeroreach import designs as D, search as S
from zeroreach.grids import envelope_bound


@pytest.mark.parametrize('dim', [1, 2])
@pytest.mark.parametrize('grid', ['all', 'paper'])
def test_two_regions_matches_independent_enumeration(dim, grid, tmp_path):
    n=40
    d=D.generate(dim,n,0,zero='two_regions',M=T.M)
    orders=S.knn_orders(d['W']) if dim==2 else None
    rec=T.run_task((dim,n,0),'two_regions',tmp_path,display=True,grid=grid)
    sizes=list(range(1,n+1)) if grid=='all' else d['grid']
    for name,y in [('raw',d['y']),('adjusted',d['adj_indep'])]:
        ev=S.Shared1D(y,d['z'],d['draws']) if dim==1 else S.Shared2D(y,d['z'],d['draws'],orders)
        path={k:ev.max_p(k) for k in sizes}
        bound,first=envelope_bound(path,sizes,T.ALPHA)
        assert rec[name+'_bound']==bound
        assert rec[name+'_first_rejected']==first
    assert rec['K0']==8 and rec['zero_count']==12
    assert rec['raw_statistic']=='within_region_centered_covariance'
    assert rec['adjusted_statistic']=='independent_ols_residual_centered_covariance'
    with np.load(tmp_path/f'display_d{dim}_n{n}_rep0.npz') as display:
        assert str(display['raw_statistic'])==rec['raw_statistic']
        assert str(display['adjusted_statistic'])==rec['adjusted_statistic']


def test_experiment_scopes_do_not_alias():
    with pytest.raises(ValueError,match='run_two_regions'):
        C.run_config('two_regions','all')
    with pytest.raises(ValueError,match='run_calibration'):
        T.run_config('centre10','all')
    with pytest.raises(ValueError,match='Incompatible existing record'):
        C.validate_record(T.run_config('two_regions','all'),'centre10','all')
