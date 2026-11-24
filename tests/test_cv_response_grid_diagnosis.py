import numpy as np
import pytest
from scripts.diagnose_cv_response_grids import compare


def test_phase_averaging_can_cancel_discrepancy_without_reclassifying_gate():
    b=np.full((4,2,3),.001);a=b.copy();a[:,0,1]+=[.001,-.001,.001,-.001]
    r=compare(a,b,['T0','T1'],['T0','R1','R2'],[['T0','R1']])
    assert r['original_tolerance_ratio']>1
    assert r['phase_mean_max_difference']==0
    assert r['worst']['in_training_objective']
    assert 'passed' not in r


def test_worst_response_outside_training_objective_is_explicit():
    b=np.zeros((4,2,3));a=b.copy();a[2,1,2]=.2;a[1,0,1]=.01
    r=compare(a,b,['T0','T1'],['T0','R1','R2'],[['T0','R1']])
    assert r['worst']==dict(phase=2,target='T1',receiver='R2',in_training_objective=False,coarse=.2,fine=0.)
    assert r['training_observed']['phasewise_max_difference']==.01
    assert r['outside_training_objective']['phasewise_max_difference']==.2


def test_invalid_comparisons_refused():
    with pytest.raises(ValueError):compare(np.zeros((3,2,2)),np.zeros((3,2,2)),['T','U'],['R','S'],[])
