import numpy as np
import pytest
from scripts.diagnose_cycle_sampling import summarize


def test_localizes_cycle_and_measured_pair_without_changing_gate():
    y=np.zeros((16,2,3));y[:4,1,2]=1.
    measured=np.zeros((2,3),dtype=bool);measured[1,2]=True
    row=summarize(y,['T0','T1'],['T0','R1','R2'],measured)
    assert row['pairs_above_tolerance']==row['measured_pairs_above_tolerance']==1
    assert row['worst']['omitted_cycle']==0 and row['worst']['target']=='T1' and row['worst']['receiver']=='R2'
    assert row['worst']['cycle_means']==[1.,0.,0.,0.]
    assert row['original_screen']['leave_one_cycle_out_max_difference']==.25
    assert not row['original_screen']['passed']


def test_stable_cycles_and_self_response_support_are_explicit():
    y=np.ones((16,1,1))
    row=summarize(y,['RMDL'],['RMDL'])
    assert row['original_screen']['passed'] and row['pairs_above_tolerance']==0
    assert row['worst']['self_response'] and row['worst']['measured_in_development_pool'] is None
    with pytest.raises(ValueError):summarize(y[:4],['RMDL'],['RMDL'])
