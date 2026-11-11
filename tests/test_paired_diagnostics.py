import numpy as np
import pytest
from r_elegans.paired_diagnostics import bound_occupancy,dynamic_range


def test_bound_report_distinguishes_nonzero_from_near_bounds():
    limit=np.log(2.)
    report=bound_occupancy([0,.1,.999*limit,-limit],{'channels':2,'edges':2},2.)
    assert report['categories']['channels']['nonzero']==1
    assert report['categories']['channels']['near_upper']==0
    assert report['categories']['edges']['near_upper']==1
    assert report['categories']['edges']['near_lower']==1
    with pytest.raises(ValueError,match='exceed'):bound_occupancy([limit+.01],{'x':1},2.)


def test_sustained_signal_distinguishes_offset_ramp_and_decay():
    t=np.arange(0,20,.01)
    signal=np.column_stack((5+np.sin(2*np.pi*2*t),.2*t,
                            np.exp(-t)*np.sin(2*np.pi*2*t),np.ones(len(t))))
    report=dynamic_range(signal,.01)
    assert report['repeated_indices']==[0]
    assert report['sustained_p2p'][0]>1.9
    assert report['sustained_p2p'][1:]==[0.,0.,0.]
    assert report['late_detrended_std'][1]<1e-10
    assert report['late_linear_slope_per_second'][1]==pytest.approx(.2)
