import numpy as np
from r_elegans.alternation import alternation_statistics,pattern_resolution_check


def trace(signal):
    t=np.linspace(0,12,6001)
    q=np.full((len(t),11),.2);q[:,4]+=signal(t)
    return dict(dense_time_s=t,dense_joints=q)


def test_sustained_alternation_rejects_flat_ramp_decay_and_one_bend():
    wave=trace(lambda t:2e-5*np.sin(2*np.pi*1.2*t))
    result=alternation_statistics(wave)
    assert result['pattern_pass'] and result['alternating_joint']==4
    assert result['peaks']>=2 and result['troughs']>=2
    assert result['window_amplitude_ratio']>.9
    for bad in [trace(lambda t:0*t),trace(lambda t:1e-4*t),
                trace(lambda t:1e-3*np.exp(-t)*np.sin(2*np.pi*t)),
                trace(lambda t:2e-5*np.exp(-((t-6)/.5)**2))]:
        assert not alternation_statistics(bad)['pattern_pass']


def test_pattern_convergence_tracks_same_joint_and_turn_counts():
    a=alternation_statistics(trace(lambda t:2e-5*np.sin(2*np.pi*1.2*t)))
    b=alternation_statistics(trace(lambda t:2.01e-5*np.sin(2*np.pi*1.2*t+.01)))
    assert pattern_resolution_check(a,b,1e-5)['passed']
    c=alternation_statistics(trace(lambda t:2e-5*np.sin(2*np.pi*2.4*t)))
    assert not pattern_resolution_check(a,c,1e-5)['passed']
