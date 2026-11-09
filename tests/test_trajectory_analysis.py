import json
import numpy as np
import pytest

from r_elegans.trajectory_analysis import analyze_trajectory


def trace(speed=0.,traveling=False,static=False):
    dt=.002;t=np.arange(0,30,dt);period=.25 if traveling else 5./23.05
    wave=np.sin(2*np.pi*t/period)
    v=-50+10*np.tile(wave[:,None],(1,3));a=np.zeros((len(t),11))
    q=np.zeros_like(a)
    if traveling:
        q=.015*np.sin(2*np.pi*t[:,None]/period-np.linspace(0,np.pi,11))
        q[:,-1]*=1.3
    elif not static:q[:,-3:]=wave[:,None]*np.array([.005,.01,.015])
    if static:v[:]=-50
    position=np.column_stack((speed*t+(.003*np.sin(2*np.pi*t/period-.15*2*np.pi) if not static else 0.),np.zeros_like(t)))
    return analyze_trajectory(v,q,a,position,np.zeros_like(t),dt=dt,body_length=1.,neuron_ids=('RMDL','RMDR','RMDVL'))


def test_equal_phase_drift_recovers_translation_without_rewarding_bounded_oscillation():
    periodic,_=trace()
    assert min(periodic['raw_forward_block_speeds_body_lengths_s'])>1e-4
    assert abs(periodic['equal_phase_regression_speed_body_lengths_s'])<1e-7
    translating,_=trace(speed=.001)
    assert translating['equal_phase_regression_speed_body_lengths_s']==pytest.approx(.001,abs=1e-7)
    assert translating['cycles']>50
    assert translating['median_shape_closure_rms_rad']<1e-8


def test_spatial_analysis_distinguishes_single_pattern_from_phase_shifted_shape():
    standing,_=trace()
    assert standing['anterior_three_joint_variance_fraction']==pytest.approx(1.)
    assert standing['shape_pca_variance_ratios'][0]>1.-1e-10
    moving,_=trace(traveling=True)
    assert moving['shape_pca_variance_ratios'][0]<.7
    assert len(moving['spectral_phase'])==11
    assert min(x['coherence'] for x in moving['spectral_phase'])>.99


def test_static_control_has_no_cycles_or_nonfinite_report():
    report,_=trace(static=True)
    assert report['cycles']==0 and report['dominant_frequency_hz'] is None
    assert report['equal_phase_regression_speed_body_lengths_s'] is None
    json.dumps(report,allow_nan=False)
