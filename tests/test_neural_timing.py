from types import SimpleNamespace
import numpy as np

from r_elegans.connection_amplitude import amplitude_score
from r_elegans.neural_timing import timing_score,propose_timing


def trace(frequency, amplitude=.01):
    t=np.linspace(0,60,6001)
    return dict(dense_time_s=t,dense_joints=amplitude*np.sin(2*np.pi*frequency*t)[:,None]*np.ones((1,11)),
                dense_voltage_mV=5*np.sin(2*np.pi*frequency*t)[:,None])


def test_slowing_allowed_without_losing_rhythm_or_body_coverage():
    baseline=amplitude_score(trace(2.6),0,window_seconds=15)
    slow=timing_score(trace(1.8),0,target_hz=1.8,window_seconds=15,baseline=baseline)
    fast=timing_score(trace(2.6),0,target_hz=1.8,window_seconds=15,baseline=baseline)
    assert slow['neural_guard'] and slow['target_passed']
    assert slow['data_loss']<fast['data_loss']
    # Faster neuronal rhythms can coexist with a slow gait; don't penalize
    # every neuron for differing from the body-undulation frequency.
    independent=trace(1.8)
    independent['dense_voltage_mV']=trace(2.6)['dense_voltage_mV']
    actual_body=timing_score(independent,0,target_hz=1.8,window_seconds=15,baseline=baseline)
    assert actual_body['target_passed']
    quiet=timing_score(trace(1.8,1e-6),0,target_hz=1.8,window_seconds=15,baseline=baseline)
    assert not quiet['safety_passed'] and not quiet['target_passed']


def test_all_proposal_modes_obey_original_anchor_bounds_and_leave_masked_densities_out():
    channels=np.array([[False,True,True],[False,True,True]])
    spec=SimpleNamespace(channels=channels,chemical=np.array([[False,True],[True,False]]),gap=np.array([[False,True],[True,False]]))
    rng=np.random.default_rng(6800)
    lower=np.full(7,-.69);upper=np.full(7,.69);x=np.array([.68]*7)
    kinds=set()
    for _ in range(100):
        candidate,description=propose_timing(x,spec,rng,.2,lower,upper,[0,1])
        kinds.add(description['kind'])
        assert candidate.shape==(7,) and np.all(candidate>=lower) and np.all(candidate<=upper)
    assert kinds=={'connections','densities_or_joint'}
