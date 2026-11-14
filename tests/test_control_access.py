from types import SimpleNamespace
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from r_elegans.control_access import cases,membrane_probe,compare
from r_elegans.behavior_control import CONTROL_GROUPS


def test_grades_targets_and_time_window_are_explicit_and_jittable():
    names=list(dict.fromkeys(n for group in CONTROL_GROUPS.values() for n in group))+['DB01','AWAL']
    assert len(cases())==17
    for case in cases():
        probe=jax.jit(membrane_probe(names,case))
        for t in (0.,29.99,40.,59.):np.testing.assert_array_equal(probe(t).conductance_nS,0)
        actual=probe(35.)
        expected=np.array([case['conductance_nS'] if n in CONTROL_GROUPS.get(case['group'],()) else 0. for n in names])
        np.testing.assert_allclose(actual.conductance_nS,expected)
        np.testing.assert_allclose(actual.weighted_reversal_pA,expected*case['reversal_mV'])
    with pytest.raises(ValueError):membrane_probe(names,dict(group='stop_candidate',conductance_nS=1.,reversal_mV=0))


def fixture(kind):
    t=np.arange(0,60.01,.01);q=.01*np.sin(2*np.pi*1.8*t)[:,None]*np.ones((1,11))
    if kind in ('reversible','stuck'):q[(t>=30)&(t<(40 if kind=='reversible' else 61))]=0
    if kind=='pre_changed':q[t<30]*=.9
    return dict(dense_time_s=t,dense_joints=q,dense_voltage_mV=np.zeros((len(t),3)))


def test_probe_success_requires_quiet_motion_then_recovery_and_same_baseline():
    sham=fixture('moving')
    good=compare(fixture('reversible'),sham)
    assert good['reversible_diagnostic_suppression'] and not good['learned_stopping_qualified']
    assert not compare(sham,sham)['reversible_diagnostic_suppression']
    assert not compare(fixture('stuck'),sham)['reversible_diagnostic_suppression']
    assert not compare(fixture('pre_changed'),sham)['pre_cue_match']
    fast=fixture('reversible');use=fast['dense_time_s']>=40
    fast['dense_joints'][use]=.01*np.sin(2*np.pi*3.6*fast['dense_time_s'][use])[:,None]
    assert not compare(fast,sham)['reversible_diagnostic_suppression']
    malformed=fixture('moving');malformed['dense_time_s']+=.001
    with pytest.raises(ValueError):compare(malformed,sham)


def test_probe_combines_with_sensing_without_freezing_brain_or_replacing_muscles(monkeypatch):
    import r_elegans.body.closed_loop as loop
    from r_elegans.brain.conductance_network import MembraneConductanceInput
    brain=SimpleNamespace(voltage_mV=jnp.array([-40.,-60.]))
    state=SimpleNamespace(sensory='memory',brain_body=SimpleNamespace(brain=brain,body='body'))
    sensory=MembraneConductanceInput(jnp.array([.1,.2]),jnp.array([0.,-16.]))
    diagnostic=MembraneConductanceInput(jnp.array([.01,0]),jnp.array([0.,0.]))
    seen=[]
    monkeypatch.setattr(loop,'transduce_sensory_signals',lambda *a,**k:('updated_memory',sensory,'observation'))
    def advance(*args,**kwargs):seen.append(kwargs['membrane_input']);return brain
    monkeypatch.setattr(loop,'advance_network',advance)
    monkeypatch.setattr(loop,'channel_muscle_activations',lambda *a:'normal_NMJ_result')
    ordinary=loop.embodied_neural_step(state,None,None,None,None,None,None,None,None,.002,20)
    probed=loop.embodied_neural_step(state,None,None,None,None,None,None,None,None,.002,20,diagnostic_membrane_input=diagnostic)
    np.testing.assert_allclose(seen[0].conductance_nS,[.1,.2])
    np.testing.assert_allclose(seen[1].conductance_nS,[.11,.2])
    assert ordinary[0] is brain and probed[0] is brain
    assert probed[1:4]==('updated_memory','normal_NMJ_result','observation')
