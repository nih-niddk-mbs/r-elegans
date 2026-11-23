"""Scientific leakage, checked descent, state propagation and audit failure tests."""
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.run_cv_peptide_pilot import fold_statistics,train_choice,audit,write


def test_fold_preprocessing_is_animal_disjoint_and_heldout_values_cannot_change_train():
    def rows():
        return [dict(subject=f'w{i}',trials=[dict(subject=f'w{i}',target='A',z={'B':float(i)})]) for i in range(25)]
    original=rows();result=fold_statistics(original)
    held=set(result['subjects']['heldout_subjects'])
    changed=rows()
    for r in changed:
        if r['subject'] in held:r['trials'][0]['z']['B']=100000.
    other=fold_statistics(changed)
    assert result['subjects']==other['subjects']
    assert not held & set(result['subjects']['train_subjects'])
    np.testing.assert_array_equal(result['stats']['train']['mean_z'],other['stats']['train']['mean_z'])
    assert result['stats']['heldout']['mean_z']!=other['stats']['heldout']['mean_z']
    assert sorted(s for f in result['fivefold_membership']['folds'] for s in f['heldout_subjects'])==sorted(r['subject'] for r in original)
    with pytest.raises(ValueError,match='Repeated'):fold_statistics(original+[original[0]])


def test_selection_uses_actual_safe_training_descent_only():
    current=dict(name='base',safe=True,train_loss=1.,heldout_loss=0.)
    better=dict(name='better',safe=True,train_loss=.9,heldout_loss=100.)
    uphill=dict(name='uphill',safe=True,train_loss=1.1,heldout_loss=-100.)
    unsafe=dict(name='unsafe',safe=False,train_loss=.1)
    assert train_choice(current,[better,uphill,unsafe])==better
    assert train_choice(current,[uphill,unsafe])==current
    with pytest.raises(ValueError,match='Unsafe'):train_choice(unsafe,[better])


def test_missing_execution_cannot_be_a_successful_audit(tmp_path):
    r=audit(tmp_path)
    assert r['audit_complete'] and not r['execution_complete'] and not r['integrity_passed']
    assert not r['long_run_ready'] and r['unavailable_evidence']
    write(tmp_path/'numpy.json',dict(array=np.arange(2),boolean=np.bool_(True)))
    assert json.loads((tmp_path/'numpy.json').read_text())==dict(array=[0,1],boolean=True)


def test_peptide_memory_survives_sham_and_changes_neural_response(monkeypatch):
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace
    import r_elegans.brain.peptide_state_replay as module
    from r_elegans.brain.conductance_network import ConductanceNetworkState
    # Tiny passive receiver provides an exact causal test of the state plumbing.
    def initial(f,c):
        return ConductanceNetworkState(jnp.array([-35.]),jnp.zeros((1,35)),jnp.zeros(1),jnp.zeros(1))
    def advance(s,p,f,dt,steps,c,*,membrane_input):
        v=s.voltage_mV+dt*(membrane_input.weighted_reversal_pA-membrane_input.conductance_nS*s.voltage_mV)
        return s._replace(voltage_mV=v,indicator=v)
    monkeypatch.setattr(module,'initial_network_state',initial)
    monkeypatch.setattr(module,'advance_network',advance)
    fixed=SimpleNamespace(release_threshold_mV=jnp.array([-45.]),release_slope_mV=jnp.array([10.]))
    # SimpleNamespace is not a JAX pytree: use the real fixed tuple shape via closure.
    from typing import NamedTuple
    class F(NamedTuple):
        release_threshold_mV:object
        release_slope_mV:object
    fixed=F(fixed.release_threshold_mV,fixed.release_slope_mV)
    _,replay=module.make_engine(SimpleNamespace(synaptic_unit_nS=.03),[0],dt=.1,step_ms=100.)
    neural=jax.tree.map(lambda x:jnp.stack([x]*4),initial(None,None))
    start=module.FullState(neural,jnp.full((4,1),.7))
    exc,inh=jnp.ones((1,1)),jnp.zeros((1,1))
    means,final,traces=replay(None,fixed,exc,inh,jnp.log(5.),start,jnp.zeros((2,1)))
    assert np.all(np.asarray(final.peptide)>0)
    np.testing.assert_array_equal(means[:,0],means[:,1])
    off=replay(None,fixed,exc*0,inh,jnp.log(5.),start,jnp.zeros((2,1)))[0]
    assert np.max(abs(np.asarray(means-off)))>1.
    # A different stored peptide state must change the continuation, even in sham.
    cold=replay(None,fixed,exc,inh,jnp.log(5.),start._replace(peptide=start.peptide*0),jnp.zeros((2,1)))[0]
    assert np.max(abs(np.asarray(means-cold)))>.01
