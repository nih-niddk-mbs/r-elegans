"""Full-model identity, explicit extrasynaptic state and restart semantics."""
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
import numpy as np
import pytest
from r_elegans.brain.peptide_checkpoint import bundle_model,load_model,validate,save_state,load_state,make_rest
from scripts.experiment_records import file_sha256

NETWORK=Path(__file__).resolve().parents[1]/'r_elegans/assets/mac_rhythm_v1_polarized/network.npz'


def source_state(tmp_path):
    with np.load(NETWORK) as z:n=len(z['neuron_ids'])
    state=tmp_path/'state.npz'
    np.savez(state,start_voltage_mV=np.full((4,n),-35.),start_gates=np.full((4,n,35),.5),
        start_calcium_uM=np.full((4,n),.05),start_indicator=np.zeros((4,n)),start_peptide=np.full((4,n),.6))
    return state,dict(sender=np.ones((2,n)),receiver=np.ones((2,n)))


def test_same_wired_weights_have_distinct_explicit_full_model_identities(tmp_path):
    state,asset=source_state(tmp_path)
    paths=[tmp_path/'off.npz',tmp_path/'on.npz']
    for enabled,path in zip((False,True),paths):
        bundle_model(NETWORK,state,path,enabled=enabled,asset=asset,asset_sha256='test-asset',network_sha256=file_sha256(NETWORK))
    assert file_sha256(paths[0])!=file_sha256(paths[1])
    off,on=[load_model(p) for p in paths]
    for x,y in zip(off[0],on[0]):np.testing.assert_array_equal(x,y)
    assert not np.asarray(off[4]).any() and np.asarray(on[4]).max()>0
    np.testing.assert_array_equal(on[6].peptide,np.full((4,302),.6))
    from r_elegans.brain.conductance_network import load_network
    with pytest.raises(ValueError,match='Unsupported'):load_network(paths[1])


def test_invalid_memory_or_implicit_enabled_flag_is_rejected(tmp_path):
    state,asset=source_state(tmp_path);path=tmp_path/'model.npz'
    bundle_model(NETWORK,state,path,enabled=True,asset=asset,asset_sha256='test',network_sha256='test')
    with np.load(path) as z:a={k:z[k] for k in z.files}
    original=a['start_peptide'].copy();a['start_peptide'][0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):validate(a)
    a['start_peptide']=original;a['extrasynaptic_enabled']=np.array(1)
    with pytest.raises(ValueError,match='boolean'):validate(a)


def test_neural_and_peptide_serialized_continuation_matches_uninterrupted(tmp_path,monkeypatch):
    import jax
    import jax.numpy as jnp
    import r_elegans.brain.conductance_network as module
    from r_elegans.brain.peptide_state_replay import FullState
    def passive(s,p,f,dt,steps,c,*,membrane_input):
        v=s.voltage_mV+dt*(membrane_input.weighted_reversal_pA-membrane_input.conductance_nS*s.voltage_mV)
        return s._replace(voltage_mV=v,indicator=v)
    monkeypatch.setattr(module,'advance_network',passive)
    class F(NamedTuple):
        release_threshold_mV:object
        release_slope_mV:object
    f=F(jnp.array([-45.]),jnp.array([10.]))
    start=FullState(module.ConductanceNetworkState(jnp.array([-35.]),jnp.zeros((1,35)),jnp.array([.05]),jnp.array([0.])),jnp.array([.6]))
    pp=SimpleNamespace(log_tau_s=jnp.log(5.))
    # SimpleNamespace is not a JAX argument pytree; use existing named parameter tuple.
    from r_elegans.brain.extrasynaptic import PeptideParams
    pp=PeptideParams(jnp.array([-9.]),jnp.array([-9.]),pp.log_tau_s)
    c=SimpleNamespace(synaptic_unit_nS=.03);exc=jnp.ones((1,1));inh=exc*0
    one=make_rest(c,[0],1.,dt=.1,step_ms=100.);two=make_rest(c,[0],2.,dt=.1,step_ms=100.)
    half,_=one(None,f,pp,exc,inh,start);save_state(tmp_path/'half.npz',half)
    recovered=load_state(tmp_path/'half.npz');end,_=one(None,f,pp,exc,inh,recovered)
    full,_=two(None,f,pp,exc,inh,start)
    for a,b in zip(jax.tree.leaves(end),jax.tree.leaves(full)):np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
    cold=recovered._replace(peptide=recovered.peptide*0)
    cold_end,_=one(None,f,pp,exc,inh,cold)
    assert abs(float(cold_end.neural.voltage_mV[0]-end.neural.voltage_mV[0]))>.01
