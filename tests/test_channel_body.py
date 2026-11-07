import json
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from r_elegans.brain.conductance_network import initialize_conductance_network,ConductanceNetworkConfig,network_recording
from r_elegans.body import physical_muscle_body_params
from r_elegans.body.channel_bridge import initial_channel_body_state,channel_body_chunk,channel_muscle_activations
from r_elegans.model import load_builtin_model
from r_elegans.rl.full_network_actor import init_full_network_actor
from scripts.diagnose_channel_body import diagnostic_metrics
from scripts.run_embodied_diagnostics import diagnostic_cases,verify_input_bundle
from scripts.bootstrap_embodied_inputs import find_successful_parent


def test_parent_discovery_requires_success_and_existing_checkpoint(tmp_path):
    with pytest.raises(ValueError,match='No successful'):find_successful_parent(tmp_path)
    def cohort(name, workers):
        root=tmp_path/name;root.mkdir()
        (root/'summary.json').write_text(json.dumps(dict(workers=workers)))
        for worker in workers:
            parent=root/f"seed-{worker['seed']}/head_sweep/actor_best.npz"
            parent.parent.mkdir(parents=True);parent.write_bytes(b'checkpoint')
        return root
    def worker(seed,passed,code=0):
        return dict(seed=seed,stages=dict(head_sweep=dict(passed=passed,exit_code=code)))
    old=cohort('head-sweep-20261106',[worker(0,True)])
    new=cohort('head-sweep-20261107',[worker(3,True),worker(0,False),worker(1,True,2)])
    assert find_successful_parent(tmp_path)==new/'seed-3/head_sweep/actor_best.npz'
    (new/'seed-3/head_sweep/actor_best.npz').unlink()
    assert find_successful_parent(tmp_path)==old/'seed-0/head_sweep/actor_best.npz'


def test_physical_release_readout_preserves_nmj_locality_and_resting_tone():
    model=load_builtin_model();parent=init_full_network_actor(model,jax.random.PRNGKey(0))
    _,fixed=initialize_conductance_network(parent,model.neuron_ids)
    zero=jnp.full(302,-1e6)
    rest=channel_muscle_activations(zero,fixed,parent,model)
    expected=jax.nn.sigmoid(-parent.muscle_threshold/(.02+jax.nn.softplus(parent.raw_muscle_slope)))
    np.testing.assert_allclose(rest,expected,atol=1e-6)
    weights=np.asarray(model.neuromuscular_params.synapse_weights)
    signs=np.asarray(model.neuromuscular_params.synapse_signs)
    neuron=int(np.argmax(np.sum((weights>0)&(signs!=0),axis=0)))
    changed=channel_muscle_activations(zero.at[neuron].set(1e6),fixed,parent,model)
    np.testing.assert_array_equal(changed[weights[:,neuron]==0],rest[weights[:,neuron]==0])
    assert np.max(np.abs(changed-rest))>0
    assert np.min(changed)>=0 and np.max(changed)<=1


def test_streaming_brain_body_state_matches_continuous_execution():
    model=load_builtin_model();parent=init_full_network_actor(model,jax.random.PRNGKey(0))
    params,fixed=initialize_conductance_network(parent,model.neuron_ids)
    config=ConductanceNetworkConfig();body=physical_muscle_body_params(12,medium='liquid')
    initial=initial_channel_body_state(fixed,config)
    run=jax.jit(lambda s,n:channel_body_chunk(s,params,fixed,config,parent,model,body,steps=n),static_argnums=1)
    first,_=run(initial,25);split,_=run(first,25);whole,trace=run(initial,50)
    for a,b in zip(jax.tree_util.tree_leaves(split),jax.tree_util.tree_leaves(whole)):
        np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
    assert trace[2].shape==(50,95)
    neural=jax.jit(lambda:network_recording(params,fixed,.002,50,20,config))()
    # Body state has no hidden effect on neural dynamics in this open-loop branch.
    np.testing.assert_allclose(trace[0],neural.voltage_mV,rtol=1e-5,atol=1e-5)
    assert all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(whole))
    held,held_trace=jax.jit(lambda:channel_body_chunk(initial,params,fixed,config,parent,model,body,
        steps=50,hold_voltage=True))()
    np.testing.assert_array_equal(held.brain.voltage_mV,initial.brain.voltage_mV)
    np.testing.assert_allclose(np.ptp(held_trace[2],axis=0),0.,atol=1e-7)


def test_drift_does_not_automatically_count_as_locomotion():
    n=100;v=np.zeros((n,302));muscles=np.full((n,95),.4)
    body=dict(joint_angles=np.zeros((n,11)),position=np.stack((np.linspace(0,1,n),np.zeros(n)),axis=1),heading=np.zeros(n))
    metrics=diagnostic_metrics(v,muscles,body,.002,1.,load_builtin_model().neuron_ids)
    assert metrics['measurable_displacement']
    assert not metrics['muscle_dynamic'] and not metrics['shape_dynamic']
    assert not metrics['locomotion_demonstrated']


def test_server_matrix_and_input_checksums(tmp_path):
    cases=diagnostic_cases()
    assert len(cases)==len({c['name'] for c in cases})==9
    assert {'active','cca1_removed','held_voltage'}=={c['condition'] for c in cases}
    assert {'normalized','liquid','agar'}=={c['medium'] for c in cases}
    hashes={}
    for name in ('network.npz','parent_actor.npz'):
        (tmp_path/name).write_bytes(b'fixture-'+name.encode())
        hashes[name]=hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
    (tmp_path/'inputs.json').write_text(json.dumps(dict(sha256=hashes)))
    verify_input_bundle(tmp_path)
    (tmp_path/'network.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):verify_input_bundle(tmp_path)
