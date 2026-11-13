from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from r_elegans.behavior_control import (ControlProtocol,bath_schedule,metrics,forward_cycles,
    CONTROL_GROUPS,VALIDATION_CASES,control_report,qualify_behavior)


def trace(protocol=ControlProtocol(),kind='controlled'):
    t=np.arange(round(protocol.duration/.01)+1)*.01
    run=np.ones(len(t),dtype=bool)
    if kind in ('controlled','no-start'):
        run[t>=40+protocol.shift]=False
        if kind=='controlled':run[t>=58+protocol.shift+protocol.rest_extension]=True
        if protocol.repeated:
            run[t>=90+protocol.shift]=False;run[t>=108+protocol.shift]=True
    if kind=='quiet':run[:]=False
    q=.01*np.sin(2*np.pi*1.8*t)[:,None]*np.ones((1,11))*run[:,None]
    return dict(dense_time_s=t,dense_joints=q,dense_position=np.zeros((len(t),2)),dense_heading=np.zeros(len(t)))


def test_real_bath_events_enter_only_sensory_cells():
    import jax
    from r_elegans.body.embodied_sensory import (sample_chemical_world,validate_chemical_world,EmbodiedSensoryConfig,
        build_sensory_maps,initial_embodied_sensory_state,transduce_sensory_signals)
    from r_elegans.body import physical_muscle_body_params
    from r_elegans.model import load_builtin_model
    world=bath_schedule(ControlProtocol())
    validate_chemical_world(world)
    np.testing.assert_array_equal(sample_chemical_world(jnp.zeros(2),41.,world),[0,0,60000])
    np.testing.assert_array_equal(sample_chemical_world(jnp.zeros(2),59.,world),[1,0,50000])
    compiled=jax.jit(lambda t:sample_chemical_world(jnp.zeros(2),t,world))
    np.testing.assert_array_equal(compiled(jnp.asarray(41.)),[0,0,60000])
    model=load_builtin_model();body_params=physical_muscle_body_params(12,medium='liquid')
    body=SimpleNamespace(joint_angles=jnp.zeros(11),position=jnp.zeros(2),heading=jnp.asarray(0.),time=jnp.asarray(0.))
    config=EmbodiedSensoryConfig(b_bend_gain_nS=0,dva_gain_nS=0)
    maps=build_sensory_maps(model,config);state=initial_embodied_sensory_state(body,body_params,world,config)
    body.time=jnp.asarray(41.)
    _,membrane,observation=transduce_sensory_signals(state,body,body_params,world,maps,.1,config)
    assert observation.chemical_excitatory_nS[model.neuron_ids.index('ASEL')]>0
    outside=np.ones(302,dtype=bool);outside[np.asarray(maps.chemical_indices)]=False
    np.testing.assert_array_equal(np.asarray(membrane.conductance_nS)[outside],0)
    np.testing.assert_array_equal(sample_chemical_world(jnp.zeros(2),59.,bath_schedule(replace(ControlProtocol(),omit_start=True))),[0,0,50000])


def test_state_windows_require_restart_not_just_stopping_or_always_wiggling():
    protocol=ControlProtocol(repeated=True)
    good=metrics(trace(protocol),protocol)
    assert good['control_passed'] and not good['swimming_passed']
    assert not metrics(trace(kind='wiggle'),ControlProtocol())['control_passed']
    assert not metrics(trace(kind='quiet'),ControlProtocol())['initial_movement_passed']
    stopped=metrics(trace(kind='no-start'),ControlProtocol())
    assert stopped['initial_movement_passed'] and all(r['passed'] for r in stopped['off'])
    assert not stopped['control_passed']


def test_pause_hidden_inside_an_active_window_is_rejected():
    protocol=ControlProtocol(redundant=True);record=trace(protocol)
    pause=(record['dense_time_s']>=32)&(record['dense_time_s']<33.5)
    record['dense_joints'][pause]=0
    assert not metrics(record,protocol)['control_passed']


def gate_fixture():
    good=metrics(trace(),ControlProtocol())
    moving=metrics(trace(kind='wiggle'),ControlProtocol())
    quiet=metrics(trace(kind='quiet'),ControlProtocol())
    no_start=metrics(trace(kind='no-start'),ControlProtocol())
    rows=[]
    for case in VALIDATION_CASES:
        m=good
        if case in ('chemical-blocked','no-cue','stop-blocked'):m=moving
        if case=='start-blocked':m=no_start
        if case=='held-voltage' or case.startswith('controller-'):m=quiet
        if case=='controller-stop_candidate':m=moving
        rows.append(dict(case=case,checkpoint_sha256='a'*64,metrics=deepcopy(m),pre_cue_maximum_difference_rad=0,pre_start_maximum_difference_rad=0))
    return rows


def test_future_behavior_gate_is_complete_causal_and_checkpoint_specific():
    rows=gate_fixture();gate=control_report('a'*64,rows,True)
    assert gate['passed'] and qualify_behavior('a'*64,True,gate)
    assert not qualify_behavior('b'*64,True,gate)
    assert not qualify_behavior('a'*64,True,dict(passed=True,checkpoint_sha256='a'*64))
    assert not control_report('a'*64,rows[:-1],True)['passed']
    stale=deepcopy(rows);stale[0]['checkpoint_sha256']='b'*64
    assert not control_report('a'*64,stale,True)['passed']
    for r in rows:
        if r['case'].startswith('controller-'):r['metrics']=metrics(trace(kind='quiet'),ControlProtocol())
    assert not control_report('a'*64,rows,True)['passed']  # Paralyzing lesions aren't control evidence.


def test_independent_sensory_start_and_stop_are_required():
    rows=gate_fixture()
    next(r for r in rows if r['case']=='no-cue')['metrics']=metrics(trace(kind='no-start'),ControlProtocol())
    assert not control_report('a'*64,rows,True)['passed']  # Spontaneous stopping is insufficient.


def test_forward_motion_needs_closed_nonreciprocal_shape_cycles():
    t=np.linspace(0,10,2001);angles=2*np.pi*1.8*t[:,None]
    reciprocal=.01*np.sin(angles)*np.ones((1,11))
    wave=.01*np.sin(angles-np.linspace(0,np.pi,11))
    p=np.column_stack((t*.001*.02,np.zeros(len(t))));h=np.zeros(len(t))
    assert not forward_cycles(t,reciprocal,p,h)['passed']
    assert forward_cycles(t,wave,p,h)['passed']
    assert not forward_cycles(t,wave,-p,h)['passed']
    assert not forward_cycles(t,wave,np.zeros_like(p),h)['passed']


def test_future_policy_and_runtime_case_contract_match():
    import json
    from pathlib import Path
    policy=json.loads((Path(__file__).resolve().parents[1]/'experiments/behavior_control_policy_v1.json').read_text())
    assert policy['swimming_example_validation_cases']==list(VALIDATION_CASES)


def test_parent_snapshot_preserves_exact_selected_rng_and_hashes(tmp_path):
    import json
    from scripts.run_controlled_swimming import snapshot
    from scripts.experiment_records import file_sha256
    from scripts.compare_rhythm_worker import verify_inputs
    source=tmp_path/'inputs';source.mkdir()
    (source/'candidate.npz').write_bytes(b'original network')
    (source/'manifest.json').write_text(json.dumps(dict(sha256={'candidate.npz':file_sha256(source/'candidate.npz')},sources={})))
    worker=tmp_path/'worker';selected=worker/'selected/iteration-000003';selected.mkdir(parents=True)
    for name in ('network_best.npz','best_state.npz','configuration.json','optimizer_state.npz','rng_state.json'):
        (worker/name).write_bytes(name.encode())
    (worker/'summary.json').write_text(json.dumps(dict(selected_iteration=3)))
    (selected/'trajectory.npz').write_bytes(b'exact selected trace')
    (selected/'rng_state.json').write_text('{"selection":3}')
    copied=tmp_path/'copy';snapshot(source,copied,worker,'no-cue')
    manifest=verify_inputs(copied)
    assert (copied/'parent_training_trace.npz').read_bytes()==b'exact selected trace'
    assert json.loads((copied/'behavior_parent_selected_rng.json').read_text())=={'selection':3}
    assert manifest['continuation_parent_sha256']==file_sha256(worker/'network_best.npz')
