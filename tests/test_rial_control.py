import json,sys,time
import numpy as np
import pytest
from r_elegans.rial_control import cases,branch_probe,membrane_probe,compare,choose_membrane


@pytest.fixture(autouse=True)
def preserve_jax_precision():
    import jax
    previous=jax.config.x64_enabled
    yield
    jax.config.update('jax_enable_x64',previous)


def fixture():
    ids=['RIAL','RIAR','SMDDL','SMDDR','SMDVL','SMDVR','DB01']
    mask=np.zeros((7,7));mask[2:6,0]=1;mask[6,0]=1;mask[4,1]=1
    mean=np.linspace(.1,.7,7)
    return ids,mask,mean


def test_sparse_branch_hold_changes_only_existing_selected_edges_and_releases():
    import jax
    from r_elegans.brain.conductance_network import chemical_release_adjustment
    ids,mask,mean=fixture();case=next(c for c in cases() if c['name']=='RIAL-SMD-mean')
    probe=branch_probe(ids,mask,mean,case,'repeat');active=probe(35.)
    np.testing.assert_array_equal(active.post_indices,[2,3,4,5]);np.testing.assert_array_equal(active.pre_indices,0)
    np.testing.assert_allclose(active.release_probability,mean[0])
    for t in (0.,29.9,40.,75.,90.):assert not bool(probe(t).active)
    assert bool(probe(85.).active)
    release=np.linspace(.2,.8,7);rev=np.arange(7)*-10.;chemical=mask*.2
    dg,de=chemical_release_adjustment(chemical,release,rev,active)
    reference=chemical*np.tile(release,(7,1));reference[2:6,0]=chemical[2:6,0]*mean[0]
    np.testing.assert_allclose(chemical@release+dg,reference.sum(axis=1),atol=1e-7)
    np.testing.assert_allclose(chemical@(release*rev)+de,(reference*rev).sum(axis=1),atol=1e-6)
    assert np.all(reference>=0) and float(dg[6])==0  # Other RIAL outputs remain live.
    with pytest.raises(ValueError):branch_probe(ids,np.zeros_like(mask),mean,case,'nominal')
    duplicate=dict(case,targets=['SMDVL','SMDVL'])
    with pytest.raises(ValueError):branch_probe(ids,mask,mean,duplicate,'nominal')
    jax.jit(probe)(35.)


def test_inactive_override_preserves_neural_substep_and_active_matches_explicit_equation():
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    from pathlib import Path
    from r_elegans.brain.conductance_network import (load_network,initial_network_state,advance_network,network_synapses,
        ChemicalReleaseOverride,channel_open_fractions)
    path=Path(__file__).resolve().parents[1]/'r_elegans/assets/autonomous_wiggle_v1/network.npz'
    params,fixed,config=load_network(path)
    params,fixed=jax.tree.map(lambda a:a.astype('float64') if a.dtype.kind=='f' else a,(params,fixed))
    state=initial_network_state(fixed,config);dt=.0001
    chem,gap=network_synapses(params,fixed,config);post,pre=np.argwhere(np.asarray(chem)>0)[0]
    override=ChemicalReleaseOverride(jnp.asarray([post]),jnp.asarray([pre]),jnp.asarray([.2]),jnp.asarray(False))
    baseline=advance_network(state,params,fixed,dt,1,config)
    inactive=advance_network(state,params,fixed,dt,1,config,chemical_release_override=override)
    for a,b in zip(baseline,inactive):np.testing.assert_array_equal(a,b)
    active=advance_network(state,params,fixed,dt,1,config,chemical_release_override=override._replace(active=jnp.asarray(True)))
    r=jax.nn.sigmoid((state.voltage_mV-fixed.release_threshold_mV)/fixed.release_slope_mV)
    conductance=fixed.cells.conductances_nS*jnp.exp(params.log_channel_scale)*channel_open_fractions(state.gates)
    cells=fixed.cells
    reversals=jnp.stack([cells.calcium_reversal_mV if i in (6,7,8) else cells.sodium_reversal_mV if i==14 else cells.leak_reversal_mV if i==16 else cells.potassium_reversal_mV for i in range(17)],axis=1)
    total=conductance.sum(axis=1)+chem@r+gap.sum(axis=1)
    drive=(conductance*reversals).sum(axis=1)+chem@(r*fixed.synapse_reversal_mV)+gap@state.voltage_mV+fixed.bias_pA
    delta=chem[post,pre]*(.2-r[pre]);g=total[post]+delta;e=drive[post]+delta*fixed.synapse_reversal_mV[pre]
    expected=state.voltage_mV[post]+(e/g-state.voltage_mV[post])*(-jnp.expm1(-dt*1000*g/cells.capacitance_pF[post]))
    np.testing.assert_allclose(active.voltage_mV[post],expected,atol=1e-12,rtol=1e-12)
    keep=np.arange(302)!=post;np.testing.assert_array_equal(active.voltage_mV[keep],baseline.voltage_mV[keep])
    # Same old state means gates/calcium/indicator updates stay identical in one substep.
    for a,b in zip(active[1:],baseline[1:]):np.testing.assert_array_equal(a,b)


def test_membrane_and_branch_comparisons_have_separate_access_semantics():
    from test_smd_control import traces
    trace,sham,_=traces();ids=['RIAL','RIAR','SMDDL','SMDDR','SMDVL','SMDVR']
    branch=next(c for c in cases() if c['name']=='RIAL-SMD-mean')
    result=compare(trace,sham,ids,branch)
    assert result['membrane_inhibition_confirmed'] is None
    assert result['pre_match'] and result['active_and_recovery_preserved']
    assert result['ventral_SMD_metrics'][0]['ventral_SMD_rms_ratio']==0
    assert 'reversible_neural_target' not in result and not result['learned_stopping_qualified']
    membrane=next(c for c in cases() if c['name']=='RIAL-0.1')
    assert not compare(trace,sham,ids,membrane)['membrane_inhibition_confirmed']
    trace['dense_voltage_mV'][(trace['dense_time_s']>=30)&(trace['dense_time_s']<40),0]-=30
    result=compare(trace,sham,ids,membrane)
    assert result['membrane_inhibition_confirmed']
    assert choose_membrane([dict(case=branch,metrics=result),dict(case=membrane,metrics=result)])==membrane


def test_ria_membrane_probe_does_not_target_smd():
    ids,_,_=fixture();case=next(c for c in cases() if c['name']=='RIAL-0.1')
    probe=membrane_probe(ids,case,'nominal')
    np.testing.assert_allclose(probe(35.).conductance_nS,[.1,0,0,0,0,0,0])
    assert not np.asarray(probe(40.).conductance_nS).any()


def test_rial_watchdog_dispatch_and_total_budget(tmp_path,monkeypatch):
    import scripts.total_job_budget as runner
    out=tmp_path/'output';seen=[]
    def run(command,seconds):
        seen.append(command);(out/'summary.json').write_text(json.dumps(dict(execution_valid=True,validation_complete=True)))
        return 0,False
    monkeypatch.setattr(runner,'run_group',run)
    monkeypatch.setattr(sys,'argv',['budget','--pipeline','rial-control','--output-dir',str(out),'--source-run',str(tmp_path/'source'),'--total-minutes','60'])
    assert runner.main()==0 and seen[0][3]=='scripts.run_rial_control'
    assert json.loads((out/'deadline.json').read_text())['experiment']=='E084'
    assert json.loads((out/'budget_receipt.json').read_text())['total_minutes']==60


def test_branch_worker_supplies_only_chemical_override_and_records_held_edges(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from test_smd_control import traces
    import scripts.rial_control_worker as worker
    import scripts.neural_timing_setup as setup_module
    import scripts.compare_rhythm_worker as verifier
    import r_elegans.brain.conductance_network as network
    import r_elegans.rhythm_comparison as reconstruction
    from scripts.experiment_records import file_sha256
    test,sham,_=traces();ids,mask,mean=fixture()
    for data in (test,sham):data['dense_voltage_mV']=np.column_stack((data['dense_voltage_mV'],np.full(len(data['dense_time_s']),-40.)))
    root=tmp_path/'inputs';root.mkdir();out=tmp_path/'trial'
    reference=tmp_path/'reference.npz';np.savez_compressed(reference,**sham)
    (root/'reference.json').write_text(json.dumps(dict(path=str(reference),sha256=file_sha256(reference))))
    (root/'diagnostic_model.npz').write_bytes(b'fixture')
    np.savez_compressed(root/'diagnostic_coordinates.npz',offsets=np.zeros(1))
    np.savez_compressed(root/'baseline_release.npz',mean=mean)
    seen=[]
    def factory(dt,brain,**kwargs):
        assert kwargs['diagnostic_membrane'] is None and 'diagnostic_nmj_release' not in kwargs
        override=kwargs['diagnostic_chemical_release'](35.);seen.append(override)
        return lambda *args:test
    params=(np.zeros(1),);fixed=SimpleNamespace(chemical_mask=mask)
    monkeypatch.setattr(setup_module,'setup',lambda root:dict(decode=lambda x:params,fixed=fixed,config=None,model=SimpleNamespace(neuron_ids=ids),factory=factory,gains=np.ones(3)))
    monkeypatch.setattr(verifier,'verify_inputs',lambda root:None)
    monkeypatch.setattr(network,'load_network',lambda path:(params,fixed,None))
    monkeypatch.setattr(reconstruction,'assert_fixed_equal',lambda *args:None)
    monkeypatch.setattr(worker,'capture_run_record',lambda *args:None)
    monkeypatch.setattr(worker.shutil,'disk_usage',lambda path:SimpleNamespace(free=1024**4))
    monkeypatch.setattr(sys,'argv',['worker','--inputs',str(root),'--output-dir',str(out),'--case','RIAL-SMD-mean','--until',str(time.monotonic()+60)])
    worker.main();r=json.loads((out/'summary.json').read_text())
    assert r['experiment']=='E084' and r['no_training'] and not r['learned_stopping_qualified']
    assert r['branch_edges']['post_indices']==[2,3,4,5]
    np.testing.assert_allclose(r['branch_edges']['held_release'],mean[0])
    assert r['source_reference_errors']['dense_voltage_mV']==0
    assert file_sha256(out/'trajectory.npz')==r['trajectory_sha256']


def test_closed_loop_passes_edge_override_without_changing_sensing_or_nmjs(monkeypatch):
    from types import SimpleNamespace
    import jax.numpy as jnp
    import r_elegans.body.closed_loop as loop
    from r_elegans.brain.conductance_network import MembraneConductanceInput
    ids,mask,mean=fixture();case=next(c for c in cases() if c['name']=='RIAL-SMD-mean')
    probe=branch_probe(ids,mask,mean,case,'nominal')
    brain=SimpleNamespace(voltage_mV=jnp.full(7,-40.))
    state=SimpleNamespace(sensory='old',brain_body=SimpleNamespace(brain=brain,body=SimpleNamespace(time=35.)))
    sensory=MembraneConductanceInput(jnp.zeros(7),jnp.zeros(7));seen=[]
    monkeypatch.setattr(loop,'transduce_sensory_signals',lambda *args,**kwargs:('updated',sensory,'observed'))
    def advance(*args,**kwargs):
        assert kwargs['membrane_input'] is sensory
        seen.append(kwargs['chemical_release_override']);return brain
    monkeypatch.setattr(loop,'advance_network',advance)
    monkeypatch.setattr(loop,'channel_muscle_activations',lambda *args:'live NMJ')
    result=loop.embodied_neural_step(state,None,None,None,None,None,None,None,None,.002,20,diagnostic_chemical_release=probe)
    assert result[0] is brain and result[1:4]==('updated','live NMJ','observed')
    np.testing.assert_array_equal(seen[0].post_indices,[2,3,4,5]);assert bool(seen[0].active)
