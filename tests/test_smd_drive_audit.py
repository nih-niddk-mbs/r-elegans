import json
from pathlib import Path
import numpy as np
import pytest
from scipy.special import expit
from r_elegans.smd_drive_audit import rhythmic_inputs,load_nmj,muscle_activation,projection,residual_output


def test_actual_current_variation_is_not_mistaken_for_varying_presynaptic_drive():
    t=np.arange(0,5,.01);v=np.column_stack((-40+np.sin(2*np.pi*1.8*t),np.full(len(t),-35.)))
    network=dict(ids=['SMDVL','AVAL'],chemical=np.zeros((2,2)),gap=np.array([[0,.2],[.2,0]]),
                 threshold=np.full(2,-40.),slope=np.full(2,5.),reversal=np.zeros(2))
    r=rhythmic_inputs(t,v,network,0)
    assert r['actual_total_synaptic_current_std_pA']>.1
    assert r['source_harmonic_sum_pA']<1e-12
    v[:,1]=-35+3*np.sin(2*np.pi*1.8*t)
    r=rhythmic_inputs(t,v,network,0)
    assert r['source_harmonic_sum_pA']==pytest.approx(.6,abs=1e-10)
    assert r['top_rhythmic_inputs'][0]['neuron']=='AVAL'
    assert r['dominant_voltage_frequency_hz']==pytest.approx(1.8)


def test_nmj_and_projection_agree_with_runtime_without_neural_simulation():
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from r_elegans.model import load_builtin_model
    from r_elegans.rl.full_network_actor import actor_from_arrays
    from r_elegans.brain.conductance_network import load_network
    from r_elegans.body.channel_bridge import channel_muscle_activations
    from r_elegans.body.neuromuscular import build_muscle_projection
    root=Path(__file__).resolve().parents[1];asset=root/'r_elegans/assets/runtime_model_v1.json'
    with np.load(root/'r_elegans/assets/autonomous_wiggle_v1/parent_actor.npz',allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files};actor=actor_from_arrays(z)
    _,fixed,_=load_network(root/'r_elegans/assets/autonomous_wiggle_v1/network.npz')
    model=load_builtin_model();actor,fixed,model=jax.tree.map(lambda x:x.astype('float64') if hasattr(x,'dtype') and x.dtype.kind=='f' else x,(actor,fixed,model))
    document=json.loads(asset.read_text());nmj=load_nmj(document,arrays,model.neuron_ids)
    v=np.random.default_rng(8300).uniform(-70,-25,(10,302))
    release=expit((v-np.asarray(fixed.release_threshold_mV))/np.asarray(fixed.release_slope_mV))
    expected=np.asarray(jax.vmap(lambda x:channel_muscle_activations(x,fixed,actor,model))(jnp.asarray(v)))
    predicted=muscle_activation(release,nmj)
    np.testing.assert_allclose(predicted,expected,atol=1e-12,rtol=1e-12)
    with jax.experimental.enable_x64(False):p=build_muscle_projection(11,head_at_end=True)
    runtime=np.asarray(p.dorsal_weights,dtype=float)-np.asarray(p.ventral_weights,dtype=float)
    assert np.max(abs(predicted@projection().T-predicted@runtime.T))<1e-6


def test_counterfactual_all_release_hold_removes_command_variation():
    ids=['SMDVL','SMDVR'];t=np.arange(0,5,.01)
    v=np.column_stack((-40+5*np.sin(2*np.pi*1.8*t),-40-5*np.sin(2*np.pi*1.8*t)))
    network=dict(ids=ids,threshold=np.full(2,-40.),slope=np.full(2,5.))
    nmj=dict(signed=np.array([[1.,0.],[0.,1.]]),threshold=np.zeros(2),slope=np.full(2,.1))
    result=residual_output(v,network,nmj,np.tile([[1.,-1.]],(3,1)))
    rows={r['group']:r for r in result['conditional_release_holds']}
    assert rows['all']['centered_command_rms_ratio']<1e-12
    assert rows['ventral-SMD']['centered_command_rms_ratio']<1e-12
    assert rows['RMD']['centered_command_rms_ratio']==pytest.approx(1.)


def test_complete_offline_audit_uses_archived_asset_and_checks_reconstruction(tmp_path,monkeypatch):
    import sys
    import scripts.audit_smd_drive as runner
    from scripts.experiment_records import file_sha256
    from r_elegans.avb_audit import load_saved_network
    run=tmp_path/'source';inputs=run/'inputs';inputs.mkdir(parents=True);out=tmp_path/'audit'
    ids=['SMDVL','SMDVR','SMDDL','SMDDR','AVAL','RMDR'];n=len(ids)
    document=dict(neuron_ids=ids,nmj=dict(muscle_index=list(range(95)),neuron_index=[0 if i<48 else 1 for i in range(95)],contact_count=[1]*95,sign=[1]*95))
    record=run/'run_records/original';asset=record/'source/r_elegans/assets/runtime_model_v1.json';asset.parent.mkdir(parents=True)
    asset.write_text(json.dumps(document));(record/'source_record.json').write_text(json.dumps(dict(source_sha256={'r_elegans/assets/runtime_model_v1.json':file_sha256(asset)})))
    model=inputs/'diagnostic_model.npz'
    np.savez_compressed(model,schema_version=1,neuron_ids=ids,config_json=json.dumps(dict(synaptic_unit_nS=.03)),
        params_raw_chemical=np.zeros((n,n)),params_raw_gap=np.zeros((n,n)),params_log_channel_scale=np.zeros((n,17)),
        fixed_chemical_mask=np.ones((n,n))-np.eye(n),fixed_gap_mask=np.ones((n,n))-np.eye(n),
        fixed_synapse_reversal_mV=np.zeros(n),fixed_release_threshold_mV=np.full(n,-40.),fixed_release_slope_mV=np.full(n,5.),
        cells_conductances_nS=np.ones((n,17)),fixed_bias_pA=np.zeros(n),cells_leak_reversal_mV=np.full(n,-60.))
    pin=file_sha256(model);monkeypatch.setattr(runner,'PIN',pin)
    actor=dict(raw_nmj=np.zeros((95,n)),unknown_polarity=np.zeros((95,n)),muscle_threshold=np.zeros(95),raw_muscle_slope=np.full(95,-2.))
    np.savez_compressed(inputs/'actor.npz',**actor)
    (inputs/'manifest.json').write_text(json.dumps(dict(sha256={name:file_sha256(inputs/name) for name in ('diagnostic_model.npz','actor.npz')})))
    nmj=load_nmj(document,actor,ids);network=load_saved_network(model);records=[]
    tasks=[('sham','nominal','ordinary'),('ventral','nominal','ordinary'),('SMD-0.1','nominal','ordinary'),
           ('sham','long-hold','ordinary'),('ventral','long-hold','ordinary'),('sham','nominal','neural-half'),('ventral','nominal','neural-half')]
    for case,variant,grid in tasks:
        folder=run/(case+'--'+variant+'--'+grid);folder.mkdir()
        duration=150 if variant=='long-hold' else 60;t=np.arange(0,duration+.01,.05)
        v=-40+np.sin(2*np.pi*1.8*t[:,None]+np.arange(n))*np.arange(1,n+1)[None,:]
        target=ids[:4] if case=='SMD-0.1' else ids[:2] if case=='ventral' else []
        on=(t>=30)&(t<(90 if variant=='long-hold' else 40))
        for name in target:v[on,ids.index(name)]-=25
        release=expit((v-network['threshold'])/network['slope']);muscles=muscle_activation(release,nmj);cmd=muscles@projection().T
        path=folder/'trajectory.npz';np.savez_compressed(path,dense_time_s=t,dense_voltage_mV=v,dense_muscles=muscles,
            dense_command=cmd,dense_filtered_drive=.8*cmd,dense_joints=.4*cmd)
        records.append(dict(path=str(folder),exit_code=0,result=dict(case=dict(name=case,targets=target,conductance_nS=.1 if target else 0.),
            variant=variant,grid=grid,checkpoint_sha256=pin,trajectory_sha256=file_sha256(path))))
    summary=dict(experiment='E082',checkpoint_sha256=pin,validation_complete=True,screen=records[:3],confirmation=records[3:5],refinement=records[5:])
    (run/'summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(runner,'capture_run_record',lambda *args,**kwargs:None)
    monkeypatch.setattr(runner,'plot',lambda out,*args:(out/'signal_chain.png').write_bytes(b'figure fixture'))
    monkeypatch.setattr(sys,'argv',['audit','--run',str(run),'--output-dir',str(out)])
    runner.main();s=json.loads((out/'summary.json').read_text())
    assert s['status']=='completed' and s['no_new_simulation'] and s['no_training']
    assert len(s['results'])==4
    assert all(c['maximum_muscle_reconstruction_error']<1e-12 for c in s['checks'].values())
    evidence=json.loads((out/'evidence_hashes.json').read_text())
    assert evidence[str(asset)]==file_sha256(asset)
    assert not list(out.glob('*.npz'))
