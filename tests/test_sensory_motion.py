from copy import deepcopy
from types import SimpleNamespace
import json,sys
import numpy as np
import pytest
from scipy.special import logit
from r_elegans.sensory_motion import features,score,parameter_groups,panel_assessment
from test_rial_sensory import paired


def data(effect=.1,variant='nominal',mode='normal'):
    _,traces,ids=paired(effect=effect,variant=variant,mode=mode);rows={}
    from r_elegans.rial_sensory import protocol
    for condition,trace in traces.items():
        t=trace['dense_time_s'];release=np.full(len(t),.95)
        if condition=='cue' and mode!='blocked':
            for a,b in protocol(variant)['response']:release[(t>=a)&(t<b)]-=.0001
        trace['dense_voltage_mV'][:,ids.index('AIYL')]=-40+5*logit(release)
        rows[condition]=features(trace,ids,variant,-40,5)
    return rows


def test_reward_gradual_body_motion_reduction_preserving_relay_and_recovery():
    parent=data()['no_cue'];losses=[score(data(effect=e),parent,.05,1.05)['loss'] for e in (0.,.01,.03,.07)]
    assert all(a>b for a,b in zip(losses,losses[1:]))
    for variant in ('nominal','low-dose','shift','repeat','redundant','long-cue'):
        m=score(data(variant=variant),parent,.05)
        assert m['partial_attenuation_passed'] and m['relay_component_preserved']
        assert not m['full_start_stop_qualified'] and not m['swimming_qualified']
    for mode in ('no_recovery','weak','blocked','phase'):
        assert not score(data(mode=mode),parent,.05)['partial_attenuation_passed']


def test_rest_does_not_need_cue_on_oscillations_but_must_recover():
    pair=data();parent=pair['no_cue']
    for row in pair['cue']['response']:
        row['regional_cycles_rad']=[0.]*3
        row['regional_angular_speed_rms_rad_s']=[0.]*3
        row['frequency_hz']=[None]*3
    m=score(pair,parent,.95)
    assert m['rest_component_passed'] and not m['full_start_stop_qualified']
    pair['cue']['active'][-1]['regional_angular_speed_rms_rad_s']=[0.]*3
    assert not score(pair,parent,.95)['rest_component_passed']


def test_one_unsuppressed_region_or_lost_relay_prevents_gate():
    pair=data();parent=pair['no_cue'];pair['cue']['response'][0]['regional_angular_speed_rms_rad_s'][0]=pair['no_cue']['response'][0]['regional_angular_speed_rms_rad_s'][0]
    assert not score(pair,parent,.05)['partial_attenuation_passed']
    pair=data()
    pair['cue']['response'][0]['AIYL_release_mean']=pair['no_cue']['response'][0]['AIYL_release_mean']
    assert not score(pair,parent,.05)['relay_component_preserved']


def test_motion_groups_add_existing_rial_output_but_freeze_smd_channels_and_gaps():
    ids=('AWAL','AIYL','AIYR','RIAL','SMDVL','DB01');n=len(ids)
    channels=np.ones((n,17),bool);channels[:,16]=False
    spec=SimpleNamespace(channels=channels,chemical=np.ones((n,n),bool));size=int(channels.sum()+spec.chemical.sum())
    groups=parameter_groups(spec,ids,np.zeros(size),-np.ones(size),np.ones(size))
    assert 'chemical:RIAL->SMDVL' in [g['name'] for g in groups]
    ci=np.nonzero(channels);nc=len(ci[0]);used=[i for g in groups for i in g['indices']]
    assert all(ids[ci[0][i]] in ('AWAL','AIYL','RIAL') for i in used if i<nc)
    assert len(used)==len(set(used))


def test_bilateral_groups_add_right_path_without_changing_left_groups_or_overlapping():
    ids=('AWAL','AWAR','AIYL','AIYR','RIAL','RIAR','SMDVL','SMDVR');n=len(ids)
    channels=np.ones((n,17),bool);channels[:,16]=False
    spec=SimpleNamespace(channels=channels,chemical=np.ones((n,n),bool));size=int(channels.sum()+spec.chemical.sum())
    args=(spec,ids,np.zeros(size),-np.ones(size),np.ones(size))
    left=parameter_groups(*args);both=parameter_groups(*args,bilateral=True)
    assert both[:len(left)]==left and len(both)>len(left)
    names=[g['name'] for g in both[len(left):]]
    assert {'chemical:AIYR->RIAR','chemical:RIAR->SMDVL','density:AWAR:inward','density:RIAR:outward'}<=set(names)
    ci=np.nonzero(channels);nc=len(ci[0]);used=[i for g in both for i in g['indices']]
    assert len(used)==len(set(used))
    assert all(ids[ci[0][i]] in ('AWAL','AWAR','AIYL','AIYR','RIAL','RIAR') for i in used if i<nc)


def test_worst_region_term_penalizes_the_weakest_region_only_when_enabled():
    parent=data()['no_cue'];pair=data(effect=.03)
    row=pair['cue']['response'][0]['regional_angular_speed_rms_rad_s']
    base=parent['response'][0]['regional_angular_speed_rms_rad_s'];row[0]=base[0]
    plain=score(pair,parent,.05,1.05);worst=score(pair,parent,.05,1.05,worst_region=True)
    assert worst['loss']>plain['loss'] and worst['worst_region_term'] and 'worst_region_term' not in plain
    assert score(pair,parent,.05,1.05,worst_region=False)==plain


def test_combine_replays_concatenates_source_edges_with_shared_activation():
    import jax.numpy as jnp
    from r_elegans.brain.conductance_network import ChemicalReleaseOverride
    from r_elegans.sensory_motion import combine_replays
    one=lambda t:ChemicalReleaseOverride(jnp.array([4]),jnp.array([2]),jnp.array([.3]),jnp.asarray(t>=1))
    two=lambda t:ChemicalReleaseOverride(jnp.array([5,6]),jnp.array([3,3]),jnp.array([.6,.6]),jnp.asarray(t>=1))
    o=combine_replays([one,two])(2.)
    assert o.post_indices.tolist()==[4,5,6] and o.pre_indices.tolist()==[2,3,3] and bool(o.active)
    assert np.allclose(o.release_probability,[.3,.6,.6])


def test_panel_requires_cue_block_and_both_control_interneuron_replays():
    pair=data();m=score(pair,pair['no_cue'],.05)
    rows=[dict(status='completed',checkpoint_sha256='same',grid='ordinary',metrics=deepcopy(m)) for i in range(11)]
    rows[8]['metrics']['regional_motion_ratios']=[[1.,1.,1.]];rows[8]['metrics']['relay_response_mean_changes']=[0.]
    for r in rows[9:]:r['mediation']=dict(cue_attenuation_reduced=True,baseline_preserved=True)
    assert panel_assessment(rows,.05)['engineering_criteria_met']
    rows[10]['mediation']['cue_attenuation_reduced']=False
    assert not panel_assessment(rows,.05)['engineering_criteria_met']
    assert not panel_assessment(rows[:-1],.05)['complete']


def test_four_hour_budget_dispatches_training_and_automatic_audit(tmp_path,monkeypatch):
    import scripts.total_job_budget as runner
    out=tmp_path/'out';calls=[]
    def run(cmd,seconds):
        calls.append((cmd,seconds))
        if 'scripts.audit_relay_experiments' in cmd:
            audit=tmp_path/'out-audit';audit.mkdir();(audit/'summary.json').write_text(json.dumps(dict(audit_complete=True,integrity_passed=True)))
        else:(out/'summary.json').write_text(json.dumps(dict(execution_valid=True,all_stages_finished=True,validation_complete=True,sensory_motion_component_qualified=True)))
        return 0,False
    monkeypatch.setattr(runner,'run_group',run)
    monkeypatch.setattr(sys,'argv',['budget','--pipeline','sensory-motion','--total-minutes','240','--output-dir',str(out),'--source-run',str(tmp_path/'source')])
    assert runner.main()==0 and 'scripts.run_sensory_motion' in calls[0][0]
    assert '--objective' not in calls[0][0] and calls[0][1]<=220*60
    assert 'scripts.audit_relay_experiments' in calls[1][0]
    assert json.loads((out/'budget_receipt.json').read_text())['post_audit_component_qualified']


def test_bilateral_pipeline_dispatches_e094_profile(tmp_path,monkeypatch):
    import scripts.total_job_budget as runner
    out=tmp_path/'out';calls=[]
    def run(cmd,seconds):
        calls.append(cmd)
        if 'scripts.audit_relay_experiments' in cmd:
            audit=tmp_path/'out-audit';audit.mkdir();(audit/'summary.json').write_text(json.dumps(dict(audit_complete=True,integrity_passed=True)))
        else:(out/'summary.json').write_text(json.dumps(dict(execution_valid=True,validation_complete=True,sensory_motion_component_qualified=False)))
        return 0,False
    monkeypatch.setattr(runner,'run_group',run)
    monkeypatch.setattr(sys,'argv',['budget','--pipeline','sensory-motion-bilateral','--total-minutes','240','--output-dir',str(out),'--source-run',str(tmp_path/'source')])
    assert runner.main()==0 and 'scripts.run_sensory_motion' in calls[0] and calls[0][calls[0].index('--profile')+1]=='e094'
    assert '--objective' not in calls[0] and 'scripts.audit_relay_experiments' in calls[1]
    assert json.loads((out/'deadline.json').read_text())['experiment']=='E094'


def test_worker_keeps_aiyl_and_rial_replay_factories_distinct(tmp_path,monkeypatch):
    import time
    import scripts.rial_sensory_worker as worker
    import scripts.compare_rhythm_worker as verifier
    import r_elegans.brain.conductance_network as network
    _,traces,ids=paired(effect=.1);seen=[]
    from r_elegans.rial_sensory import protocol
    for condition,trace in traces.items():
        t=trace['dense_time_s'];release=np.full(len(t),.95)
        if condition=='cue':release[(t>=41)&(t<46)]-=.0001
        trace['dense_voltage_mV'][:,ids.index('AIYL')]=-40+5*logit(release)
    mask=np.zeros((len(ids),len(ids)));mask[ids.index('RIAL'),ids.index('AIYL')]=1;mask[2:4,ids.index('RIAL')]=1
    fixed=SimpleNamespace(release_threshold_mV=np.full(len(ids),-40.),release_slope_mV=np.full(len(ids),5.),chemical_mask=mask)
    def factory(dt,brain,**kwargs):
        cue=bool(np.asarray(kwargs['world_override'].concentration_um)[:,0].max()>0)
        probe=kwargs.get('diagnostic_chemical_release')
        if probe is not None:seen.append(tuple(np.asarray(probe(45.).pre_indices)))
        return lambda *args:traces['no_cue' if probe is not None else 'cue' if cue else 'no_cue']
    monkeypatch.setattr(worker,'_CACHE',{})
    monkeypatch.setattr(worker,'_DATA',dict(origin=np.zeros(1),lower=-np.ones(1),upper=np.ones(1),groups=[dict(indices=[0],lower=-.4,upper=.4)],
        decode=lambda x:x,fixed=fixed,config=None,model=SimpleNamespace(neuron_ids=ids),factory=factory,gains=np.ones(3),root=tmp_path))
    monkeypatch.setattr(verifier,'verify_inputs',lambda root:None)
    monkeypatch.setattr(network,'save_network',lambda path,*args:path.write_bytes(b'model'))
    monkeypatch.setattr(worker.shutil,'disk_usage',lambda path:SimpleNamespace(free=1024**4))
    until=time.monotonic()+60
    for mode in ('relay-replay','rial-replay'):
        r=worker.evaluate(dict(z=[0.],output=str(tmp_path/mode),until=until,target=.05,objective='sensory-motion',mode=mode))
        assert r['status']=='completed' and r['mediation']['cue_attenuation_reduced']
    assert seen==[(ids.index('AIYL'),)]*2+[(ids.index('RIAL'),)*2]*2


def test_curriculum_coordinator_keeps_state_and_runs_all_final_controls(tmp_path,monkeypatch):
    import concurrent.futures,hashlib,time
    import scripts.run_sensory_motion as runner
    import scripts.benchmark_cpu as topology
    from test_relay_cap_experiments import design
    from r_elegans.cma_curriculum import decode_groups
    ids,spec,origin,lower,upper,edge=design()
    schemas={'aiyr_2p5':parameter_groups(spec,ids,origin,lower,upper)}
    out=tmp_path/'run';out.mkdir();(out/'deadline.json').write_text(json.dumps(dict(coordinator_finish_by=time.monotonic()+20000)))
    parent_sha='test-parent';pair=data(effect=.1)
    monkeypatch.setattr(runner,'prepare',lambda source,out,experiment,bilateral,pathway=False:(tmp_path/'inputs',None,schemas,edge,parent_sha,'audit-path'))
    monkeypatch.setattr(runner,'capture_run_record',lambda *args:None)
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda pid:set(range(32)),raising=False)
    monkeypatch.setattr(topology,'physical_core_cpus',lambda allowed:(list(range(32)),True))
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    requests=[]
    def fake(j):
        requests.append(j);folder=__import__('pathlib').Path(j['output']);folder.mkdir(parents=True)
        np.savez(folder/'coordinates.npz',z=j['z'])
        sha=parent_sha if not np.any(j['z']) else hashlib.sha256(np.array(j['z']).tobytes()).hexdigest()
        m=score(pair,j['parent_features'] or pair['no_cue'],j['target'],j['margin'])
        return dict(output=j['output'],status='completed',checkpoint_sha256=sha,metrics=m,paired_features=pair,
            cap='aiyr_2p5',kind=j['kind'],grid=j['grid'],mediation=dict(cue_attenuation_reduced=True,baseline_preserved=True))
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,func,j):
            f=concurrent.futures.Future();f.set_result(fake(j));return f
    monkeypatch.setattr(runner.concurrent.futures,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(sys,'argv',['suite','--output-dir',str(out),'--source-run',str(tmp_path/'source'),'--generation-limit','1'])
    assert runner.main()==0
    s=json.loads((out/'summary.json').read_text())
    assert s['completed_generations']==1 and s['all_stages_finished']
    assert len([j for j in requests if j['kind']=='final'])==11
    assert {j['mode'] for j in requests if j['kind']=='final'}=={'normal','sensory-blocked','relay-replay','rial-replay'}
    assert not s['full_start_stop_qualified']
    assert (out/'optimizer/motion-seed-9300/final/optimizer.pkl').exists()


def test_expanded_parent_reconstructs_exactly_and_cannot_widen_other_bounds(tmp_path,monkeypatch):
    from pathlib import Path
    from scripts import sensory_motion_worker as motion_worker
    from scripts import rial_sensory_worker as core
    from r_elegans.brain.conductance_network import load_network,save_network
    from r_elegans.brain.joint_autonomous_fit import joint_fit_reference,apply_joint_offsets
    from scripts.assess_sensorimotor_resolution import promote_floating_values
    from r_elegans.relay_cap_experiments import edge_coordinate
    params,fixed,config=load_network(Path('r_elegans/assets/autonomous_wiggle_v1/network.npz'))
    params,fixed=promote_floating_values((params,fixed));_,spec=joint_fit_reference(fixed)
    with np.load('r_elegans/assets/autonomous_wiggle_v1/network.npz',allow_pickle=False) as z:ids=tuple(z['neuron_ids'])
    edge=edge_coordinate(spec,ids);old=np.zeros(spec.size);old[edge]=np.log(2)-.0001
    origin=old.copy();origin[edge]+=np.log(1.25);lower=np.full(spec.size,-np.log(2));upper=-lower
    np.savez(tmp_path/'cap_prior.npz',origin=origin,lower=lower,upper=upper,edge=edge)
    parent=apply_joint_offsets(params,origin-old,spec);save_network(tmp_path/'motion_parent.npz',parent,fixed,config)
    monkeypatch.setattr(core,'setup',lambda inputs:dict(base=params,fixed=fixed,config=config,spec=spec,origin=old))
    d=motion_worker.setup(tmp_path)
    assert origin[edge]>upper[edge]
    assert all(np.array_equal(a,b) for a,b in zip(d['decode'](origin),parent))
    bad=origin.copy();bad[0]=upper[0]+.001
    with pytest.raises(ValueError):d['decode'](bad)
    bad=origin.copy();bad[edge]=upper[edge]+np.log(1.25)+.001
    with pytest.raises(ValueError):d['decode'](bad)
    altered=parent._replace(log_channel_scale=parent.log_channel_scale.at[0,0].add(.1))
    save_network(tmp_path/'motion_parent.npz',altered,fixed,config)
    with pytest.raises(ValueError,match='reconstruction'):motion_worker.setup(tmp_path)


def test_scan_groups_widen_only_scanned_edges_in_the_real_network_worker(tmp_path,monkeypatch):
    from pathlib import Path
    from scripts import sensory_motion_worker as motion_worker
    from scripts import rial_sensory_worker as core
    from r_elegans.brain.conductance_network import load_network,save_network
    from r_elegans.brain.joint_autonomous_fit import joint_fit_reference,apply_joint_offsets
    from scripts.assess_sensorimotor_resolution import promote_floating_values
    from r_elegans.relay_cap_experiments import edge_coordinate
    from r_elegans.edge_scan import scan_groups,extend_upper
    params,fixed,config=load_network(Path('r_elegans/assets/autonomous_wiggle_v1/network.npz'))
    params,fixed=promote_floating_values((params,fixed));_,spec=joint_fit_reference(fixed)
    with np.load('r_elegans/assets/autonomous_wiggle_v1/network.npz',allow_pickle=False) as z:ids=tuple(z['neuron_ids'])
    edge=edge_coordinate(spec,ids);origin=np.zeros(spec.size);lower=np.full(spec.size,-np.log(2));upper=-lower
    np.savez(tmp_path/'cap_prior.npz',origin=origin,lower=lower,upper=upper,edge=edge)
    save_network(tmp_path/'motion_parent.npz',params,fixed,config)
    groups=scan_groups(spec,ids,origin,lower,upper);(tmp_path/'cap_groups.json').write_text(json.dumps(dict(scan=groups)))
    monkeypatch.setattr(core,'setup',lambda inputs:dict(base=params,fixed=fixed,config=config,spec=spec,origin=origin))
    d=motion_worker.setup(tmp_path);c=groups[0]['indices'][0]
    x=origin.copy();x[c]=np.log(4.)
    assert np.array_equal(d['upper'],extend_upper(upper,origin,groups)+np.where(np.arange(spec.size)==edge,np.log(1.25),0.))
    assert d['decode'](x) is not None
    bad=origin.copy();bad[c]=np.log(4.)+.001
    with pytest.raises(ValueError):d['decode'](bad)
    other=origin.copy();other[0]=np.log(4.)
    with pytest.raises(ValueError):d['decode'](other)


def test_parent_selection_requires_latest_clean_audit(tmp_path):
    from scripts.run_sensory_motion import prepare,E091_PARENT
    source=tmp_path/'source';source.mkdir();out=tmp_path/'out';out.mkdir()
    (source/'summary.json').write_text(json.dumps(dict(experiment='E092',execution_valid=True,validation_complete=True)))
    (source/'configuration.json').write_text(json.dumps(dict(parent_sha256=E091_PARENT)))
    audit=tmp_path/'source-audit';audit.mkdir();(audit/'summary.json').write_text(json.dumps(dict(audit_complete=True,integrity_passed=False,problems=['bad'])))
    with pytest.raises(ValueError,match='audit must pass'):prepare(source,out)
    assert not (out/'inputs').exists()


def test_prepare_selects_audited_isolated_parent_pins_hash_and_detects_changed_coordinates(tmp_path,monkeypatch):
    import scripts.run_sensory_motion as runner
    from pathlib import Path
    from scripts.experiment_records import file_sha256
    from r_elegans.brain.conductance_network import load_network,save_network
    from r_elegans.brain.joint_autonomous_fit import joint_fit_reference,apply_joint_offsets
    from scripts.assess_sensorimotor_resolution import promote_floating_values
    from r_elegans.relay_cap_experiments import edge_coordinate,upper_for
    source=tmp_path/'source';inputs=source/'inputs';inputs.mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value))
    params,fixed,config=load_network(Path('r_elegans/assets/autonomous_wiggle_v1/network.npz'))
    params,fixed=promote_floating_values((params,fixed));_,spec=joint_fit_reference(fixed)
    with np.load('r_elegans/assets/autonomous_wiggle_v1/network.npz',allow_pickle=False) as z:ids=tuple(z['neuron_ids'])
    edge=edge_coordinate(spec,ids);old=np.zeros(spec.size);old[edge]=np.log(2)-.0001
    origin=old.copy();origin[edge]+=np.log(1.25);lower=np.full(spec.size,-np.log(2));upper=-lower
    save_network(inputs/'sensory_parent.npz',params,fixed,config)
    np.savez(inputs/'cap_prior.npz',origin=old,lower=lower,upper=upper,edge=edge)
    write(inputs/'manifest.json',dict(sha256={p.name:file_sha256(p) for p in inputs.iterdir()},sources={}))
    parent_hash=file_sha256(inputs/'sensory_parent.npz');monkeypatch.setattr(runner,'E091_PARENT',parent_hash)
    write(source/'configuration.json',dict(parent_sha256=parent_hash))
    folder=source/'evaluations/final-single-edge';folder.mkdir(parents=True)
    parent=apply_joint_offsets(params,origin-old,spec);save_network(folder/'network.npz',parent,fixed,config)
    np.savez(folder/'coordinates.npz',offsets=origin,z=[0.]);(folder/'no_cue.npz').write_bytes(b'reference trace hash fixture')
    sha=file_sha256(folder/'network.npz')
    row=dict(output=str(folder),checkpoint_sha256=sha,variant='nominal',grid='ordinary',mode='normal',trajectory_sha256=dict(no_cue=file_sha256(folder/'no_cue.npz')))
    panel=dict(labels=['best-single-edge'],cap='aiyr_2p5',assessment=dict(engineering_criteria_met=True),checkpoint_sha256=sha,validation=[row])
    write(source/'summary.json',dict(experiment='E092',execution_valid=True,validation_complete=True,panels=[panel],scan=[dict(output=str(source/'evaluations/factor-1.25'),checkpoint_sha256=sha)]))
    audit=source.with_name('source-audit');audit.mkdir()
    write(audit/'summary.json',dict(audit_complete=True,integrity_passed=True,problems=[],panels=[{k:panel[k] for k in ('labels','cap','assessment')}]))
    write(audit/'evidence.json',[dict(path=str(folder/name),sha256=file_sha256(folder/name)) for name in ('network.npz','coordinates.npz','no_cue.npz')])
    monkeypatch.setattr(runner,'setup',lambda inputs:dict(spec=spec,model=SimpleNamespace(neuron_ids=ids),lower=lower,upper=upper_for('aiyr_2p5',upper,edge)))
    out=tmp_path/'out';out.mkdir()
    prepared,d,schemas,got_edge,pinned,ap=runner.prepare(source,out)
    assert got_edge==edge and pinned==sha and list(schemas)==['aiyr_2p5']
    with np.load(prepared/'cap_prior.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['origin'],origin);np.testing.assert_array_equal(z['upper'],upper)
    np.savez(folder/'coordinates.npz',offsets=origin,z=[.1])
    other=tmp_path/'other';other.mkdir()
    with pytest.raises(ValueError,match='corrected audit evidence'):runner.prepare(source,other)
