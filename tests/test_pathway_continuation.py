import concurrent.futures,hashlib,json,sys,time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from r_elegans.cma_curriculum import decode_groups
from r_elegans.pathway import NODE_TARGET,pathway_groups,extend_pathway_upper,hop1_with_edges,edge_coordinate as pathway_edge_coordinate,pathway_edges
from r_elegans.relay_cap_experiments import upper_for,edge_coordinate
from r_elegans.sensory_motion import score
from test_sensory_motion import data
from test_pathway import toy,with_nodes,IDS


def asym(pair,left=.005,right=.03):
    pair=deepcopy(pair)
    for condition,rows in pair.items():
        for category in ('active','response','late'):
            for row in rows[category]:
                bump=condition=='cue' and category=='response'
                row['node_release_mean']={'RIAL':.3+(left if bump else 0.),'RIAR':.3+(right if bump else 0.)};row['node_release_q01_q99']={'RIAL':[.2,.4],'RIAR':[.2,.4]}
    return pair


def test_weaker_side_mode_rewards_bilateral_drive_and_leaves_the_mean_mode_unchanged():
    pair=data(effect=.05);parent=pair['no_cue'];one_sided=asym(pair)
    mean=score(one_sided,parent,.05,1.05,node_target=NODE_TARGET);weak=score(one_sided,parent,.05,1.05,node_target=NODE_TARGET,node_mode='min')
    assert mean['node_release_change']==pytest.approx(.0175) and 'node_mode' not in mean and 'node_release_changes' not in mean
    assert weak['node_release_change']==pytest.approx(.005) and weak['node_mode']=='min' and weak['node_release_changes']=={'RIAL':pytest.approx(.005),'RIAR':pytest.approx(.03)}
    assert weak['node_term']>mean['node_term'] and weak['loss']>mean['loss']
    both=score(asym(pair,left=.03,right=.03),parent,.05,1.05,node_target=NODE_TARGET,node_mode='min')
    assert both['node_term']==0.


def test_continuation_groups_keep_edges_that_already_exceed_the_original_prior_and_widen_relative_to_the_new_origin():
    ids,spec,origin,lower,upper=toy()
    c=pathway_edge_coordinate(spec,ids,'AWAR','AIZR');origin[c]=np.log(3.)
    default=[g['name'] for g in pathway_groups(spec,ids,origin,lower,upper)]
    assert 'chemical:AWAR->AIZR' not in default
    groups=pathway_groups(spec,ids,origin,lower,upper,inherited=-1)
    g=next(x for x in groups if x['name']=='chemical:AWAR->AIZR')
    assert g['upper']==pytest.approx(np.log(4.)) and g['lower']==pytest.approx(-np.log(4.)) and g['widened']
    allowed=extend_pathway_upper(upper,origin,groups)
    assert allowed[c]==pytest.approx(origin[c]+np.log(4.)) and np.exp(allowed[c]-origin[c])==pytest.approx(4.)
    inherited=pathway_edge_coordinate(spec,ids,'AWAL','AIZL')
    assert 'chemical:AWAL->AIZL' not in [x['name'] for x in pathway_groups(spec,ids,origin,lower,upper,inherited=inherited)]


def test_worker_rebuilds_a_continuation_parent_from_the_lineage_parent_exactly(tmp_path,monkeypatch):
    from scripts import sensory_motion_worker as motion_worker
    from scripts import rial_sensory_worker as core
    from r_elegans.brain.conductance_network import load_network,save_network
    from r_elegans.brain.joint_autonomous_fit import joint_fit_reference,apply_joint_offsets
    from scripts.assess_sensorimotor_resolution import promote_floating_values
    params,fixed,config=load_network(Path('r_elegans/assets/autonomous_wiggle_v1/network.npz'))
    params,fixed=promote_floating_values((params,fixed));_,spec=joint_fit_reference(fixed)
    with np.load('r_elegans/assets/autonomous_wiggle_v1/network.npz',allow_pickle=False) as z:ids=tuple(z['neuron_ids'])
    edge=edge_coordinate(spec,ids);rng=np.random.RandomState(3);off1=np.zeros(spec.size);off2=np.zeros(spec.size)
    off1[rng.choice(spec.size,40,replace=False)]=rng.uniform(-.2,.2,40);off2[rng.choice(spec.size,30,replace=False)]=rng.uniform(-.3,.3,30)
    lineage=apply_joint_offsets(params,off1,spec);selected=apply_joint_offsets(lineage,off2,spec);origin=off1+off2
    lower=np.full(spec.size,-2.);upper=-lower
    np.savez(tmp_path/'cap_prior.npz',origin=origin,lower=lower,upper=upper,edge=edge)
    save_network(tmp_path/'lineage_parent.npz',lineage,fixed,config);np.savez(tmp_path/'lineage_parent_coordinates.npz',offsets=off1)
    save_network(tmp_path/'motion_parent.npz',selected,fixed,config)
    monkeypatch.setattr(core,'setup',lambda inputs:dict(base=params,fixed=fixed,config=config,spec=spec,origin=np.zeros(spec.size)))
    d=motion_worker.setup(tmp_path)
    assert all(np.array_equal(a,b) for a,b in zip(d['decode'](origin),selected))
    altered=selected._replace(log_channel_scale=selected.log_channel_scale.at[0,0].add(.1));save_network(tmp_path/'motion_parent.npz',altered,fixed,config)
    with pytest.raises(ValueError,match='reconstruction'):motion_worker.setup(tmp_path)


def test_prepare_continuation_pins_the_audited_e098_checkpoint_and_keeps_the_e092_lineage(tmp_path,monkeypatch):
    import scripts.run_sensory_motion as runner
    from scripts.experiment_records import file_sha256
    from r_elegans.brain.conductance_network import load_network,save_network
    from r_elegans.brain.joint_autonomous_fit import joint_fit_reference,apply_joint_offsets
    from scripts.assess_sensorimotor_resolution import promote_floating_values
    params,fixed,config=load_network(Path('r_elegans/assets/autonomous_wiggle_v1/network.npz'))
    params,fixed=promote_floating_values((params,fixed));_,spec=joint_fit_reference(fixed)
    with np.load('r_elegans/assets/autonomous_wiggle_v1/network.npz',allow_pickle=False) as z:ids=tuple(z['neuron_ids'])
    edge=edge_coordinate(spec,ids);lower=np.full(spec.size,-np.log(2));upper=-lower
    c1=np.zeros(spec.size);c1[edge]=np.log(2)-.0001+np.log(1.25);c=pathway_edge_coordinate(spec,ids,'AWAR','AIZR')
    c2=c1.copy();c2[c]=np.log(3.);dens=pathway_groups(spec,ids,c1,lower,upper_for('aiyr_2p5',upper,edge))[-1]['indices'][0];c2[dens]+=.05
    source=tmp_path/'source';inputs=source/'inputs';inputs.mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value))
    lineage=apply_joint_offsets(params,c1,spec);save_network(inputs/'motion_parent.npz',lineage,fixed,config);np.savez(inputs/'motion_parent_coordinates.npz',offsets=c1)
    np.savez(inputs/'cap_prior.npz',origin=c1,lower=lower,upper=upper,edge=edge)
    write(inputs/'manifest.json',dict(sha256={p.name:file_sha256(p) for p in inputs.iterdir()},sources={}))
    write(source/'parent_features.json',dict(anchor='E092'))
    folder=source/'evaluations/final-00';folder.mkdir(parents=True)
    selected=apply_joint_offsets(lineage,c2-c1,spec);save_network(folder/'network.npz',selected,fixed,config)
    np.savez(folder/'coordinates.npz',offsets=c2,z=[0.]);(folder/'no_cue.npz').write_bytes(b'selected no-cue trace fixture')
    sha=file_sha256(folder/'network.npz')
    row=dict(output=str(folder),checkpoint_sha256=sha,variant='nominal',grid='ordinary',mode='normal',trajectory_sha256=dict(no_cue=file_sha256(folder/'no_cue.npz')))
    write(source/'summary.json',dict(experiment='E098',execution_valid=True,validation_complete=True,panels=[dict(labels=['motion-seed-9800'],cap='pathway',checkpoint_sha256=sha,validation=[row])]))
    write(source/'budget_receipt.json',dict(all_steps_finished=True))
    audit=source.with_name('source-audit');audit.mkdir();write(audit/'summary.json',dict(audit_complete=True,integrity_passed=False,problems=['bad']))
    write(audit/'evidence.json',[dict(path=str(folder/n),sha256=file_sha256(folder/n)) for n in ('network.npz','coordinates.npz','no_cue.npz')])
    monkeypatch.setattr(runner,'setup',lambda i:dict(spec=spec,model=SimpleNamespace(neuron_ids=ids),lower=lower,upper=upper_for('aiyr_2p5',upper,edge)))
    out=tmp_path/'out';out.mkdir()
    with pytest.raises(ValueError,match='audit must pass'):runner.prepare_continuation(source,out)
    assert not (out/'inputs').exists()
    write(audit/'summary.json',dict(audit_complete=True,integrity_passed=True,problems=[]))
    prepared,d,schemas,got_edge,pinned,ap=runner.prepare_continuation(source,out)
    assert got_edge==edge and pinned==sha and list(schemas)==['pathway']
    with np.load(prepared/'cap_prior.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['origin'],c2)
    with np.load(prepared/'lineage_parent_coordinates.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['offsets'],c1)
    assert file_sha256(prepared/'motion_parent.npz')==sha and json.loads((prepared/'anchor_parent_features.json').read_text())==dict(anchor='E092')
    names={g['name']:g for g in schemas['pathway']}
    assert names['chemical:AWAR->AIZR']['upper']==pytest.approx(np.log(4.)) and 'chemical:AWAR->AIZR' in names
    assert json.loads((prepared/'manifest.json').read_text())['consumer_experiment']=='E099'
    np.savez(folder/'coordinates.npz',offsets=c2,z=[.1])
    other=tmp_path/'other';other.mkdir()
    with pytest.raises(ValueError,match='audit evidence'):runner.prepare_continuation(source,other)


def test_e099_coordinator_continues_from_the_checkpoint_with_the_weaker_side_term_and_the_e092_anchor(tmp_path,monkeypatch):
    import scripts.run_sensory_motion as runner
    import scripts.benchmark_cpu as topology
    ids,spec,origin,lower,upper=toy();groups=pathway_groups(spec,ids,origin,lower,upper)
    out=tmp_path/'run';out.mkdir();(out/'deadline.json').write_text(json.dumps(dict(coordinator_finish_by=time.monotonic()+20000)))
    inputs=tmp_path/'inputs';inputs.mkdir();parent_sha='checkpoint';pair=asym(data(effect=.1),left=.02,right=.02)
    anchor=data(effect=.1)['no_cue'];(inputs/'anchor_parent_features.json').write_text(json.dumps(anchor))
    d=dict(spec=spec,model=SimpleNamespace(neuron_ids=ids))
    monkeypatch.setattr(runner,'prepare_continuation',lambda source,o,experiment:(inputs,d,{'pathway':groups},0,parent_sha,'audit-path') if experiment=='E099' else 1/0)
    monkeypatch.setattr(runner,'capture_run_record',lambda *args:None)
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda pid:set(range(128)),raising=False)
    monkeypatch.setattr(topology,'physical_core_cpus',lambda allowed:(list(range(128)),True))
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    requests=[]
    def fake(j):
        requests.append(j);folder=Path(j['output']);folder.mkdir(parents=True);np.savez(folder/'coordinates.npz',z=j['z'])
        sha=parent_sha if not np.any(j['z']) else hashlib.sha256(np.array(j['z']).tobytes()).hexdigest()
        m=score(pair,j['parent_features'] or pair['no_cue'],j['target'],j['margin'],worst_region=j['worst_region'],node_target=j.get('node_target'),node_mode=j.get('node_mode','mean'))
        return dict(output=j['output'],status='completed',checkpoint_sha256=sha,metrics=m,paired_features=pair,cap=j['cap'],kind=j['kind'],grid=j['grid'],
            mediation=dict(cue_attenuation_reduced=True,baseline_preserved=True))
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,func,j):
            f=concurrent.futures.Future();f.set_result(fake(j));return f
    monkeypatch.setattr(runner.concurrent.futures,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(sys,'argv',['suite','--output-dir',str(out),'--source-run',str(tmp_path/'e098'),'--profile','e099','--generation-limit','1'])
    assert runner.main()==0
    s=json.loads((out/'summary.json').read_text());config=json.loads((out/'configuration.json').read_text())
    assert config['continuation'] and config['node_mode']=='min' and config['seed']==9900 and config['population']==21 and s['experiment']=='E099'
    assert json.loads((out/'parent_features.json').read_text())==anchor
    assert all(j['node_mode']=='min' and j['node_target']==NODE_TARGET for j in requests)
    assert all(j['parent_features']==anchor for j in requests if not j['output'].endswith('/parent'))
    assert (out/'optimizer/motion-seed-9900/final/optimizer.pkl').exists() and len([j for j in requests if j['kind']=='final'])==11


def test_continuation_pipeline_dispatches_the_e099_profile(tmp_path,monkeypatch):
    import scripts.total_job_budget as budget
    out=tmp_path/'out';calls=[]
    def run(cmd,seconds):
        calls.append(cmd)
        if 'scripts.audit_relay_experiments' in cmd:
            audit=tmp_path/'out-audit';audit.mkdir();(audit/'summary.json').write_text(json.dumps(dict(audit_complete=True,integrity_passed=True)))
        else:(out/'summary.json').write_text(json.dumps(dict(execution_valid=True,all_stages_finished=True,validation_complete=True,sensory_motion_component_qualified=False)))
        return 0,False
    monkeypatch.setattr(budget,'run_group',run)
    monkeypatch.setattr(sys,'argv',['budget','--pipeline','pathway-continuation','--total-minutes','180','--output-dir',str(out),'--source-run',str(tmp_path/'e098')])
    assert budget.main()==0 and calls[0][calls[0].index('--profile')+1]=='e099' and json.loads((out/'deadline.json').read_text())['experiment']=='E099'


def test_audit_reconstructs_continuation_groups_and_accepts_an_edge_already_above_the_original_bound(tmp_path):
    from scripts.audit_relay_experiments import audit
    from scripts.experiment_records import file_sha256
    ids,spec,origin,lower,upper=toy();n=len(ids);nc=int(np.count_nonzero(spec.channels))
    pairs=(np.nonzero(spec.chemical)[0],np.nonzero(spec.chemical)[1])
    edge=nc+int(np.flatnonzero((pairs[0]==ids.index('AIYL'))&(pairs[1]==ids.index('AIYR')))[0])
    c_wide=pathway_edge_coordinate(spec,ids,'AWAR','AIZR');origin[c_wide]=np.log(3.)
    groups=pathway_groups(spec,ids,origin,lower,upper_for('aiyr_2p5',upper,edge),inherited=edge)
    allowed=extend_pathway_upper(upper_for('aiyr_2p5',upper,edge),origin,groups)
    run=tmp_path/'run';inputs=run/'inputs';inputs.mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value))
    def raw_for(updates):
        raw=np.zeros((n,n))
        for (post,pre),factor in updates.items():raw[post,pre]=np.log(np.expm1(np.log(2.)*factor))
        return raw
    base=dict(neuron_ids=np.array(ids),cells_conductances_nS=np.ones((n,17)),fixed_chemical_mask=spec.chemical,params_log_channel_scale=np.zeros((n,17)),
        params_raw_chemical=raw_for({divmod(c_wide-nc,n):3.}),fixed_release_threshold_mV=np.full(n,-40.),fixed_release_slope_mV=np.full(n,5.))
    np.savez(inputs/'motion_parent.npz',**base);np.savez(inputs/'motion_parent_coordinates.npz',offsets=origin)
    np.savez(inputs/'source_bounds.npz',lower=lower,upper=upper);np.savez(inputs/'cap_prior.npz',origin=origin,lower=lower,upper=upper,edge=edge)
    write(inputs/'cap_groups.json',dict(pathway=groups))
    reference=tmp_path/'external-trace';reference.write_bytes(b'reference evidence')
    write(inputs/'sensory_reference.json',dict(path=str(reference),sha256=file_sha256(reference)))
    write(inputs/'manifest.json',dict(sha256={p.name:file_sha256(p) for p in inputs.iterdir()}))
    pair=asym(data(effect=.1));write(run/'parent_features.json',pair['no_cue'])
    write(run/'configuration.json',dict(parent_sha256=file_sha256(inputs/'motion_parent.npz'),only_extended_coordinate=edge,node_target=NODE_TARGET,pathway=True,continuation=True,node_mode='min'))
    folder=run/'evaluations'/'parent';folder.mkdir(parents=True);z=[0.]*len(groups)
    x=decode_groups(z,groups,origin,lower,allowed);np.savez(folder/'network.npz',**base);np.savez(folder/'coordinates.npz',offsets=x,z=z)
    request=dict(output=str(folder),cap='pathway',kind='fit',z=z,retain_traces=False,variant='nominal',mode='normal',grid='ordinary',reference=False,target=.05,margin=1.05,
        worst_region=True,node_target=NODE_TARGET,node_mode='min',pathway_hop1=['AIZL'],parent_features=pair['no_cue'])
    write(folder/'request.json',request)
    result=dict(status='completed',output=str(folder),checkpoint_sha256=file_sha256(folder/'network.npz'),coordinates_sha256=file_sha256(folder/'coordinates.npz'),paired_features=pair,
        metrics=score(pair,pair['no_cue'],.05,1.05,worst_region=True,node_target=NODE_TARGET,node_mode='min'),trajectory_sha256={})
    write(folder/'summary.json',result)
    write(run/'summary.json',dict(experiment='E099',evaluation_count=1,panels=[],scan=[result],scan_controls=[],arms={},completed_generations=0,
        sensory_motion_component_qualified=False,rest_restart_component_qualified=False,full_start_stop_qualified=False,swimming_qualified=False))
    assert audit(run,tmp_path/'audit')==0
    cfg=json.loads((run/'configuration.json').read_text());cfg['continuation']=False;write(run/'configuration.json',cfg)
    assert audit(run,tmp_path/'bad-audit')==2
    assert any('Unexpected group policy' in p for p in json.loads((tmp_path/'bad-audit/summary.json').read_text())['problems'])
