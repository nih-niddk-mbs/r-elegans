import concurrent.futures,hashlib,json,sys,time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from r_elegans.cma_curriculum import decode_groups
from r_elegans.pathway import (HOP1,NODE_TARGET,MAX_FACTOR,pathway_edges,pathway_groups,hop1_with_edges,extend_pathway_upper,widened_coordinates)
from r_elegans.relay_cap_experiments import edge_coordinate,upper_for
from r_elegans.sensory_motion import score
from test_sensory_motion import data

SIDES=('L','R')
IDS=tuple(['AWAL','AWAR','RIAL','RIAR','AIYL','AIYR']+[n+s for n in HOP1 if n not in ('AIY',) for s in SIDES]+
    ['SMDDL','SMDDR','SMDVL','SMDVR','RMDDL','RMDDR','RMDL','RMDR','RMDVL','RMDVR'])


def toy(mask_fn=None,room=1.):
    ids=IDS;n=len(ids);channels=np.ones((n,17),bool);channels[:,16]=False
    chem=np.ones((n,n),bool) if mask_fn is None else mask_fn(ids)
    spec=SimpleNamespace(channels=channels,chemical=chem);size=int(channels.sum()+chem.sum())
    upper=np.full(size,np.log(2.));lower=np.full(size,-room);origin=np.zeros(size)
    return ids,spec,origin,lower,upper


def with_nodes(pair,change=.02,level=.3,band=(.2,.4)):
    pair=deepcopy(pair)
    for condition,rows in pair.items():
        for category in ('active','response','late'):
            for row in rows[category]:
                bump=change if condition=='cue' and category=='response' else 0.
                row['node_release_mean']={'RIAL':level+bump,'RIAR':level+bump};row['node_release_q01_q99']={'RIAL':list(band),'RIAR':list(band)}
    return pair


def test_pathway_edges_cover_awa_to_hop1_to_ria_and_ria_to_motor_once_each():
    ids,spec,origin,lower,upper=toy()
    edges=pathway_edges(spec,ids);assert len(edges)==len(set(edges))
    assert ('AWAL','AIZL') in edges and ('AIZR','RIAR') in edges and ('RIAL','SMDVL') in edges and ('RIAR','RMDDL') in edges
    assert not any(e[0].startswith('SMD') for e in edges) and len(hop1_with_edges(spec,ids))==2*len(HOP1)
    only_aiz=lambda ids_:np.array([[ (post in ('AIZL','AIZR','RIAL','RIAR') and pre in ('AWAL','AWAR','AIZL','AIZR')) for pre in ids_] for post in ids_])
    i2,s2,*_=toy(only_aiz);assert {h[:3] for h in hop1_with_edges(s2,i2)}=={'AIZ'}


def test_pathway_groups_widen_only_edges_and_keep_densities_inside_the_original_prior():
    ids,spec,origin,lower,upper=toy()
    groups=pathway_groups(spec,ids,origin,lower,upper)
    edge_groups=[g for g in groups if g.get('widened')];density=[g for g in groups if not g.get('widened')]
    assert len(edge_groups)==len(pathway_edges(spec,ids)) and len(density)==16
    assert all(g['upper']==pytest.approx(np.log(MAX_FACTOR)) and g['lower']==pytest.approx(-1.) and len(g['indices'])==1 for g in edge_groups)
    assert all(g['upper']<=.4+1e-12 and g['lower']>=-.4-1e-12 for g in density)
    used=[i for g in groups for i in g['indices']];assert len(used)==len(set(used))
    allowed=extend_pathway_upper(upper,origin,groups)
    assert set(np.flatnonzero(allowed!=upper))==widened_coordinates(groups) and np.all(allowed>=upper)
    for g in edge_groups[:3]:
        x=decode_groups([1. if i==groups.index(g) else 0. for i in range(len(groups))],groups,origin,lower,allowed)
        assert np.exp(x[g['indices'][0]])==pytest.approx(MAX_FACTOR)


def test_pathway_groups_reject_missing_neurons_and_skip_inherited_exception_coordinates():
    ids,spec,origin,lower,upper=toy()
    with pytest.raises(ValueError):pathway_groups(spec,tuple(n for n in ids if n!='AIZL'),origin,lower,upper)
    from r_elegans.pathway import edge_coordinate as ec
    coordinate=ec(spec,ids,'AWAL','AIZL');origin[coordinate]=upper[coordinate]+.1
    names=[g['name'] for g in pathway_groups(spec,ids,origin,lower,upper)]
    assert 'chemical:AWAL->AIZL' not in names and 'chemical:AWAR->AIZR' in names


def test_pathway_groups_exist_in_the_real_network_and_the_worker_accepts_a_four_fold_widened_edge(tmp_path,monkeypatch):
    from scripts import sensory_motion_worker as motion_worker
    from scripts import rial_sensory_worker as core
    from r_elegans.brain.conductance_network import load_network,save_network
    from r_elegans.brain.joint_autonomous_fit import joint_fit_reference
    from scripts.assess_sensorimotor_resolution import promote_floating_values
    params,fixed,config=load_network(Path('r_elegans/assets/autonomous_wiggle_v1/network.npz'))
    params,fixed=promote_floating_values((params,fixed));_,spec=joint_fit_reference(fixed)
    with np.load('r_elegans/assets/autonomous_wiggle_v1/network.npz',allow_pickle=False) as z:ids=tuple(z['neuron_ids'])
    edge=edge_coordinate(spec,ids);origin=np.zeros(spec.size);lower=np.full(spec.size,-np.log(2));upper=-lower
    groups=pathway_groups(spec,ids,origin,lower,upper)
    assert sum(1 for g in groups if g.get('widened'))==37 and len(groups)==53 and {e[1] for e in pathway_edges(spec,ids) if e[0].startswith('AWA')}>={'AIZL','AIZR','AIYL','AIYR','ADFR'}
    np.savez(tmp_path/'cap_prior.npz',origin=origin,lower=lower,upper=upper,edge=edge);save_network(tmp_path/'motion_parent.npz',params,fixed,config)
    (tmp_path/'cap_groups.json').write_text(json.dumps(dict(pathway=groups)))
    monkeypatch.setattr(core,'setup',lambda inputs:dict(base=params,fixed=fixed,config=config,spec=spec,origin=origin))
    d=motion_worker.setup(tmp_path);c=next(g for g in groups if g.get('widened'))['indices'][0]
    x=origin.copy();x[c]=np.log(4.);assert d['decode'](x) is not None
    bad=x.copy();bad[c]=np.log(4.)+.01
    with pytest.raises(ValueError):d['decode'](bad)
    density=next(g for g in groups if not g.get('widened'))['indices'][0];other=origin.copy();other[density]=np.log(4.)
    with pytest.raises(ValueError):d['decode'](other)


def test_node_term_rewards_a_larger_ria_release_change_and_penalizes_a_railed_node():
    pair=data(effect=.05);parent=pair['no_cue']
    none=score(pair,parent,.05,1.05)
    assert 'node_term' not in none
    low=score(with_nodes(pair,change=.002),parent,.05,1.05,node_target=NODE_TARGET)
    high=score(with_nodes(pair,change=.03),parent,.05,1.05,node_target=NODE_TARGET)
    assert low['node_release_change']==pytest.approx(.002) and high['node_term']==0. and low['node_term']>.5
    assert low['loss']>high['loss'] and high['loss']==pytest.approx(none['loss']) and not low['node_headroom_penalty']
    railed=score(with_nodes(pair,change=.03,level=.97,band=(.9,.995)),parent,.05,1.05,node_target=NODE_TARGET)
    assert railed['node_headroom_penalty']>0 and railed['loss']>high['loss']


def test_e098_coordinator_trains_the_pathway_with_a_node_target_and_population_of_21(tmp_path,monkeypatch):
    import scripts.run_sensory_motion as runner
    import scripts.benchmark_cpu as topology
    ids,spec,origin,lower,upper=toy();groups=pathway_groups(spec,ids,origin,lower,upper)
    out=tmp_path/'run';out.mkdir();(out/'deadline.json').write_text(json.dumps(dict(coordinator_finish_by=time.monotonic()+20000)))
    parent_sha='test-parent';pair=with_nodes(data(effect=.1))
    d=dict(spec=spec,model=SimpleNamespace(neuron_ids=ids))
    monkeypatch.setattr(runner,'prepare',lambda source,o,experiment,bilateral,pathway=False:(tmp_path/'inputs',d,{'pathway':groups},0,parent_sha,'audit-path') if pathway and experiment=='E098' else 1/0)
    monkeypatch.setattr(runner,'capture_run_record',lambda *args:None)
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda pid:set(range(128)),raising=False)
    monkeypatch.setattr(topology,'physical_core_cpus',lambda allowed:(list(range(128)),True))
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    requests=[]
    def fake(j):
        requests.append(j);folder=Path(j['output']);folder.mkdir(parents=True);np.savez(folder/'coordinates.npz',z=j['z'])
        sha=parent_sha if not np.any(j['z']) else hashlib.sha256(np.array(j['z']).tobytes()).hexdigest()
        m=score(pair,j['parent_features'] or pair['no_cue'],j['target'],j['margin'],worst_region=j['worst_region'],node_target=j.get('node_target'))
        return dict(output=j['output'],status='completed',checkpoint_sha256=sha,metrics=m,paired_features=pair,cap=j['cap'],kind=j['kind'],grid=j['grid'],
            mediation=dict(cue_attenuation_reduced=True,baseline_preserved=True))
    class Pool:
        def __init__(self,**kwargs):self.kwargs=kwargs
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,func,j):
            f=concurrent.futures.Future();f.set_result(fake(j));return f
    monkeypatch.setattr(runner.concurrent.futures,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(sys,'argv',['suite','--output-dir',str(out),'--source-run',str(tmp_path/'source'),'--profile','e098','--generation-limit','1'])
    assert runner.main()==0
    s=json.loads((out/'summary.json').read_text());config=json.loads((out/'configuration.json').read_text())
    assert config['pathway'] and config['node_target']==NODE_TARGET and config['population']==21 and config['seed']==9800 and config['workers']==21
    assert config['pathway_hop1']==hop1_with_edges(spec,ids)
    assert len([j for j in requests if j['kind']=='fit'])==21 and all(j['node_target']==NODE_TARGET and j['pathway_hop1'] and j['cap']=='pathway' for j in requests)
    assert len([j for j in requests if j['kind']=='final'])==11 and s['experiment']=='E098' and s['all_stages_finished']
    assert (out/'optimizer/motion-seed-9800/final/optimizer.pkl').exists() and not s['full_start_stop_qualified']


def test_pathway_pipeline_dispatches_the_e098_profile(tmp_path,monkeypatch):
    import scripts.total_job_budget as budget
    out=tmp_path/'out';calls=[]
    def run(cmd,seconds):
        calls.append(cmd)
        if 'scripts.audit_relay_experiments' in cmd:
            audit=tmp_path/'out-audit';audit.mkdir();(audit/'summary.json').write_text(json.dumps(dict(audit_complete=True,integrity_passed=True)))
        else:(out/'summary.json').write_text(json.dumps(dict(execution_valid=True,all_stages_finished=True,validation_complete=True,sensory_motion_component_qualified=False)))
        return 0,False
    monkeypatch.setattr(budget,'run_group',run)
    monkeypatch.setattr(sys,'argv',['budget','--pipeline','pathway-training','--total-minutes','480','--output-dir',str(out),'--source-run',str(tmp_path/'source')])
    assert budget.main()==0 and calls[0][calls[0].index('--profile')+1]=='e098' and 'scripts.run_sensory_motion' in calls[0] and 'scripts.audit_relay_experiments' in calls[1]
    assert json.loads((out/'deadline.json').read_text())['experiment']=='E098'


def test_saved_data_audit_accepts_a_widened_pathway_edge_and_rejects_a_second_changed_coordinate(tmp_path):
    from scripts.audit_relay_experiments import audit
    from scripts.experiment_records import file_sha256
    ids,spec,origin,lower,upper=toy();n=len(ids);nc=int(np.count_nonzero(spec.channels))
    edge=nc+int(np.flatnonzero((np.nonzero(spec.chemical)[0]==ids.index('AIYL'))&(np.nonzero(spec.chemical)[1]==ids.index('AIYR')))[0])
    groups=pathway_groups(spec,ids,origin,lower,upper);allowed_base=upper_for('aiyr_2p5',upper,edge)
    run=tmp_path/'run';inputs=run/'inputs';inputs.mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value))
    base=dict(neuron_ids=np.array(ids),cells_conductances_nS=np.ones((n,17)),fixed_chemical_mask=spec.chemical,params_log_channel_scale=np.zeros((n,17)),
        params_raw_chemical=np.zeros((n,n)),fixed_release_threshold_mV=np.full(n,-40.),fixed_release_slope_mV=np.full(n,5.))
    np.savez(inputs/'motion_parent.npz',**base);np.savez(inputs/'motion_parent_coordinates.npz',offsets=origin)
    np.savez(inputs/'source_bounds.npz',lower=lower,upper=upper);np.savez(inputs/'cap_prior.npz',origin=origin,lower=lower,upper=upper,edge=edge)
    write(inputs/'cap_groups.json',dict(pathway=groups))
    reference=tmp_path/'external-trace';reference.write_bytes(b'reference evidence')
    write(inputs/'sensory_reference.json',dict(path=str(reference),sha256=file_sha256(reference)))
    write(inputs/'manifest.json',dict(sha256={p.name:file_sha256(p) for p in inputs.iterdir()}))
    pair=with_nodes(data(effect=.1));write(run/'parent_features.json',pair['no_cue'])
    write(run/'configuration.json',dict(parent_sha256=file_sha256(inputs/'motion_parent.npz'),only_extended_coordinate=edge,node_target=NODE_TARGET,pathway=True))
    def raw_for(updates):
        raw=np.zeros((n,n))
        for (post,pre),factor in updates.items():raw[post,pre]=np.log(np.expm1(np.log(2.)*factor))
        return raw
    def evaluation(label,z,updates):
        folder=run/'evaluations'/label;folder.mkdir(parents=True)
        x=decode_groups(z,groups,origin,lower,extend_pathway_upper(allowed_base,origin,groups));model=dict(base);model['params_raw_chemical']=raw_for(updates)
        np.savez(folder/'network.npz',**model);np.savez(folder/'coordinates.npz',offsets=x,z=z)
        request=dict(output=str(folder),cap='pathway',kind='fit',z=list(z),retain_traces=False,variant='nominal',mode='normal',grid='ordinary',reference=False,
            target=.05,margin=1.05,worst_region=True,node_target=NODE_TARGET,pathway_hop1=['AIZL'],parent_features=pair['no_cue'])
        write(folder/'request.json',request)
        r=dict(status='completed',output=str(folder),checkpoint_sha256=file_sha256(folder/'network.npz'),coordinates_sha256=file_sha256(folder/'coordinates.npz'),
            paired_features=pair,metrics=score(pair,pair['no_cue'],.05,1.05,worst_region=True,node_target=NODE_TARGET),trajectory_sha256={})
        write(folder/'summary.json',r);return r
    def summarize(results):
        write(run/'summary.json',dict(experiment='E098',evaluation_count=len(results),panels=[],scan=results,scan_controls=[],arms={},completed_generations=0,
            sensory_motion_component_qualified=False,rest_restart_component_qualified=False,full_start_stop_qualified=False,swimming_qualified=False))
    z0=[0.]*len(groups);z1=list(z0);z1[0]=1.
    c0=groups[0]['indices'][0]-nc;edge0=divmod(c0,n);c1=groups[1]['indices'][0]-nc;edge1=divmod(c1,n)
    results=[evaluation('parent',z0,{}),evaluation('widened',z1,{edge0:4.})];summarize(results)
    assert audit(run,tmp_path/'audit')==0
    folder=Path(results[1]['output']);model=dict(base);model['params_raw_chemical']=raw_for({edge0:4.,edge1:1.3})
    np.savez(folder/'network.npz',**model);results[1]['checkpoint_sha256']=file_sha256(folder/'network.npz');write(folder/'summary.json',results[1]);summarize(results)
    assert audit(run,tmp_path/'bad-audit')==2
    assert any('Frozen model changed' in p or 'Physical coordinate mismatch' in p for p in json.loads((tmp_path/'bad-audit/summary.json').read_text())['problems'])
