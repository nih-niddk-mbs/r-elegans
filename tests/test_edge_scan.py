import json,sys,time,hashlib
import concurrent.futures
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from r_elegans.cma_curriculum import decode_groups
from r_elegans.edge_scan import EDGES,FACTORS,MAX_FACTOR,scan_groups,extend_upper,planned_cases,finalists,reduction_percent
from r_elegans.sensory_motion import score
from test_sensory_motion import data

IDS=('AWAL','AWAR','AIYL','AIYR','AIBR','AIZL','AIZR','RIAL','RIAR','SMDVL','SMDVR')


def design(room=.47):
    channels=np.ones((len(IDS),17),bool);channels[:,16]=False
    spec=SimpleNamespace(channels=channels,chemical=np.ones((len(IDS),len(IDS)),bool))
    size=int(channels.sum()+spec.chemical.sum());upper=np.full(size,np.log(2.));lower=np.full(size,-room)
    return spec,np.zeros(size),lower,upper


def test_scan_groups_cover_declared_edges_once_with_widened_upper_and_unwidened_lower():
    spec,origin,lower,upper=design()
    groups=scan_groups(spec,IDS,origin,lower,upper)
    assert [g['name'] for g in groups]==[f'chemical:{a}->{b}' for a,b in EDGES]
    used=[i for g in groups for i in g['indices']];assert len(used)==len(set(used))==len(EDGES)
    assert all(g['upper']==pytest.approx(np.log(MAX_FACTOR)) and g['lower']==pytest.approx(-.47) for g in groups)
    origin[used[0]]=np.log(2.)+.1
    with pytest.raises(ValueError):scan_groups(spec,IDS,origin,lower,upper)


def test_planned_cases_skip_factors_below_original_lower_bound_and_decode_exactly():
    spec,origin,lower,upper=design()
    groups=scan_groups(spec,IDS,origin,lower,upper);cases=planned_cases(groups)
    assert len(cases)==len(EDGES)*len(FACTORS)
    assert {c['factor'] for c in cases if not c['runnable']}=={.5}
    allowed=extend_upper(upper,origin,groups)
    for c in cases:
        if not c['runnable']:continue
        x=decode_groups(c['z'],groups,origin,lower,allowed)
        changed=np.flatnonzero(x!=origin);assert changed.tolist()==groups[c['group']]['indices']
        assert np.exp(x[changed[0]]-origin[changed[0]])==pytest.approx(c['factor'])
    full=design(room=1.)
    assert all(c['runnable'] for c in planned_cases(scan_groups(full[0],IDS,full[1],full[2],full[3])))


def test_only_scanned_coordinates_gain_upper_room():
    spec,origin,lower,upper=design();groups=scan_groups(spec,IDS,origin,lower,upper)
    allowed=extend_upper(upper,origin,groups)
    assert set(np.flatnonzero(allowed!=upper))=={g['indices'][0] for g in groups}
    assert np.all(allowed>=upper)


def row(name,effect,safe=True,status='completed'):
    pair=data(effect=effect);m=score(pair,data()['no_cue'],.01,1.);m['safe']=safe
    return dict(output=name,status=status,metrics=m)


def test_finalists_rank_by_weakest_region_among_gait_preserving_cases():
    rows=[row('a',.02),row('b',.05),row('c',.09,safe=False),row('d',.03,status='failed'),row('e',.04)]
    rows[3].pop('metrics')
    assert [r['output'] for r in finalists(rows)]==['b','e']
    assert reduction_percent(rows[1]['metrics'])>reduction_percent(rows[0]['metrics'])


def test_edge_scan_budget_dispatches_runner_and_its_own_audit(tmp_path,monkeypatch):
    import scripts.total_job_budget as runner
    out=tmp_path/'out';calls=[]
    def run(cmd,seconds):
        calls.append(cmd)
        if 'scripts.audit_edge_scan' in cmd:
            audit=tmp_path/'out-audit';audit.mkdir();(audit/'summary.json').write_text(json.dumps(dict(audit_complete=True,integrity_passed=True)))
        else:(out/'summary.json').write_text(json.dumps(dict(execution_valid=True,all_stages_finished=True,validation_complete=True,sensory_motion_component_qualified=False)))
        return 0,False
    monkeypatch.setattr(runner,'run_group',run)
    monkeypatch.setattr(sys,'argv',['budget','--pipeline','edge-scan','--total-minutes','180','--output-dir',str(out),'--source-run',str(tmp_path/'source')])
    assert runner.main()==0 and 'scripts.run_edge_scan' in calls[0] and '--objective' not in calls[0] and '--profile' not in calls[0]
    assert 'scripts.audit_edge_scan' in calls[1]
    assert json.loads((out/'deadline.json').read_text())['experiment']=='E095'
    assert json.loads((out/'budget_receipt.json').read_text())['post_audit_component_qualified'] is False


def test_coordinator_scans_all_runnable_cases_then_controls_two_finalists(tmp_path,monkeypatch):
    import scripts.run_edge_scan as runner
    import scripts.benchmark_cpu as topology
    import scripts.compare_rhythm_worker as verifier
    spec,origin,lower,upper=design()
    out=tmp_path/'run';out.mkdir();(out/'deadline.json').write_text(json.dumps(dict(coordinator_finish_by=time.monotonic()+20000)))
    inputs=out/'inputs';inputs.mkdir();np.savez(inputs/'cap_prior.npz',origin=origin,lower=lower,upper=upper,edge=0)
    (inputs/'manifest.json').write_text(json.dumps(dict(sha256={},sources={})))
    d=dict(spec=spec,model=SimpleNamespace(neuron_ids=IDS));parent_sha='test-parent'
    monkeypatch.setattr(runner,'prepare',lambda source,o,experiment,bilateral:(inputs,d,None,0,parent_sha,'audit-path'))
    monkeypatch.setattr(runner,'capture_run_record',lambda *args:None)
    monkeypatch.setattr(verifier,'verify_inputs',lambda root:None)
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda pid:set(range(32)),raising=False)
    monkeypatch.setattr(topology,'physical_core_cpus',lambda allowed:(list(range(32)),True))
    requests=[]
    def fake(j):
        requests.append(j);folder=Path(j['output']);folder.mkdir(parents=True)
        np.savez(folder/'coordinates.npz',z=j['z'])
        z=np.asarray(j['z']);effect=0. if not z.any() else .02+.01*int(np.flatnonzero(z)[0])
        pair=data(effect=effect);sha=parent_sha if not z.any() else hashlib.sha256(z.tobytes()).hexdigest()
        return dict(output=j['output'],status='completed',checkpoint_sha256=sha,metrics=score(pair,j['parent_features'] or pair['no_cue'],j['target'],j['margin']),
            paired_features=pair,cap='scan',kind=j['kind'],grid=j['grid'],mode=j['mode'])
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,func,j):
            f=concurrent.futures.Future();f.set_result(fake(j));return f
    monkeypatch.setattr(runner.concurrent.futures,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(sys,'argv',['scan','--output-dir',str(out),'--source-run',str(tmp_path/'source')])
    assert runner.main()==0
    s=json.loads((out/'summary.json').read_text())
    assert len(s['scan'])==len(EDGES)*(len(FACTORS)-1) and len(s['skipped'])==len(EDGES) and len(s['finalists'])==2
    assert len(s['controls'])==8 and s['all_stages_finished'] and s['evaluation_count']==1+40+8
    assert {j['mode'] for j in requests if j['kind']=='final'}=={'normal','sensory-blocked'}
    assert len(s['finalist_checks'])==2 and all(c['complete'] and c['same_checkpoint'] for c in s['finalist_checks'])
    assert not s['sensory_motion_component_qualified'] and not s['full_start_stop_qualified']
    assert json.loads((inputs/'cap_groups.json').read_text())['scan'][0]['name']=='chemical:AWAR->AIYR'
    assert all(Path(c['output']).name.startswith('final-') for c in s['controls'])


def test_saved_data_audit_accepts_single_widened_edge_and_rejects_second_changed_edge(tmp_path):
    from scripts.audit_edge_scan import audit
    from scripts.experiment_records import file_sha256
    spec,origin,lower,upper=design(room=1.);n=len(IDS);nc=int(np.count_nonzero(spec.channels))
    from r_elegans.relay_cap_experiments import edge_coordinate,upper_for
    inherited=edge_coordinate(SimpleNamespace(channels=spec.channels,chemical=spec.chemical),IDS);origin[inherited]=upper[inherited]+.2
    run=tmp_path/'run';inputs=run/'inputs';inputs.mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value))
    base=dict(neuron_ids=IDS,cells_conductances_nS=np.ones((n,17)),fixed_chemical_mask=spec.chemical,params_log_channel_scale=np.zeros((n,17)),
        params_raw_chemical=np.zeros((n,n)),fixed_release_threshold_mV=np.full(n,-40.),fixed_release_slope_mV=np.full(n,5.))
    np.savez(inputs/'motion_parent.npz',**base);np.savez(inputs/'motion_parent_coordinates.npz',offsets=origin)
    np.savez(inputs/'source_bounds.npz',lower=lower,upper=upper);np.savez(inputs/'cap_prior.npz',origin=origin,lower=lower,upper=upper,edge=inherited)
    groups=scan_groups(spec,IDS,origin,lower,upper);write(inputs/'cap_groups.json',dict(scan=groups));cases=planned_cases(groups)
    reference=tmp_path/'external-trace';reference.write_bytes(b'reference evidence')
    write(inputs/'sensory_reference.json',dict(path=str(reference),sha256=file_sha256(reference)))
    write(inputs/'manifest.json',dict(sha256={p.name:file_sha256(p) for p in inputs.iterdir()}))
    pair=data(effect=.02);parent_features=pair['no_cue'];write(run/'parent_features.json',parent_features)
    write(run/'configuration.json',dict(parent_sha256=file_sha256(inputs/'motion_parent.npz'),planned=cases))
    def raw_for(updates):
        raw=np.zeros((n,n))
        for (post,pre),factor in updates.items():raw[post,pre]=np.log(np.expm1(np.log(2.)*factor))
        return raw
    def evaluation(label,z,updates,reference_job=False):
        folder=run/'evaluations'/label;folder.mkdir(parents=True)
        x=decode_groups(z,groups,origin,lower,extend_upper(upper_for('aiyr_2p5',upper,inherited),origin,groups));model=dict(base);model['params_raw_chemical']=raw_for(updates)
        np.savez(folder/'network.npz',**model);np.savez(folder/'coordinates.npz',offsets=x,z=z)
        parent=None if reference_job else parent_features
        write(folder/'request.json',dict(output=str(folder),cap='scan',kind='scan',z=list(z),retain_traces=False,variant='nominal',mode='normal',grid='ordinary',
            reference=reference_job,target=.01,margin=1.,parent_features=parent))
        r=dict(status='completed',output=str(folder),checkpoint_sha256=file_sha256(folder/'network.npz'),coordinates_sha256=file_sha256(folder/'coordinates.npz'),
            paired_features=pair,metrics=score(pair,parent or pair['no_cue'],.01,1.),trajectory_sha256={})
        write(folder/'summary.json',r);return r
    def summarize(baseline,scanned):
        write(run/'summary.json',dict(experiment='E095',evaluation_count=2,baseline=baseline,scan=[scanned],controls=[],finalists=[scanned['output']],validation_complete=False))
    case=next(c for c in cases if c['runnable'] and c['factor']==1.5 and c['group']==0)
    edge0=divmod(groups[0]['indices'][0]-nc,n);edge1=divmod(groups[1]['indices'][0]-nc,n)
    baseline=evaluation('parent',[0.]*len(groups),{},True);scanned=evaluation('scan-00-f1.5',case['z'],{edge0:1.5})
    summarize(baseline,scanned)
    assert audit(run,tmp_path/'audit')==0
    # A second changed edge is not a single-edge dose-response and must be flagged.
    folder=Path(scanned['output']);model=dict(base);model['params_raw_chemical']=raw_for({edge0:1.5,edge1:1.2})
    np.savez(folder/'network.npz',**model);scanned['checkpoint_sha256']=file_sha256(folder/'network.npz')
    write(folder/'summary.json',scanned);summarize(baseline,scanned)
    assert audit(run,tmp_path/'bad-audit')==2
    assert any('Frozen model changed' in p or 'Physical coordinate mismatch' in p for p in json.loads((tmp_path/'bad-audit/summary.json').read_text())['problems'])
