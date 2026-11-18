import json,sys
from types import SimpleNamespace
import numpy as np
import pytest
from r_elegans.relay_cap_experiments import edge_coordinate,upper_for,panel_assessment
from r_elegans.relay_range import parameter_groups
from r_elegans.cma_curriculum import decode_groups


def design():
    ids=('AWAL','AIYL','AIYR','RIAL')
    channels=np.ones((4,17),bool);channels[:,16]=False
    spec=SimpleNamespace(channels=channels,chemical=np.ones((4,4),bool))
    size=int(channels.sum()+spec.chemical.sum());upper=np.full(size,np.log(2.));lower=-upper
    edge=edge_coordinate(spec,ids);origin=np.zeros(size);origin[edge]=upper[edge]-.0001
    return ids,spec,origin,lower,upper,edge


def test_only_declared_edge_cap_expands_and_other_coordinates_are_frozen():
    ids,spec,origin,lower,upper,edge=design()
    expanded=upper_for('aiyr_2p5',upper,edge)
    assert np.flatnonzero(expanded!=upper).tolist()==[edge]
    assert np.exp(expanded[edge])==pytest.approx(2.5)
    groups=parameter_groups(spec,ids,origin,lower,expanded)
    j=next(i for i,g in enumerate(groups) if g['name']=='chemical:AIYR->AIYL')
    z=np.zeros(len(groups));z[j]=1
    x=decode_groups(z,groups,origin,lower,expanded)
    assert np.flatnonzero(x!=origin).tolist()==[edge]
    with pytest.raises(ValueError):decode_groups(z,groups,origin,lower,upper)
    with pytest.raises(ValueError):upper_for('unrecorded',upper,edge)


def test_matched_seed_streams_independent_and_snapshot_preserves_next_population(tmp_path):
    cma=pytest.importorskip('cma')
    from scripts.run_relay_experiments import new_optimizer
    from scripts.run_cma_sensorimotor import optimizer_snapshot
    import pickle
    a=new_optimizer(cma,5,9200);b=new_optimizer(cma,5,9200)
    # Initializing/sampling a third arm must not reset either paired stream.
    new_optimizer(cma,5,9201).ask()
    xa,xb=a.ask(),b.ask();np.testing.assert_array_equal(xa,xb)
    a.tell(xa,list(range(12)));optimizer_snapshot(tmp_path/'snapshot',a,0,1)
    with (tmp_path/'snapshot/optimizer.pkl').open('rb') as f:restored=pickle.load(f)['es']
    np.testing.assert_array_equal(a.ask(),restored.ask())


def test_panel_cannot_qualify_incomplete_different_checkpoints_or_failed_low_dose():
    metrics=dict(safe=True,baseline_and_recovery_preserved=True,useful_response_preserved=True,
        partial_attenuation_passed=True,response_amplitude_ratios=[.99997],relay_response_mean_changes=[-.00003])
    rows=[dict(status='completed',checkpoint_sha256='same',grid='ordinary',metrics=dict(metrics)) for i in range(10)]
    rows[8]['metrics']=dict(metrics,response_amplitude_ratios=[1.])
    assert panel_assessment(rows)['engineering_criteria_met']
    rows[1]['metrics']=dict(metrics,partial_attenuation_passed=False)
    assert not panel_assessment(rows)['engineering_criteria_met']
    assert not panel_assessment(rows[:-1])['complete']


def test_total_budget_includes_custom_audit_without_promotion(tmp_path,monkeypatch):
    import scripts.total_job_budget as runner
    out=tmp_path/'out';calls=[]
    def run(cmd,seconds):
        calls.append((cmd,seconds))
        if 'scripts.audit_relay_experiments' in cmd:
            audit=tmp_path/'out-audit';audit.mkdir();(audit/'summary.json').write_text(json.dumps(dict(audit_complete=True,integrity_passed=True)))
        else:(out/'summary.json').write_text(json.dumps(dict(execution_valid=True,all_stages_finished=True,validation_complete=True)))
        return 0,False
    monkeypatch.setattr(runner,'run_group',run)
    monkeypatch.setattr(sys,'argv',['budget','--pipeline','relay-experiments','--total-minutes','480','--output-dir',str(out),'--source-run',str(tmp_path/'parent')])
    assert runner.main()==0
    assert 'scripts.run_relay_experiments' in calls[0][0] and '--objective' not in calls[0][0]
    assert 'scripts.audit_relay_experiments' in calls[1][0]
    assert calls[0][1]<=460*60 and calls[1][1]<=480*60
    receipt=json.loads((out/'budget_receipt.json').read_text())
    assert receipt['all_steps_finished'] and receipt['audit_integrity_passed']
    assert not receipt['post_audit_component_qualified']


def test_expanded_worker_preserves_model_on_deadline_and_auditor_detects_tampering(tmp_path,monkeypatch):
    from scripts import relay_cap_worker as capworker
    from scripts import rial_sensory_worker as worker
    from scripts.audit_relay_experiments import coordinate_check
    from r_elegans.brain.conductance_network import load_network,save_network
    from r_elegans.brain.joint_autonomous_fit import joint_fit_reference
    from pathlib import Path
    params,fixed,config=load_network(Path('r_elegans/assets/autonomous_wiggle_v1/network.npz'))
    from scripts.assess_sensorimotor_resolution import promote_floating_values
    params,fixed=promote_floating_values((params,fixed))
    _,spec=joint_fit_reference(fixed)
    ids=tuple(np.load('r_elegans/assets/autonomous_wiggle_v1/network.npz',allow_pickle=False)['neuron_ids'])
    edge=edge_coordinate(spec,ids);origin=np.zeros(spec.size);origin[edge]=np.log(2)-.0001
    lower=np.full(spec.size,-np.log(2));upper=-lower
    schemas={cap:parameter_groups(spec,ids,origin,lower,upper_for(cap,upper,edge)) for cap in ('original','aiyr_2p5')}
    capworker._BASE=dict(base=params,fixed=fixed,config=config,spec=spec,origin=origin,gains=np.ones(3),model=SimpleNamespace(neuron_ids=ids))
    capworker._PRIOR=dict(lower=lower,upper=upper,origin=origin,edge=edge);capworker._SCHEMAS=schemas
    monkeypatch.setattr(worker.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    z=np.zeros(len(schemas['aiyr_2p5']));i=next(i for i,g in enumerate(schemas['aiyr_2p5']) if g['indices']==[edge]);z[i]=1
    folder=tmp_path/'expired'
    result=capworker.evaluate(dict(z=z.tolist(),output=str(folder),cap='aiyr_2p5',kind='single_edge',objective='relay-range',until=-1,variant='nominal',mode='normal'))
    assert result['status']=='time_budget'
    assert result['checkpoint_sha256'] and result['coordinates_sha256']
    parent=tmp_path/'parent.npz';save_network(parent,params,fixed,config)
    with np.load(folder/'coordinates.npz',allow_pickle=False) as data:x=data['offsets']
    assert x[edge]>upper[edge]
    assert coordinate_check(parent,folder/'network.npz',origin,x)
    bad=x.copy();bad[0]+=.1
    assert not coordinate_check(parent,folder/'network.npz',origin,bad)


def test_coordinator_completes_matched_cohort_and_deduplicated_final_panel(tmp_path,monkeypatch):
    import concurrent.futures,hashlib,time
    from scripts import run_relay_experiments as runner
    import scripts.benchmark_cpu as topology
    ids,spec,origin,lower,upper,edge=design()
    schemas={cap:parameter_groups(spec,ids,origin,lower,upper_for(cap,upper,edge)) for cap in ('original','aiyr_2p5')}
    out=tmp_path/'run';out.mkdir();(out/'deadline.json').write_text(json.dumps(dict(coordinator_finish_by=time.monotonic()+20000)))
    monkeypatch.setattr(runner,'prepare',lambda source,out:(tmp_path/'inputs',dict(origin=origin,lower=lower,upper=upper),schemas,edge))
    monkeypatch.setattr(runner,'capture_run_record',lambda *args:None)
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda pid:set(range(32)),raising=False)
    monkeypatch.setattr(topology,'physical_core_cpus',lambda allowed:(list(range(32)),True))
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    requests=[]
    def fake_evaluate(j):
        requests.append(j);folder=__import__('pathlib').Path(j['output']);folder.mkdir(parents=True)
        np.savez(folder/'coordinates.npz',z=j['z'])
        x=decode_groups(j['z'],schemas[j['cap']],origin,lower,upper_for(j['cap'],upper,edge))
        sha=runner.PARENT if np.array_equal(x,origin) else hashlib.sha256(x.tobytes()).hexdigest()
        metrics=dict(safe=True,baseline_and_recovery_preserved=True,useful_response_preserved=True,
            partial_attenuation_passed=True,response_amplitude_ratios=[1. if j['mode']=='sensory-blocked' else .99997],relay_response_mean_changes=[-.00003],loss=1.+float(np.linalg.norm(j['z'])))
        return dict(output=j['output'],status='completed',checkpoint_sha256=sha,cap=j['cap'],kind=j['kind'],
            grid=j['grid'],metrics=metrics,paired_features=dict(no_cue={}))
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,func,j):
            f=concurrent.futures.Future();f.set_result(fake_evaluate(j));return f
    monkeypatch.setattr(runner.concurrent.futures,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(sys,'argv',['suite','--output-dir',str(out),'--source-run',str(tmp_path/'source'),'--generation-limit','1'])
    assert runner.main()==0
    s=json.loads((out/'summary.json').read_text())
    assert s['completed_matched_generations']==1 and s['all_stages_finished']
    assert len([j for j in requests if j['kind']=='fit'])==48
    assert len(s['panels'])==1 and len(s['panels'][0]['validation'])==10
    for name in s['arms']:
        assert (out/'optimizer'/name/'g-00000-told/metadata.json').exists()
        assert (out/'optimizer'/name/'final/optimizer.pkl').exists()


@pytest.mark.parametrize('retain',[False,True])
def test_saved_data_audit_runs_and_rejects_changed_frozen_array(tmp_path,retain):
    from pathlib import Path
    from scripts.audit_relay_experiments import audit
    from scripts.experiment_records import file_sha256
    from r_elegans.relay_range import score
    from test_relay_range import relay_pair
    ids,spec,origin,lower,upper,edge=design()
    run=tmp_path/'run';inputs=run/'inputs';inputs.mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value))
    data=dict(neuron_ids=ids,cells_conductances_nS=np.ones((4,17)),fixed_chemical_mask=spec.chemical,
        params_log_channel_scale=np.zeros((4,17)),params_raw_chemical=np.zeros((4,4)),
        fixed_release_threshold_mV=np.full(4,-40.),fixed_release_slope_mV=np.full(4,5.))
    np.savez(inputs/'sensory_parent.npz',**data)
    np.savez(inputs/'sensory_parent_coordinates.npz',offsets=origin)
    np.savez(inputs/'source_bounds.npz',lower=lower,upper=upper)
    np.savez(inputs/'cap_prior.npz',origin=origin,lower=lower,upper=upper,edge=edge)
    schemas={cap:parameter_groups(spec,ids,origin,lower,upper_for(cap,upper,edge)) for cap in ('original','aiyr_2p5')}
    write(inputs/'cap_groups.json',schemas)
    reference=tmp_path/'external-trace';reference.write_bytes(b'reference evidence')
    write(inputs/'sensory_reference.json',dict(path=str(reference),sha256=file_sha256(reference)))
    write(inputs/'manifest.json',dict(sha256={p.name:file_sha256(p) for p in inputs.iterdir()}))
    write(run/'configuration.json',dict(parent_sha256=file_sha256(inputs/'sensory_parent.npz'),only_extended_coordinate=edge))
    paired=relay_pair();traces={}
    if retain:
        from r_elegans.relay_range import features
        from scipy.special import logit
        t=np.arange(0.,70.001,.02)
        for condition in ('no_cue','cue'):
            release=np.full(len(t),.94)
            if condition=='cue':release[(t>=41)&(t<46)]+=.0001
            voltage=np.full((len(t),len(ids)),-40.)
            voltage[:,ids.index('AIYL')]=-40+5*logit(release)
            traces[condition]=dict(dense_time_s=t,dense_voltage_mV=voltage,
                dense_joints=.01*np.sin(2*np.pi*1.8*t[:,None]+np.arange(11)[None,:]/10))
            paired[condition]=features(traces[condition],ids,'nominal',-40.,5.)
    write(run/'parent_features.json',paired['no_cue'])
    folder=run/'evaluations/case';folder.mkdir(parents=True)
    np.savez(folder/'network.npz',**data);np.savez(folder/'coordinates.npz',offsets=origin,z=np.zeros(len(schemas['original'])))
    request=dict(output=str(folder),cap='original',kind='fit',z=[0.]*len(schemas['original']),retain_traces=retain,
        variant='nominal',mode='normal',parent_features=paired['no_cue'],target=.02,margin=1.3,response_aggregation='weakest')
    write(folder/'request.json',request)
    result=dict(status='completed',output=str(folder),checkpoint_sha256=file_sha256(folder/'network.npz'),
        coordinates_sha256=file_sha256(folder/'coordinates.npz'),paired_features=paired,
        metrics=score(paired,paired['no_cue'],.02,1.3,response_aggregation='weakest'),trajectory_sha256={})
    for condition,trace in traces.items():
        np.savez_compressed(folder/(condition+'.npz'),**trace)
        result['trajectory_sha256'][condition]=file_sha256(folder/(condition+'.npz'))
    write(folder/'summary.json',result)
    write(run/'summary.json',dict(evaluation_count=1,panels=[],scan=[result],scan_controls=[],arms={},completed_matched_generations=0))
    assert audit(run,tmp_path/'audit')==0
    if retain:
        signals=json.loads((tmp_path/'audit/case-signals.json').read_text())['windows']
        assert len(signals)==4 and 'AIYL' in signals[0]['release_by_neuron']
    data['fixed_release_slope_mV']=np.full(4,6.);np.savez(folder/'network.npz',**data)
    # Even updating the recorded model hash cannot hide a frozen-field mutation.
    result['checkpoint_sha256']=file_sha256(folder/'network.npz');write(folder/'summary.json',result)
    assert audit(run,tmp_path/'bad-audit')==2
    problems=json.loads((tmp_path/'bad-audit/summary.json').read_text())['problems']
    assert any('Frozen model changed' in p for p in problems)


def test_report_keeps_reaudits_separate_and_does_not_hide_pending_audit(tmp_path):
    from scripts.report_relay_experiments import latest_run,audit_path
    run=tmp_path/'relay-experiments-20261118T035453Z-398433';run.mkdir()
    (run/'configuration.json').write_text('{}')
    original=tmp_path/(run.name+'-audit');original.mkdir()
    (original/'summary.json').write_text('{"integrity_passed":false}')
    corrected=tmp_path/(run.name+'-audit-20261118T120000Z-1');corrected.mkdir()
    assert latest_run(tmp_path)==run
    assert audit_path(run)==corrected/'summary.json'
    assert not audit_path(run).exists()  # Pending latest audit stays pending.
    (corrected/'summary.json').write_text('{"integrity_passed":true}')
    assert audit_path(run)==corrected/'summary.json'
