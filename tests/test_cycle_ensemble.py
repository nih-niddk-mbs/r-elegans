import json
import sys
import numpy as np
import pytest
from r_elegans.brain.cycle_ensemble import spread_order,extension_phases,ensemble_metrics
from scripts import run_cycle_ensemble as run


def test_nested_cycle_bins_cover_long_window_and_keep_groups():
    order=spread_order(32)
    assert order[:2]==[0,31] and len(set(order))==32
    t=(np.arange(32000)+1)*.001
    ix,m=extension_phases(-30+10*np.sin(2*np.pi*t/.23),.001)
    assert len(ix)==112 and len(set(ix))==112
    assert max(m['cycle_mean_time_s'][:4])-min(m['cycle_mean_time_s'][:4])>30
    assert len(set(m['selected_cycles']))==28


def test_stability_requires_temporal_and_prefix_agreement_not_just_small_leaveout():
    y=np.ones((32*4,2,3))*.001;y[16*4:]*=1.2
    result=ensemble_metrics(y,list(range(32)),previous=y[:16*4],tolerance=.00005)
    assert result['leave_one_out_passed'] and not result['temporal_halves_passed'] and not result['passed']
    stable=np.ones_like(y)*.001
    assert ensemble_metrics(stable,list(range(32)),stable[:16*4],.00005)['passed']


def test_panel_derivation_keeps_self_and_downstream_failures():
    y=np.zeros((16,2,302));y[:4,0,0]=1;y[:4,1,3]=1
    assert run.affected_targets(y,['RMDL','RMDR'],.01)==['RMDL','RMDR']


def test_nested_tasks_add_only_missing_cycle_groups():
    m=dict(cases=[dict(arm='network',level=0,model_sha256='x')],panel=['T0','T1','T2','T3','T4'])
    small=run.tasks_for(m,8);middle=run.tasks_for(m,16);large=run.tasks_for(m,32)
    assert len(small)==2 and len(middle)==6 and len(large)==14
    assert middle[:2]==small and large[:6]==middle


def test_extension_checkpoint_materialization_requires_no_gpu(tmp_path):
    case=dict(model_sha256='model',starts_sha256='starts')
    values=dict(voltage_mV=np.zeros((112,302)),gates=np.zeros((112,302,35)),calcium_uM=np.zeros((112,302)),indicator=np.zeros((112,302)),peptide=np.zeros((112,302)))
    end={k:v[-1] for k,v in values.items()}
    meta=dict(model_sha256='model',base_starts_sha256='starts')
    run.resume.atomic_npz(tmp_path/'extension.npz',metadata_json=np.array(json.dumps(meta)),**values,**{'end_'+k:v for k,v in end.items()})
    run.materialize_extension(tmp_path,case)
    assert (tmp_path/'extension.json').exists() and (tmp_path/'extension-end.npz').exists()
    run.materialize_extension(tmp_path,case)


def test_status_report_and_selection_are_readonly(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'BASE',tmp_path);monkeypatch.setattr(run,'jobs',lambda:[])
    monkeypatch.setattr(run.subprocess,'Popen',lambda *a,**k:pytest.fail('Read-only launched'))
    for args in (['--status'],['--select-output'],['--output-dir',str(tmp_path),'--report']):
        monkeypatch.setattr(sys,'argv',['ensemble']+args);run.main()
    assert not list(tmp_path.iterdir())


def test_admitted_stage_finishes_after_allowance_but_next_stage_not_admitted(tmp_path,monkeypatch):
    run.resume.atomic_json(tmp_path/'manifest.json',dict(cases=[],panel=[]))
    run.resume.atomic_json(tmp_path/'job.json',dict(sampling_deadline_unix=0.))
    run.resume.atomic_json(tmp_path/'sampling-state.json',dict(admitted=[8],completed=[],stop_reason=None))
    calls=[];monkeypatch.setattr(run,'run_workers',lambda root,stage:calls.append(stage) or 0)
    monkeypatch.setattr(run,'evaluate',lambda root,m,n:dict(cycles=n,panel_qualified=False))
    monkeypatch.setattr(run,'estimate_stage',lambda *a:100.)
    assert run.coordinate(tmp_path)==0 and calls==[0,8]
    assert (tmp_path/'stage-8.json').exists() and not (tmp_path/'stage-16.json').exists()


def test_main_sampling_deadline_does_not_kill_validation(tmp_path,monkeypatch):
    from scripts import total_job_budget
    root=tmp_path/'job';source=tmp_path/'source'
    run.resume.atomic_json(root/'manifest.json',dict(source_sha256={},input_sha256={}))
    monkeypatch.setattr(run,'BASE',tmp_path);monkeypatch.setattr(run,'jobs',lambda:[])
    monkeypatch.setattr(run,'capture_run_record',lambda *a,**k:None)
    calls=[]
    def finish(command,seconds):calls.append(seconds);return 0,False
    monkeypatch.setattr(total_job_budget,'run_group',finish)
    monkeypatch.setattr(run,'audit',lambda root:dict(audit_complete=True,execution_complete=True,integrity_passed=True,panel_qualified=False))
    monkeypatch.setattr(sys,'argv',['ensemble','--source-run',str(source),'--output-dir',str(root)])
    assert run.main()==0 and calls==[None]
    receipt=json.loads((root/'budget_receipt.json').read_text())
    assert receipt['sampling_minutes']==180 and receipt['validation_allowed_to_finish'] and receipt['all_steps_finished']


def test_existing_work_refuses_new_job(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'jobs',lambda:[dict(pid=123)])
    monkeypatch.setattr(sys,'argv',['ensemble','--output-dir',str(tmp_path/'new')])
    with pytest.raises(SystemExit):run.main()
    assert not (tmp_path/'new').exists()


def test_prepare_reuses_original_cycles_and_derives_union_of_affected_targets(tmp_path):
    source=tmp_path/'source';source.mkdir();root=tmp_path/'ensemble';root.mkdir()
    targets=['RMDL','RMDR','N0'];models={};arms={}
    for arm in ('network','peptide'):
        model=source/arm/'parent-model.npz';model.parent.mkdir();np.savez_compressed(model,neuron_ids=np.array(['RMDL','RMDR']+[f'N{i}' for i in range(300)]))
        models[arm]=dict(parent_sha256=run.file_sha256(model),cpus=list(range(4)))
        arms[arm]=dict(selected_level=0,full_target_response_check=dict(passed=True,tolerance=.01),level_checks=[dict(passed=True),dict(passed=True)])
        for level in (0,1):
            folder=source/arm/f'full-{level}';folder.mkdir()
            y=np.zeros((16,3,302));y[:4,0 if arm=='network' else 1,0]=1
            np.savez_compressed(folder/'response.npz',response=y)
            np.savez_compressed(folder/'starts.npz',voltage_mV=np.zeros((16,302)))
            run.resume.atomic_json(folder/'phases.json',dict(indices=list(range(16))))
            run.resume.atomic_json(folder/'metrics.json',dict(sampling=dict(tolerance=.01),seconds=100.))
    run.resume.atomic_json(source/'manifest.json',dict(full_targets=targets,models=models,source_sha256={},policy=dict(cycles=4,phases_per_cycle=4,warmup_seconds=60)))
    run.resume.atomic_json(source/'audit.json',dict(audit_complete=True,execution_complete=True,integrity_passed=True,arms=arms,artifact_sha256={str(p):run.file_sha256(p) for p in source.rglob('*.npz')}))
    run.resume.atomic_json(source/'budget_receipt.json',dict(all_steps_finished=True))
    before={str(p):run.file_sha256(p) for p in source.rglob('*') if p.is_file()}
    run.prepare(root,source)
    m=json.loads((root/'manifest.json').read_text())
    assert m['panel']==['RMDL','RMDR'] and len(m['cases'])==4
    for c in m['cases']:
        saved=run.resume.old.prior.arrays(run.folder_for(root,c)/'base-response.npz')['response']
        original=run.resume.old.prior.arrays(source/c['arm']/f"full-{c['level']}"/'response.npz')['response']
        np.testing.assert_array_equal(saved,original[:,:2])
        assert c['tolerance']==.01
    assert all(run.file_sha256(p)==h for p,h in before.items())


def test_saved_batches_validate_nested_ensemble_and_both_clocks(tmp_path):
    cases=[dict(arm=arm,level=level,model_sha256='model',base_cycle_times=[60.,61.,62.,63.],extension_start_s=64.,tolerance=1e-5) for arm in ('network','peptide') for level in (0,1)]
    m=dict(cases=cases,panel=['RMDL'],numeric_tolerance=dict(network=1e-5,peptide=1e-5))
    time=np.arange(12000)*.001
    sham=np.broadcast_to((-30+10*np.sin(2*np.pi*time*4))[None,None,:,None],(16,2,12000,2)).copy()
    for case in cases:
        folder=run.folder_for(tmp_path,case);folder.mkdir(parents=True)
        base=np.zeros((16,1,302));base[:,:,0]=.001;run.resume.atomic_npz(folder/'base-response.npz',response=base)
        run.resume.atomic_json(folder/'extension.json',dict(cycle_mean_time_s=[.5,31.5,15.5,23.5]))
    for task in run.tasks_for(m,8):
        means=np.zeros((16,6,302));means[:,0,0]=.001
        data=dict(means=means,voltage_min=np.full((16,6),-40.),voltage_max=np.full((16,6),-20.),sham_voltage=sham,final_gates=np.zeros((16,6,302,35)))
        for key in ('voltage_mV','calcium_uM','indicator','peptide'):data['final_'+key]=np.zeros((16,6,302))
        run.resume.atomic_npz(run.task_path(tmp_path,task),provenance_json=np.array(json.dumps(task)),**data)
    result=run.evaluate(tmp_path,m,8)
    assert result['panel_qualified'] and all(c['sampling']['passed'] for c in result['cases'])
    assert all(n['maximum_difference']==0. for n in result['numerical'])
    task=run.tasks_for(m,8)[0];data=run.read_task(tmp_path,task);data['means'][:4,0,0]=.1
    run.resume.atomic_npz(run.task_path(tmp_path,task),provenance_json=np.array(json.dumps(task)),**data)
    assert not run.evaluate(tmp_path,m,8)['panel_qualified']
