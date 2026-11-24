import json
from pathlib import Path
import sys
import numpy as np
import pytest
from scripts import continue_midpoint_validation as resume


def task(begin=0):
    return dict(arm='network',level=1,begin=begin,targets=['RIVL'] if begin==0 else ['RMDR'],model_sha256='model',starts_sha256='start')


def batch(t):
    means=np.zeros((16,6,302));means[:,0,0]=1e-4
    a=dict(means=means,voltage_min=np.full((16,6),-40.),voltage_max=np.full((16,6),-20.),
        sham_voltage=np.zeros((16,2,12000,2)) if t['begin']==0 else np.empty(0),final_gates=np.zeros((16,6,302,35)))
    for key in ('voltage_mV','calcium_uM','indicator','peptide'):a['final_'+key]=np.zeros((16,6,302))
    return a


def test_atomic_batch_checkpoint_survives_without_sidecar_and_rejects_wrong_provenance(tmp_path):
    t=task();p=resume.batch_path(tmp_path,t)
    resume.atomic_npz(p,provenance_json=np.array(json.dumps(t)),**batch(t))
    actual=resume.read_batch(tmp_path,t)
    np.testing.assert_array_equal(actual['means'],batch(t)['means'])
    changed=t|dict(model_sha256='wrong')
    with pytest.raises(ValueError,match='provenance'):resume.read_batch(tmp_path,changed)
    resume.atomic_json(p.with_suffix('.json'),dict(sha256='bad'))
    with pytest.raises(ValueError,match='checksum'):resume.read_batch(tmp_path,t)


def test_finalize_reuses_batches_and_never_calls_gpu(tmp_path,monkeypatch):
    tasks=[task(),task(4)]
    resume.atomic_json(tmp_path/'manifest.json',dict(tasks=tasks,models={}))
    for t in tasks:resume.atomic_npz(resume.batch_path(tmp_path,t),provenance_json=np.array(json.dumps(t)),**batch(t))
    finalized=[]
    monkeypatch.setattr(resume,'finish_level',lambda p:finalized.append(p))
    resume.finalize(tmp_path)
    result=resume.old.prior.arrays(tmp_path/'network/full-1/response.npz')
    assert result['response'].shape==(16,2,302)
    assert result['controls'].shape==(2,16,2,302)
    assert finalized==[tmp_path/'network/full-1']
    assert all(resume.batch_path(tmp_path,t).exists() for t in tasks)


def test_readonly_commands_do_not_launch_or_create_files(tmp_path,monkeypatch):
    monkeypatch.setattr(resume,'BASE',tmp_path);monkeypatch.setattr(resume,'active_jobs',lambda:[])
    monkeypatch.setattr(resume.subprocess,'Popen',lambda *a,**k:pytest.fail('Read-only launched'))
    for args in (['--status'],['--output-dir',str(tmp_path),'--report'],['--select-output']):
        monkeypatch.setattr(sys,'argv',['resume']+args);resume.main()
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('metadata',['manifest.json','job.json'])
def test_select_output_returns_existing_continuation_for_source(tmp_path,monkeypatch,metadata):
    monkeypatch.setattr(resume,'BASE',tmp_path)
    source=tmp_path/'source';root=tmp_path/'midpoint-continuation-test'
    resume.atomic_json(root/metadata,dict(continuation_source=str(source.resolve())))
    assert resume.select_output(source)==root


def test_active_job_refuses_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(resume,'active_jobs',lambda:[dict(pid=123)])
    monkeypatch.setattr(sys,'argv',['resume','--output-dir',str(tmp_path/'new')])
    with pytest.raises(SystemExit):resume.main()
    assert not (tmp_path/'new').exists()


def test_completed_run_does_not_start_again(tmp_path,monkeypatch):
    source=tmp_path/'source';root=tmp_path/'run'
    resume.atomic_json(root/'manifest.json',dict(continuation_source=str(source.resolve())))
    resume.atomic_json(root/'budget_receipt.json',dict(all_steps_finished=True))
    monkeypatch.setattr(resume,'BASE',tmp_path);monkeypatch.setattr(resume,'active_jobs',lambda:[])
    monkeypatch.setattr(resume.subprocess,'Popen',lambda *a,**k:pytest.fail('Completed run restarted'))
    monkeypatch.setattr(sys,'argv',['resume','--source-run',str(source),'--output-dir',str(root)])
    assert resume.main()==0


def test_supervisor_does_not_impose_validation_timeout(tmp_path,monkeypatch):
    from scripts import total_job_budget
    source=tmp_path/'source';root=tmp_path/'run'
    resume.atomic_json(root/'manifest.json',dict(continuation_source=str(source.resolve()),source_sha256={}))
    monkeypatch.setattr(resume,'BASE',tmp_path);monkeypatch.setattr(resume,'active_jobs',lambda:[])
    monkeypatch.setattr(resume,'capture_run_record',lambda *a,**k:None)
    calls=[]
    def complete(command,seconds):calls.append(seconds);return 0,False
    monkeypatch.setattr(total_job_budget,'run_group',complete)
    monkeypatch.setattr(resume,'audit',lambda p:dict(audit_complete=True,execution_complete=True,integrity_passed=True,parent_validation_passed=False))
    monkeypatch.setattr(sys,'argv',['resume','--source-run',str(source),'--output-dir',str(root)])
    assert resume.main()==0 and calls==[None]
    receipt=json.loads((root/'budget_receipt.json').read_text())
    assert receipt['all_steps_finished'] and receipt['validation_allowed_to_finish'] and not receipt['parent_validation_passed']


def test_prepare_reuses_three_complete_levels_and_only_schedules_missing_network_fine(tmp_path):
    from r_elegans.brain.midpoint_replay import cycle_phases
    source=tmp_path/'source';source.mkdir();root=tmp_path/'continued';root.mkdir()
    ids=['RMDL']+[f'N{i}' for i in range(301)]
    targets=[f'T{i}' for i in range(135)]
    models={}
    for arm in ('network','peptide'):
        model=source/arm/'parent-model.npz';model.parent.mkdir()
        np.savez_compressed(model,neuron_ids=np.array(ids))
        models[arm]=dict(parent_sha256=resume.file_sha256(model))
        resume.atomic_json(source/arm/'summary.json',dict(selected_level=0))
        for level in (0,1):
            folder=source/arm/f'full-{level}';folder.mkdir()
            dt=resume.old.clock(level)[0];t=(np.arange(round(4/dt))+1)*dt
            ref=-30+10*np.cos(2*np.pi*t/.23);ix,meta=cycle_phases(ref,dt);meta['reference_voltage_mV']=ref.tolist()
            state=np.zeros((16,302));state[:,0]=ref[ix]
            np.savez_compressed(folder/'starts.npz',voltage_mV=state,gates=np.zeros((16,302,35)),calcium_uM=np.zeros((16,302)),indicator=np.zeros((16,302)),peptide=np.zeros((16,302)))
            resume.atomic_json(folder/'phases.json',meta)
            if arm=='network' and level==1:continue
            time=np.arange(12000)*.001
            sham=np.broadcast_to((-30+10*np.sin(2*np.pi*time*2))[None,None,:,None],(16,2,12000,2)).copy()
            values=dict(response=np.zeros((16,135,302)),controls=np.zeros((1,16,2,302)),sham_voltage=sham,voltage_min=np.full((1,16,6),-40.),voltage_max=np.full((1,16,6),-20.))
            np.savez_compressed(folder/'response.npz',**values);np.savez_compressed(folder/'final.npz',voltage_mV=np.zeros((16,6,302)))
            resume.finish_level(folder)
    resume.atomic_json(source/'manifest.json',dict(models=models,full_targets=targets,source_sha256={},input_sha256={}))
    resume.atomic_json(source/'budget_receipt.json',dict(all_steps_finished=False))
    resume.atomic_json(source/'audit.json',dict(audit_complete=True,integrity_passed=True,artifact_sha256={str(p):resume.file_sha256(p) for p in source.rglob('*.npz')}))
    before={str(p):resume.file_sha256(p) for p in source.rglob('*') if p.is_file()}
    resume.prepare(root,source)
    m=json.loads((root/'manifest.json').read_text())
    assert len(m['reused_levels'])==3
    assert len(m['tasks'])==34
    assert all(t['arm']=='network' and t['level']==1 for t in m['tasks'])
    assert m['tasks'][-1]['begin']==132 and len(m['tasks'][-1]['targets'])==3
    assert len(m['tasks'][::2])==len(m['tasks'][1::2])==17
    assert all(resume.file_sha256(p)==h for p,h in before.items())
    assert resume.file_sha256(root/'network/full-1/starts.npz')==resume.file_sha256(source/'network/full-1/starts.npz')


def test_private_validation_group_accepts_no_time_limit():
    from scripts.total_job_budget import run_group
    assert run_group([sys.executable,'-c','pass'],None)==(0,False)


def test_failed_write_keeps_previous_committed_checkpoint(tmp_path,monkeypatch):
    t=task();path=resume.batch_path(tmp_path,t)
    resume.atomic_npz(path,provenance_json=np.array(json.dumps(t)),**batch(t))
    before=resume.file_sha256(path)
    def interrupted(file,**values):file.write(b'incomplete');raise OSError('simulated disk interruption')
    monkeypatch.setattr(resume.np,'savez_compressed',interrupted)
    with pytest.raises(OSError):resume.atomic_npz(path,provenance_json=np.array(json.dumps(t)),**batch(t))
    assert resume.file_sha256(path)==before
    assert resume.read_batch(tmp_path,t)['means'].shape==(16,6,302)


def test_all_saved_batches_skip_gpu_workers(tmp_path,monkeypatch):
    t=task();resume.atomic_json(tmp_path/'manifest.json',dict(tasks=[t]))
    resume.atomic_npz(resume.batch_path(tmp_path,t),provenance_json=np.array(json.dumps(t)),**batch(t))
    monkeypatch.setattr(resume.subprocess,'Popen',lambda *a,**k:pytest.fail('Replayed saved batch'))
    finalized=[];monkeypatch.setattr(resume,'finalize',lambda p:finalized.append(p))
    assert resume.coordinate(tmp_path)==0 and finalized==[tmp_path]
