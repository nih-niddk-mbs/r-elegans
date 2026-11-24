"""Duplicate prevention, immutable cache reuse and explicit command semantics."""
import fcntl
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from scripts import cv_validation_job as jobs
from scripts.continue_cv_peptide_validation import verified_copy,check_cached_grid,verify_environment
from scripts.experiment_records import file_sha256


def test_process_check_distinguishes_reports_workers_and_launchers():
    output='''100 01:00 python -m scripts.run_cv_peptide_validation --output-dir /r/old --report
101 00:50 python -m scripts.run_cv_peptide_validation --output-dir /r/live --worker network
102 00:01 python -m scripts.cv_validation_job --module scripts.continue_cv_peptide_validation -- --output-dir /r/next
103 00:01 python -m scripts.cv_validation_job --status
104 00:01 python unrelated.py
'''
    found=jobs.active_jobs(output)
    assert [r['pid'] for r in found]==[101,102]
    assert found[0]['role']=='worker' and found[1]['output_dir']=='/r/next'


def test_own_timeout_and_launcher_ancestors_are_not_false_duplicates(monkeypatch):
    def ps(command,**kwargs):
        if 'ppid=' in command:return {12:'11',11:'10',10:'1'}[int(command[-1])]
        return '''10 00:01 python -m scripts.cv_validation_job --module scripts.continue_cv_peptide_validation -- --output-dir /r/self
11 00:01 timeout 3600 python -m scripts.continue_cv_peptide_validation --output-dir /r/self
12 00:01 python -m scripts.continue_cv_peptide_validation --output-dir /r/self
88 00:10 python -m scripts.run_cv_peptide_validation --output-dir /r/other --worker peptide
'''
    monkeypatch.setattr(jobs.subprocess,'check_output',ps)
    assert [r['pid'] for r in jobs.active_jobs(exclude_pid=12)]==[88]


def test_launch_refuses_active_work_before_creating_any_lock_or_process(tmp_path,monkeypatch):
    monkeypatch.setattr(jobs,'BASE',tmp_path)
    monkeypatch.setattr(jobs,'active_jobs',lambda **kw:[dict(pid=99)])
    monkeypatch.setattr(jobs.subprocess,'call',lambda *a,**k:pytest.fail('Duplicate launch'))
    monkeypatch.setattr(sys,'argv',['guard','--module','scripts.run_cv_peptide_validation','--','--output-dir','/r/new'])
    assert jobs.main()==3
    assert list(tmp_path.iterdir())==[]


def test_exclusive_lock_prevents_simultaneous_launch_race(tmp_path,monkeypatch):
    monkeypatch.setattr(jobs,'BASE',tmp_path)
    monkeypatch.setattr(jobs,'active_jobs',lambda **kw:[])
    monkeypatch.setattr(jobs.subprocess,'call',lambda *a,**k:pytest.fail('Locked launch'))
    monkeypatch.setattr(sys,'argv',['guard','--module','scripts.run_cv_peptide_validation','--','--output-dir','/r/new'])
    with (tmp_path/'.cv-peptide-validation.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert jobs.main()==3


def test_status_and_report_do_not_launch(monkeypatch,tmp_path):
    monkeypatch.setattr(jobs,'active_jobs',lambda **kw:[])
    monkeypatch.setattr(jobs.subprocess,'call',lambda *a,**k:pytest.fail('Read-only status launched'))
    jobs.status(tmp_path)
    assert list(tmp_path.iterdir())==[]


def test_stop_targets_only_explicit_duplicate_coordinator(monkeypatch,tmp_path):
    monkeypatch.setattr(jobs,'BASE',tmp_path)
    target=tmp_path/'cv-peptide-validation-repeat';other=tmp_path/'cv-peptide-validation-original'
    killed=[]
    def active(**kw):
        a=[dict(pid=10,role='coordinate',output_dir=str(other))]
        return a if killed else a+[dict(pid=20,role='coordinate',output_dir=str(target))]
    monkeypatch.setattr(jobs,'active_jobs',active)
    monkeypatch.setattr(jobs.os,'getpgid',lambda pid:pid)
    monkeypatch.setattr(jobs.os,'killpg',lambda pid,sig:killed.append(pid))
    assert jobs.stop_run(target)==0 and killed==[20]
    with pytest.raises(ValueError,match='Explicit validation'):jobs.stop_run(tmp_path.parent/'elsewhere')


def test_reuse_requires_saved_hash_and_does_not_overwrite_source(tmp_path):
    source=tmp_path/'stage.npz';dest=tmp_path/'copy.npz'
    np.savez(source,v=np.arange(3));sha=file_sha256(source)
    assert not verified_copy(source,dest,{}) and not dest.exists()
    assert verified_copy(source,dest,{str(source):sha})
    assert file_sha256(source)==sha==file_sha256(dest)
    np.savez(source,v=np.arange(4))
    with pytest.raises(ValueError,match='Changed checkpoint'):verified_copy(source,dest,{str(source):sha})
    assert file_sha256(dest)==sha


def test_continuation_rejects_mixed_runtime_environment(tmp_path):
    folder=tmp_path/'run_records'/'original';folder.mkdir(parents=True)
    path=folder/'environment.json'
    path.write_text(json.dumps(dict(python_version='original-python',packages=dict(jax='0.6.2',numpy='2.4.6'),numerical_environment={'JAX_ENABLE_X64':'true'})))
    args=dict(versions=dict(jax='0.6.2',numpy='2.4.6'),python_version='original-python',settings={'JAX_ENABLE_X64':'true'})
    assert str(path) in verify_environment(tmp_path,**args)
    with pytest.raises(ValueError,match='environment differs'):verify_environment(tmp_path,**(args|dict(versions=dict(jax='changed'))))
    with pytest.raises(ValueError,match='settings differ'):verify_environment(tmp_path,**(args|dict(settings={'JAX_ENABLE_X64':'false'})))


def test_cache_checks_reject_clock_or_response_mismatch():
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    n=2;conditions=n+2;frames=12000
    means=np.zeros((4,conditions,302));voltage=np.full((4,conditions,frames,2),-35.)
    trace=dict(means=means,response=means[:,:n]-means[:,-1:],rmd=voltage,peptide_mean=np.zeros((4,conditions,frames)))
    row=dict(level=0,duplicate_sham_error=0.,finite=True,rhythms=[rhythm_metrics(voltage[i,-1,:,c],.001) for i in range(4) for c in range(2)])
    check_cached_grid(trace,dict(v=np.ones(1)),row,n)
    trace['response'][0,0,0]=1.
    with pytest.raises(ValueError,match='response arithmetic'):check_cached_grid(trace,{},row,n)
    trace['response'][0,0,0]=0.;trace['rmd']=voltage[:,:,:-1]
    with pytest.raises(ValueError,match='shape/clock'):check_cached_grid(trace,{},row,n)


def validation_summary():
    return dict(completed=True,checks_passed=False,e122_response_reproduction_max_difference=0.,
        grids=[dict(level=i,finite=True,duplicate_sham_error=0.,rhythms=[dict(passed=True) for _ in range(8)]) for i in range(3)],
        response_comparisons=[dict(passed=True),dict(passed=True)],restart=dict(passed=True),
        long=[dict(level=i,finite=True,all_cell_voltage_bounded=True,checks=[dict(passed=True),dict(passed=True)]) for i in range(2)],
        long_cycle_checks=[dict(passed=True),dict(passed=True)],batching=[])


def test_report_names_failures_without_reclassifying_passing_components():
    from copy import deepcopy
    from scripts.report_cv_peptide_readiness import validation_failures
    s=validation_summary()
    s['long'][0]['checks'][0]=dict(passed=False,gates=dict(amplitude_retained=False),amplitude_retention=.7)
    s['long_cycle_checks'][1]=dict(passed=False,period_relative_error=.02)
    before=deepcopy(s);r=validation_failures(s)
    assert [f['gate'] for f in r['failed']]==['300s grid0 RMDL','300s coarse/fine cycle RMDR']
    assert r['unavailable']==[] and s==before
    r=validation_failures(validation_summary())
    assert not r['failed'] and r['unavailable']  # Contradictory aggregate is explicit.


def test_validation_report_reads_existing_json_without_launching_or_writing(tmp_path,monkeypatch):
    import subprocess
    from scripts.report_cv_peptide_readiness import report
    (tmp_path/'manifest.json').write_text(json.dumps(dict(experiment='E123-continuation')))
    for arm in ('network','peptide'):
        (tmp_path/arm).mkdir();(tmp_path/arm/'summary.json').write_text(json.dumps(validation_summary()))
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('Reporter launched a process'))
    report(tmp_path)
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
