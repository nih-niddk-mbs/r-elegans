"""E126 stage continuation: reuse audited artifacts, checkpoint each batch, finish validation.

No training or wall-clock validation cutoff. --status/--report/--select-output are
read-only. Source run is never modified. Uses both GPUs for remaining batches.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from scripts import run_midpoint_validation as old
from scripts.experiment_records import capture_run_record,file_sha256

REPO=old.REPO;BASE=old.BASE
SOURCE=BASE/'midpoint-validation-20261124T165552Z-1120675'
MODULE='scripts.continue_midpoint_validation'


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.tmp-'+str(os.getpid()))
    with temp.open('w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path);sync_directory(path.parent)


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def atomic_npz(path,**values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.tmp-'+str(os.getpid()))
    with temp.open('wb') as f:np.savez_compressed(f,**values);f.flush();os.fsync(f.fileno())
    os.replace(temp,path);sync_directory(path.parent)


def active_jobs():
    from scripts.cv_validation_job import ancestor_pids
    found=old.jobs();exclude=ancestor_pids(os.getpid())
    for line in subprocess.check_output(['ps','-ww','-u',str(os.getuid()),'-o','pid=,etime=,args='],text=True).splitlines():
        row=line.split(None,2)
        if len(row)!=3:continue
        pid,elapsed,cmd=row
        if int(pid) in exclude:continue
        try:a=shlex.split(cmd)
        except ValueError:continue
        if any(a[i:i+2]==['-m',MODULE] for i in range(len(a)-1)) and not any(x in a for x in ('--status','--report','--check','--select-output')):
            found.append(dict(pid=int(pid),elapsed=elapsed,command=cmd))
    return found


def select_output(source):
    found=[]
    for name in ('manifest.json','job.json'):
        for path in BASE.glob('midpoint-continuation-*/'+name):
            if json.loads(path.read_text()).get('continuation_source')==str(source.resolve()):found.append(path.parent)
    if found:return max(found)
    return BASE/('midpoint-continuation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(os.getpid()))


def verify_phases(folder,model,level):
    from r_elegans.brain.midpoint_replay import cycle_phases
    state=old.prior.arrays(folder/'starts.npz');meta=json.loads((folder/'phases.json').read_text())
    indices,computed=cycle_phases(meta['reference_voltage_mV'],old.clock(level)[0])
    if computed!={k:v for k,v in meta.items() if k!='reference_voltage_mV'}:raise ValueError('Source phase arithmetic '+str(folder))
    if state['voltage_mV'].shape!=(16,302) or not all(np.isfinite(a).all() for a in state.values()):raise ValueError('Source full state invalid')
    ids=old.prior.arrays(model)['neuron_ids'].tolist()
    if not np.array_equal(state['voltage_mV'][:,ids.index('RMDL')],np.asarray(meta['reference_voltage_mV'])[indices]):raise ValueError('Phase state correspondence')


def finish_level(folder):
    values=old.prior.arrays(folder/'response.npz');final=old.prior.arrays(folder/'final.npz')
    row=dict(metrics=old.full_metrics(values,final),sampling=old.cycle_sampling(values['response']),
        recovery='Saved-data finalization; no forward simulation')
    row['sha256']={str(folder/n):file_sha256(folder/n) for n in ('response.npz','final.npz','starts.npz','phases.json')}
    atomic_json(folder/'metrics.json',row)
    return row


def prepare(root,source):
    source=source.resolve();audit=json.loads((source/'audit.json').read_text())
    if not audit.get('audit_complete') or not audit.get('integrity_passed'):raise ValueError('Audited source integrity required before continuation')
    m=json.loads((source/'manifest.json').read_text())
    for path,h in m['source_sha256'].items():
        if file_sha256(REPO/path)!=h:raise ValueError('Pinned E126 source changed '+path)
    # Copy stage artifacts; original records and all recursive dependencies remain external.
    copied={}
    for arm in m['models']:
        for src in sorted((source/arm).rglob('*')):
            if not src.is_file() or 'run_records' in src.relative_to(source).parts or src.name in ('progress.json','failure.json'):continue
            relative=src.relative_to(source);dest=root/relative;expected=file_sha256(src)
            if src.suffix=='.npz' and audit.get('artifact_sha256',{}).get(str(src))!=expected:raise ValueError('Source artifact hash '+str(src))
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                temp=dest.with_name('.'+dest.name+'.copy-'+str(os.getpid()));shutil.copy2(src,temp);os.replace(temp,dest)
            if file_sha256(dest)!=expected or file_sha256(src)!=expected:raise ValueError('Copy/source changed '+str(src))
            copied[str(relative)]=expected
    tasks=[];reused=[]
    for arm,spec in m['models'].items():
        s=json.loads((root/arm/'summary.json').read_text());level0=s['selected_level']
        if level0 is None:continue
        for level in (level0,level0+1):
            folder=root/arm/f'full-{level}';verify_phases(folder,root/arm/'parent-model.npz',level)
            if (folder/'metrics.json').exists():
                row=json.loads((folder/'metrics.json').read_text())
                for p,h in row['sha256'].items():
                    if file_sha256(p)!=h:raise ValueError('Completed source stage hash '+p)
                if old.full_metrics(old.prior.arrays(folder/'response.npz'),old.prior.arrays(folder/'final.npz'))!=row['metrics']:raise ValueError('Completed stage metric mismatch')
                reused.append(dict(arm=arm,level=level,status='complete_source_stage'));continue
            if (folder/'response.npz').exists() and (folder/'final.npz').exists():
                finish_level(folder);reused.append(dict(arm=arm,level=level,status='recovered_complete_raw_outputs'));continue
            # Original E126 has no per-batch outputs: only this missing level is replayed.
            for begin in range(0,len(m['full_targets']),4):
                tasks.append(dict(arm=arm,level=level,begin=begin,targets=m['full_targets'][begin:begin+4],
                    model_sha256=spec['parent_sha256'],starts_sha256=file_sha256(folder/'starts.npz')))
    m.update(continuation_source=str(source),continuation_policy='Validation allowed to finish; no training/time cutoff. Atomic per-batch checkpoints; no new weights/criteria.',
        copied_sha256=copied,reused_levels=reused,tasks=tasks)
    m['input_sha256'].update({str(p):file_sha256(p) for p in (source/'audit.json',source/'manifest.json',source/'budget_receipt.json')})
    m['source_sha256']['scripts/continue_midpoint_validation.py']=file_sha256(REPO/'scripts/continue_midpoint_validation.py')
    atomic_json(root/'manifest.json',m)


def batch_path(root,task):return root/task['arm']/f"full-{task['level']}"/'batches'/f"batch-{task['begin']:03d}.npz"


def read_batch(root,task):
    path=batch_path(root,task);a=old.prior.arrays(path)
    if json.loads(str(a.pop('provenance_json')))!=task:raise ValueError('Batch provenance '+str(path))
    shapes={'means':(16,6,302),'voltage_min':(16,6),'voltage_max':(16,6),
        'final_voltage_mV':(16,6,302),'final_gates':(16,6,302,35),
        'final_calcium_uM':(16,6,302),'final_indicator':(16,6,302),'final_peptide':(16,6,302),
        'sham_voltage':(16,2,12000,2) if task['begin']==0 else (0,)}
    if set(a)!=set(shapes) or any(a[k].shape!=v for k,v in shapes.items()) or not all(np.isfinite(v).all() for v in a.values()):raise ValueError('Batch shapes/finite '+str(path))
    meta=path.with_suffix('.json')
    if meta.exists() and json.loads(meta.read_text())['sha256']!=file_sha256(path):raise ValueError('Batch checksum '+str(path))
    return a


def worker(root,slot):
    m=json.loads((root/'manifest.json').read_text())
    for path,h in m['source_sha256'].items():
        if file_sha256(REPO/path)!=h:raise ValueError('Runtime source changed '+path)
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,m['models'][('network','peptide')[slot]]['cpus'])
    import jax
    import jax.numpy as jnp
    from r_elegans.brain.peptide_checkpoint import load_model,load_state
    from r_elegans.brain.midpoint_replay import make_replay
    if not jax.config.x64_enabled or jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('One visible float64 GPU required')
    capture_run_record(root/f'worker-{slot}',REPO,dict(slot=slot),phase='E126-batch-continuation')
    cache={}
    for task in m['tasks'][slot::2]:
        path=batch_path(root,task)
        if path.exists():read_batch(root,task);continue
        arm,level=task['arm'],task['level'];key=(arm,level);folder=root/arm/f'full-{level}'
        if key not in cache:
            model=root/arm/'parent-model.npz'
            if file_sha256(model)!=task['model_sha256'] or file_sha256(folder/'starts.npz')!=task['starts_sha256']:raise ValueError('Checkpoint/model changed')
            p,f,c,pp,exc,inh,_,a=load_model(model);ids=a['neuron_ids'].tolist();start=load_state(folder/'starts.npz');dt,step=old.clock(level)
            kernel=make_replay(c,[ids.index('RMDL'),ids.index('RMDR')],dt,step,trace_mode='shams')
            cache[key]=(p,f,pp,exc,inh,ids,start,kernel)
        p,f,pp,exc,inh,ids,start,kernel=cache[key]
        progress=dict(arm=arm,level=level,begin=task['begin'],targets=task['targets'])
        atomic_json(root/f'progress-{slot}.json',dict(stage='computing_batch',**progress));tick=time.monotonic()
        stimuli=np.zeros((6,302));stimuli[np.arange(len(task['targets'])),[ids.index(t) for t in task['targets']]]=.03
        means,end,trace=jax.tree.map(np.asarray,kernel(p,f,pp,exc,inh,start,jnp.asarray(stimuli)))
        atomic_json(root/f'progress-{slot}.json',dict(stage='checkpointing_batch',**progress))
        data=dict(means=means,voltage_min=trace[4].min(axis=-1),voltage_max=trace[5].max(axis=-1),
            sham_voltage=trace[0] if task['begin']==0 else np.empty(0),**{'final_'+k:v for k,v in dict(end.neural._asdict(),peptide=end.peptide).items()})
        if not all(np.isfinite(a).all() for a in data.values()):
            failed=path.parent/'failed'/path.name
            atomic_npz(failed,provenance_json=np.array(json.dumps(task,sort_keys=True)),**data)
            raise ValueError('Nonfinite batch; diagnostic output preserved at '+str(failed))
        atomic_npz(path,provenance_json=np.array(json.dumps(task,sort_keys=True)),**data)
        read_batch(root,task)
        atomic_json(path.with_suffix('.json'),dict(sha256=file_sha256(path),seconds=time.monotonic()-tick,provenance=task))
        atomic_json(root/f'progress-{slot}.json',dict(stage='batch_saved_verified',**progress))
        print(json.dumps(dict(event='batch_saved_verified',seconds=time.monotonic()-tick,**progress)),flush=True)
    atomic_json(root/f'progress-{slot}.json',dict(stage='assigned_batches_complete'))


def finalize(root):
    m=json.loads((root/'manifest.json').read_text());completed=[]
    groups=sorted({(t['arm'],t['level']) for t in m['tasks']})
    for arm,level in groups:
        folder=root/arm/f'full-{level}'
        # An existing new-level completion is immutable and checked by audit.
        if (folder/'metrics.json').exists():continue
        tasks=[t for t in m['tasks'] if (t['arm'],t['level'])==(arm,level)]
        if not all(batch_path(root,t).exists() for t in tasks):continue
        atomic_json(root/'stage.json',dict(stage='assembling_level',arm=arm,level=level))
        responses=[];controls=[];lo=[];hi=[];final={};sham=None
        for task in tasks:
            a=read_batch(root,task);means=a['means'];n=len(task['targets'])
            responses.append(means[:,:n]-means[:,-1:]);controls.append(means[:,-2:]);lo.append(a['voltage_min']);hi.append(a['voltage_max'])
            if task['begin']==0:sham=a['sham_voltage']
            for k,v in a.items():
                if k.startswith('final_'):final.setdefault(k[6:],[]).append(v)
        atomic_npz(folder/'response.npz',response=np.concatenate(responses,axis=1),controls=np.stack(controls),sham_voltage=sham,voltage_min=np.stack(lo),voltage_max=np.stack(hi))
        atomic_npz(folder/'final.npz',**{k:np.stack(v) for k,v in final.items()})
        atomic_json(root/'stage.json',dict(stage='computing_level_metrics',arm=arm,level=level));finish_level(folder);completed.append((arm,level))
    for arm in m['models']:
        path=root/arm/'summary.json';s=json.loads(path.read_text());level=s['selected_level']
        if level is None:continue
        levels=[root/arm/f'full-{i}' for i in (level,level+1)]
        if not all((p/'metrics.json').exists() for p in levels):continue
        rows=[json.loads((p/'metrics.json').read_text()) for p in levels]
        values=[old.prior.arrays(p/'response.npz')['response'] for p in levels]
        check=old.check_response(values[0].mean(0),values[1].mean(0))
        s.update(full_target_complete=True,full_target_response_check=check,parent_validation_passed=bool(check['passed'] and all(r['metrics']['passed'] and r['sampling']['passed'] for r in rows)),stop_reason='validation_finished; sampling criteria unchanged')
        atomic_json(path,s)
    return completed


def coordinate(root):
    m=json.loads((root/'manifest.json').read_text());missing=[t for t in m['tasks'] if not batch_path(root,t).exists()]
    processes=[];logs=[]
    try:
        for slot in range(2):
            assigned=m['tasks'][slot::2]
            if not any(not batch_path(root,t).exists() for t in assigned):continue
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(slot),JAX_PLATFORMS='cuda',JAX_ENABLE_X64='true')
            log=(root/f'worker-{slot}.log').open('a');logs.append(log)
            processes.append(subprocess.Popen([sys.executable,'-u','-m',MODULE,'--output-dir',str(root),'--worker',str(slot)],env=env,stdout=log,stderr=subprocess.STDOUT))
        print(json.dumps(dict(pending_batches=len(missing),validation_time_limit=None)),flush=True)
        while any(p.poll() is None for p in processes):
            progress={}
            for slot in range(2):
                path=root/f'progress-{slot}.json';progress[slot]=json.loads(path.read_text()) if path.exists() else 'no pending assignment'
            print(json.dumps(progress),flush=True);time.sleep(15)
        codes=[p.returncode for p in processes];atomic_json(root/'worker_exit_codes.json',codes)
        finalize(root)
        return 0 if all(c==0 for c in codes) else 2
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()


def audit(root):
    mp=root/'manifest.json'
    if not mp.exists():
        result=dict(audit_complete=True,execution_complete=False,integrity_passed=False,problems=['Preparation incomplete']);atomic_json(root/'audit.json',result);return result
    m=json.loads(mp.read_text());problems=[]
    for relative,h in m['copied_sha256'].items():
        # Only summaries of incomplete levels are deliberately finalized; original source never changes.
        if relative.endswith('/summary.json') or (relative.endswith('/metrics.json') and any(str(Path(relative).parent)==f"{t['arm']}/full-{t['level']}" for t in m['tasks'])):continue
        if file_sha256(root/relative)!=h:problems.append('Reused artifact changed '+relative)
    for task in m['tasks']:
        if batch_path(root,task).exists():read_batch(root,task)
    result=old.audit(root)
    result['problems'].extend(problems);result['integrity_passed']=not result['problems']
    result['parent_validation_passed']=bool(result['parent_validation_passed'] and not problems)
    result['continuation_source']=m['continuation_source'];result['reused_levels']=m['reused_levels']
    result['saved_batches']=sum(batch_path(root,t).exists() for t in m['tasks']);result['planned_batches']=len(m['tasks'])
    result['training_budget_minutes']=0;result['validation_allowed_to_finish']=True
    result['worker_error_history']=[json.loads(p.read_text()) for p in sorted(root.glob('failure-worker-*.json'))]
    result['unavailable'].append('Original interrupted network level1 batches were unsaved; replay from saved starts required. Batch checkpoints apply to this continuation, not historical in-memory work.')
    atomic_json(root/'audit.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path);p.add_argument('--source-run',type=Path,default=SOURCE)
    p.add_argument('--worker',type=int,choices=(0,1));p.add_argument('--coordinate',action='store_true');p.add_argument('--report',action='store_true');p.add_argument('--status',action='store_true');p.add_argument('--check',action='store_true');p.add_argument('--select-output',action='store_true');a=p.parse_args()
    if a.select_output:print(select_output(a.source_run));return
    if a.status or a.check:
        jobs=active_jobs();print('READ-ONLY; no job started. Active jobs:',json.dumps(jobs));return 3 if a.check and jobs else 0
    if not a.output_dir:p.error('--output-dir required')
    root=a.output_dir.resolve()
    if a.report:
        print('READ-ONLY REPORT:',root)
        for name in ('audit.json','budget_receipt.json'):
            path=root/name;r=json.loads(path.read_text()) if path.exists() else {};print(name,json.dumps({k:v for k,v in r.items() if k!='artifact_sha256'}))
        return
    if a.worker is not None:
        try:return worker(root,a.worker)
        except Exception as exc:
            atomic_json(root/f'failure-worker-{a.worker}-{time.time_ns()}.json',dict(slot=a.worker,error=repr(exc)));raise
    if a.coordinate:return coordinate(root)
    if active_jobs():p.error('Active work exists; nothing started')
    import fcntl
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'.cv-peptide-validation.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('Shared GPU lock held; nothing started')
        if active_jobs():p.error('Active work detected after lock; nothing started')
        if root==REPO or REPO in root.parents or root in REPO.parents:p.error('External output required')
        manifest=root/'manifest.json'
        if manifest.exists():
            m=json.loads(manifest.read_text())
            if m['continuation_source']!=str(a.source_run.resolve()):p.error('Continuation source mismatch')
            receipt=root/'budget_receipt.json'
            if receipt.exists() and json.loads(receipt.read_text()).get('all_steps_finished'):
                print('Already completed; nothing started. Use --report:',root);return 0
        else:
            root.mkdir(parents=True,exist_ok=True)
            job=root/'job.json'
            if job.exists() and json.loads(job.read_text())['continuation_source']!=str(a.source_run.resolve()):p.error('Preparation source mismatch')
            atomic_json(job,dict(continuation_source=str(a.source_run.resolve()),validation_allowed_to_finish=True))
            if shutil.disk_usage(root).free<5*1024**3:raise ValueError('Requires5GiB free')
            print('Verifying/copying source stages; no completed forward computation will be repeated:',a.source_run,flush=True)
            prepare(root,a.source_run)
        for path,h in json.loads(manifest.read_text())['source_sha256'].items():
            if file_sha256(REPO/path)!=h:raise ValueError('Runtime changed; do not silently resume')
        started=time.monotonic();capture_run_record(root,REPO,dict(source_run=str(a.source_run.resolve())),phase='E126-continuation-supervisor')
        atomic_json(root/'budget_receipt.json',dict(status='running',training_budget_minutes=0,validation_allowed_to_finish=True,all_steps_finished=False,audit_complete=False))
        code=1
        try:
            from scripts.total_job_budget import run_group
            # No validation wall-clock cutoff. The helper cleans up private descendants after exit.
            code,_=run_group([sys.executable,'-u','-m',MODULE,'--output-dir',str(root),'--coordinate'],None)
        except Exception as exc:traceback.print_exc();atomic_json(root/'failure.json',dict(error=repr(exc)))
        try:result=audit(root)
        except Exception as exc:traceback.print_exc();result=dict(audit_complete=False,execution_complete=False,integrity_passed=False,problems=[repr(exc)]);atomic_json(root/'audit.json',result)
        receipt=dict(status='finished',exit_code=code,timed_out=False,elapsed_seconds=time.monotonic()-started,training_budget_minutes=0,validation_allowed_to_finish=True,
            all_steps_finished=bool(code==0 and result['execution_complete'] and result['audit_complete']),audit_complete=result['audit_complete'],
            parent_validation_passed=bool(result.get('parent_validation_passed')),training_launched=False,model_promoted=False)
        atomic_json(root/'budget_receipt.json',receipt);print(json.dumps(receipt));return 0 if receipt['all_steps_finished'] and result['integrity_passed'] else 2


if __name__=='__main__':main()
