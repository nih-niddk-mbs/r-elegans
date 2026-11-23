"""Continue completed E123 validation stages in a new run; no fitting.

Original sources/receipts are immutable. Completed stages are copied only after
checking original post-run artifact hashes and clocks. Interrupted JIT work must
restart; completed grid replays and long traces are not rerun.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
import numpy as np
from scripts.experiment_records import capture_run_record,file_sha256
from scripts.run_cv_peptide_pilot import write,physical_cpus
from scripts.run_cv_peptide_validation import REPO,arrays,compare,state_delta,post_audit

SOURCE='/home/carsonc/r-elegans-results/cv-peptide-validation-20261123T220426Z-974332'


def verify_environment(source,*,versions=None,python_version=None,settings=None):
    from importlib.metadata import version
    versions=versions if versions is not None else {n:version(n) for n in ('jax','jaxlib','numpy','scipy')}
    python_version=python_version if python_version is not None else sys.version
    settings=settings if settings is not None else {k:os.environ.get(k) for k in ('JAX_ENABLE_X64','JAX_DEFAULT_MATMUL_PRECISION','XLA_FLAGS')}
    records=list(source.glob('run_records/*/environment.json'))
    records+=list(source.glob('network/run_records/*/environment.json'))+list(source.glob('peptide/run_records/*/environment.json'))
    if not records:raise ValueError('Original environment evidence missing; cached outputs cannot be mixed blindly')
    for path in records:
        r=json.loads(path.read_text());packages={k.lower():v for k,v in r['packages'].items()}
        if r['python_version']!=python_version or any(packages.get(n)!=v for n,v in versions.items()):raise ValueError('Source Python/JAX/NumPy/SciPy environment differs')
        if any(r['numerical_environment'].get(k)!=v for k,v in settings.items()):raise ValueError('Source numerical settings differ')
    return {str(p):file_sha256(p) for p in records}


def verified_copy(source,destination,hashes):
    """Missing/uncommitted artifacts are not checkpoints; changed ones fail."""
    source=Path(source)
    if not source.exists() or str(source) not in hashes:return False
    expected=hashes[str(source)]
    if file_sha256(source)!=expected:raise ValueError('Changed checkpoint: '+str(source))
    # Validate zip integrity/array readability before treating it as completed.
    arrays(source)
    shutil.copy2(source,destination)
    if file_sha256(destination)!=expected or file_sha256(source)!=expected:raise ValueError('Checkpoint changed during copy')
    return True


def check_cached_grid(trace,final,row,target_count):
    """Independent saved-data checks, including on incomplete source arms."""
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    dt=.001/2**row['level'];frames=round(12./dt);conditions=target_count+2
    shapes=dict(response=(4,target_count,302),means=(4,conditions,302),rmd=(4,conditions,frames,2),peptide_mean=(4,conditions,frames))
    if any(trace[k].shape!=shape for k,shape in shapes.items()):raise ValueError('Cached grid shape/clock mismatch')
    if not np.array_equal(trace['response'],trace['means'][:,:target_count]-trace['means'][:,-1:]):raise ValueError('Cached response arithmetic mismatch')
    checks=[rhythm_metrics(trace['rmd'][i,-1,:,cell],dt) for i in range(4) for cell in range(2)]
    finite=all(np.isfinite(v).all() for v in list(trace.values())+list(final.values()))
    duplicate=float(np.max(abs(trace['means'][:,-1]-trace['means'][:,-2])))
    if checks!=row['rhythms'] or finite!=row['finite'] or duplicate!=row['duplicate_sham_error']:raise ValueError('Cached grid metric mismatch')


def prepare(root,source,minutes):
    budget=json.loads((source/'budget_receipt.json').read_text());audit=json.loads((source/'audit.json').read_text())
    if not budget.get('audit_complete') or not audit.get('audit_complete'):raise ValueError('Complete source post-run audit required')
    allowed={'Incomplete network','Incomplete peptide'}
    if set(audit.get('problems',[]))-allowed:raise ValueError('Source has integrity problems beyond incomplete work')
    old=json.loads((source/'manifest.json').read_text());hashes=audit['artifact_sha256']
    for name,sha in old['source_sha256'].items():
        if file_sha256(REPO/name)!=sha:raise ValueError('Source mismatch: '+name)
    for path,sha in old['input_sha256'].items():
        if file_sha256(path)!=sha:raise ValueError('Input changed: '+path)
    evidence=dict(old['input_sha256'])|verify_environment(source);resume={};cpus=physical_cpus()
    if len(cpus)<8:raise ValueError('Eight physical cores required')
    for name in ('manifest.json','budget_receipt.json','audit.json'):
        evidence[str(source/name)]=file_sha256(source/name)
    for arm in ('network','peptide'):
        out=root/arm;out.mkdir();prior=source/arm
        for name in ('model.npz','e122-state.npz'):
            if not verified_copy(prior/name,out/name,hashes):raise ValueError('Missing audited '+name)
            evidence[str(prior/name)]=hashes[str(prior/name)]
        if file_sha256(out/'model.npz')!=old['model_sha256'][arm]:raise ValueError('Source model changed')
        progress=json.loads((prior/'progress.json').read_text()) if (prior/'progress.json').exists() else {}
        rows={int(r['level']):r for r in progress.get('grids',[])}
        resume[arm]=dict(grids={},long=[],restart=False)
        if (prior/'progress.json').exists():evidence[str(prior/'progress.json')]=file_sha256(prior/'progress.json')
        for level in range(3):
            names=[f'grid-{level}.npz',f'grid-{level}-final.npz']
            row=rows.get(level)
            committed=all(str(prior/n) in hashes and (prior/n).exists() for n in names)
            if row is not None and committed:
                if row['dt']!=.001/2**level or row['step_ms']!=.03125/2**level:raise ValueError('Cached grid clock mismatch')
                for name in names:
                    verified_copy(prior/name,out/name,hashes);evidence[str(prior/name)]=hashes[str(prior/name)]
                check_cached_grid(arrays(out/names[0]),arrays(out/names[1]),row,len(old['targets']))
                resume[arm]['grids'][str(level)]=row
            # New continuations save anchors too; original E123 did not.
            name=f'grid-{level}-start.npz'
            if verified_copy(prior/name,out/name,hashes):evidence[str(prior/name)]=hashes[str(prior/name)]
        restart_names=['restart-input.npz','uninterrupted-state.npz','uninterrupted-trace.npz','recovered-state.npz','recovered-trace.npz']
        if all(str(prior/n) in hashes and (prior/n).exists() for n in restart_names):
            readable=True
            for name in restart_names:
                if file_sha256(prior/name)!=hashes[str(prior/name)]:raise ValueError('Changed recovery artifact')
                try:arrays(prior/name)
                except (ValueError,EOFError,zipfile.BadZipFile):readable=False
            if readable:
                for name in restart_names:
                    verified_copy(prior/name,out/name,hashes);evidence[str(prior/name)]=hashes[str(prior/name)]
                resume[arm]['restart']=True
            else:resume[arm]['unusable_recovery_reason']='At least one archive was incomplete at timeout; rerun recovery stage'
        long_committed={int(r['level']) for r in progress.get('long',[])}
        for level in (0,1):
            names=[f'long-{level}.npz',f'long-{level}-final.npz']
            if level in long_committed and all(str(prior/n) in hashes and (prior/n).exists() for n in names):
                for name in names:
                    verified_copy(prior/name,out/name,hashes);evidence[str(prior/name)]=hashes[str(prior/name)]
                resume[arm]['long'].append(level)
    m=old|dict(experiment='E123-continuation',resume_source=str(source),resume=resume,input_sha256=evidence,
        worker_cpus=dict(network=cpus[:4],peptide=cpus[4:8]),worker_deadline_unix=time.time()+(minutes-10)*60,total_minutes=minutes)
    m['source_sha256']=dict(old['source_sha256'])|{str(Path(__file__).relative_to(REPO)):file_sha256(__file__),
        'scripts/cv_validation_job.py':file_sha256(REPO/'scripts/cv_validation_job.py')}
    write(root/'manifest.json',m)
    print('Verified reusable stages:',json.dumps(resume),flush=True)


def worker(root,arm):
    m=json.loads((root/'manifest.json').read_text());out=root/arm;reuse=m['resume'][arm]
    print(json.dumps(dict(arm=arm,status='starting_continuation',reusable=reuse)),flush=True)
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,m['worker_cpus'][arm])
    for path,sha in m['source_sha256'].items():
        if file_sha256(REPO/path)!=sha:raise ValueError('Runtime source changed')
    if file_sha256(out/'model.npz')!=m['model_sha256'][arm]:raise ValueError('Model hash mismatch')
    import jax
    import jax.numpy as jnp
    from r_elegans.brain.peptide_checkpoint import load_model,load_state,make_rest,save_state
    from r_elegans.brain.peptide_state_replay import make_engine
    from r_elegans.brain.rhythm_qualification import rhythm_metrics,compare_cycles
    if not jax.config.x64_enabled or jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('One visible float64 GPU required')
    capture_run_record(out,REPO,dict(arm=arm),phase='E123-stage-continuation')
    p,f,c,pp,exc,inh,stored,a=load_model(out/'model.npz');ids=a['neuron_ids'].tolist()
    rmd=[ids.index('RMDL'),ids.index('RMDR')];targets=m['targets'];engines={};starts={};phases={}
    gs=np.zeros((len(targets)+2,len(ids)));gs[np.arange(len(targets)),[ids.index(t) for t in targets]]=m['stimulus_nS']
    endpoint=[];long=[];restart=None;benchmarks=[]
    def progress():write(out/'progress.json',dict(grids=endpoint,long=long,restart=restart,batching=benchmarks))
    def engine(level):
        if level not in engines:engines[level]=make_engine(c,rmd,dt=.001/2**level,step_ms=.03125/2**level)
        return engines[level]
    def get_start(level):
        if level not in starts:
            path=out/f'grid-{level}-start.npz'
            if path.exists():starts[level]=load_state(path)
            else:
                print(json.dumps(dict(arm=arm,grid=level,status='reconstructing_unsaved_phase_anchors')),flush=True)
                starts[level],phases[level]=engine(level)[0](p,f,exc,inh,pp.log_tau_s)
                save_state(path,starts[level])
        return starts[level]
    for level in range(3):
        if str(level) in reuse['grids']:
            row=reuse['grids'][str(level)]
            print(json.dumps(dict(arm=arm,grid=level,status='reused_verified_complete_stage')),flush=True)
        else:
            before=time.monotonic();dt=.001/2**level;start=get_start(level)
            means,final,trace=jax.tree.map(np.asarray,engine(level)[1](p,f,exc,inh,pp.log_tau_s,start,jnp.asarray(gs)))
            row=dict(level=level,dt=dt,step_ms=.03125/2**level,elapsed_seconds=time.monotonic()-before,
                duplicate_sham_error=float(np.max(abs(means[:,-1]-means[:,-2]))),
                rhythms=[rhythm_metrics(trace[0][i,-1,:,cell],dt) for i in range(4) for cell in range(2)],
                finite=all(np.isfinite(v).all() for v in jax.tree.leaves((means,final,trace))),phase=phases.get(level))
            np.savez_compressed(out/f'grid-{level}.npz',response=means[:,:len(targets)]-means[:,-1:],means=means,rmd=trace[0],peptide_mean=trace[1])
            save_state(out/f'grid-{level}-final.npz',final)
            print(json.dumps(dict(arm=arm,grid=level,status='computed',seconds=row['elapsed_seconds'])),flush=True)
        endpoint.append(row);progress()
    reproduction=float(np.max(abs(arrays(out/'grid-0.npz')['response']-arrays(out/'e122-state.npz')['response'])))
    comparisons=[compare(arrays(out/f'grid-{i}.npz')['response'],arrays(out/f'grid-{i+1}.npz')['response']) for i in range(2)]
    if not reuse['restart']:
        start=jax.tree.map(lambda x:x[0],stored)
        halfway,_=make_rest(c,rmd,1.)(p,f,pp,exc,inh,start);save_state(out/'restart-input.npz',halfway)
        uninterrupted,trace=jax.tree.map(np.asarray,make_rest(c,rmd,2.)(p,f,pp,exc,inh,start))
        save_state(out/'uninterrupted-state.npz',uninterrupted);np.savez_compressed(out/'uninterrupted-trace.npz',rmd=trace[0])
        subprocess.run([sys.executable,'-u','-m','scripts.run_cv_peptide_validation','--output-dir',str(root),'--recover',arm],check=True)
    delta=state_delta(arrays(out/'uninterrupted-state.npz'),arrays(out/'recovered-state.npz'))
    error=float(np.max(abs(arrays(out/'uninterrupted-trace.npz')['rmd'][1000:]-arrays(out/'recovered-trace.npz')['rmd'])))
    restart=dict(state_errors=delta,rmd_trace_error_mV=error,passed=max([error,*delta.values()])<=1e-10,
        scope='Serialized neural+peptide continuation; not optimizer/crash resume')
    progress()
    for level in (0,1):
        dt=.001/2**level
        if level not in reuse['long']:
            print(json.dumps(dict(arm=arm,stage='long',level=level,status='starting')),flush=True)
            start=jax.tree.map(lambda x:x[0],get_start(level))
            end,trace=jax.tree.map(np.asarray,make_rest(c,rmd,300.,dt=dt,step_ms=.03125/2**level)(p,f,pp,exc,inh,start))
            save_state(out/f'long-{level}-final.npz',end)
            np.savez_compressed(out/f'long-{level}.npz',rmd=trace[0],voltage_min=trace[1],voltage_max=trace[2],peptide_mean=trace[3])
        r=arrays(out/f'long-{level}.npz');end=arrays(out/f'long-{level}-final.npz')
        if r['rmd'].shape!=(round(300./dt),2):raise ValueError('Cached long trace clock/shape mismatch')
        long.append(dict(level=level,dt=dt,checks=[rhythm_metrics(r['rmd'][:,cell],dt) for cell in range(2)],
            finite=all(np.isfinite(v).all() for v in list(r.values())+list(end.values())),
            all_cell_voltage_bounded=bool(r['voltage_min'].min()>=-120 and r['voltage_max'].max()<=80)))
        progress()
    coarse=arrays(out/'long-0.npz')['rmd'];fine=arrays(out/'long-1.npz')['rmd'][1::2]
    cycles=[compare_cycles(coarse[:,i],fine[:,i],.001) for i in range(2)]
    for copies in (1,2,4):
        g=np.concatenate([gs[:-2]]*copies+[gs[-2:]])
        prior=benchmarks[-1] if benchmarks else None
        seconds=prior['warm_seconds'] if prior else endpoint[0]['elapsed_seconds']
        streams=prior['streams'] if prior else 4*len(gs)
        if time.time()+max(180.,2.6*seconds*4*len(g)/streams)>m['worker_deadline_unix']:break
        start=get_start(0);replay=engine(0)[1]
        before=time.monotonic();jax.block_until_ready(replay(p,f,exc,inh,pp.log_tau_s,start,jnp.asarray(g)));cold=time.monotonic()-before
        before=time.monotonic();jax.block_until_ready(replay(p,f,exc,inh,pp.log_tau_s,start,jnp.asarray(g)));elapsed=time.monotonic()-before
        benchmarks.append(dict(target_slots=len(g)-2,streams=4*len(g),first_call_seconds=cold,warm_seconds=elapsed,target_trials_per_second=4*(len(g)-2)/elapsed));progress()
    result=dict(completed=True,arm=arm,grids=endpoint,response_comparisons=comparisons,e122_response_reproduction_max_difference=reproduction,
        restart=restart,long=long,long_cycle_checks=cycles,batching=benchmarks,reuse=reuse,
        checks_passed=bool(reproduction<=1e-10 and restart['passed'] and all(r['finite'] and r['duplicate_sham_error']<1e-12 and all(x['passed'] for x in r['rhythms']) for r in endpoint)
            and all(x['passed'] for x in comparisons) and all(r['finite'] and r['all_cell_voltage_bounded'] and all(x['passed'] for x in r['checks']) for r in long) and all(x['passed'] for x in cycles)),
        unavailable=['No fitting/full fivefold/behavior qualification','Interrupted computations and JIT compilation restart','Peptide gains fixed','No optimizer or reboot recovery'])
    write(out/'summary.json',result)


def coordinate(root):
    processes=[];logs=[]
    try:
        for gpu,arm in enumerate(('network','peptide')):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',JAX_ENABLE_X64='true',XLA_PYTHON_CLIENT_PREALLOCATE='false')
            log=(root/(arm+'.log')).open('w');logs.append(log)
            processes.append(subprocess.Popen([sys.executable,'-u','-m','scripts.continue_cv_peptide_validation','--output-dir',str(root),'--worker',arm],env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            print(json.dumps(dict(running=[arm for arm,p in zip(('network','peptide'),processes) if p.poll() is None])),flush=True)
            try:
                r=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                with (root/'gpu_telemetry.jsonl').open('a') as f:f.write(json.dumps(dict(time_unix=time.time(),returncode=r.returncode,rows=r.stdout.splitlines()))+'\n')
            except (OSError,subprocess.TimeoutExpired):pass
            time.sleep(15)
        write(root/'worker_exit_codes.json',{arm:p.returncode for arm,p in zip(('network','peptide'),processes)})
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for f in logs:f.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--source',type=Path,default=Path(SOURCE))
    p.add_argument('--minutes',type=int,default=240);p.add_argument('--worker',choices=('network','peptide'));p.add_argument('--coordinate',action='store_true');p.add_argument('--report',action='store_true')
    a=p.parse_args();root=a.output_dir.resolve()
    if a.report:
        # Old reporter reads both schemas; it never starts a job.
        from scripts.run_cv_peptide_validation import main as report_main
        sys.argv=[sys.argv[0],'--output-dir',str(root),'--report'];return report_main()
    if a.worker:return worker(root,a.worker)
    if a.coordinate:return coordinate(root)
    if not 30<=a.minutes<=1440:p.error('30-1440 total minutes required')
    from scripts.cv_validation_job import active_jobs
    if active_jobs(exclude_pid=os.getpid()):p.error('Validation already active; nothing started')
    if root==REPO or REPO in root.parents or root in REPO.parents:p.error('External output required')
    root.mkdir(parents=True,exist_ok=False);started=time.monotonic();code=1;timed_out=False
    capture_run_record(root,REPO,vars(a),phase='E123-stage-continuation')
    try:
        prepare(root,a.source.resolve(),a.minutes)
        from scripts.total_job_budget import run_group
        code,timed_out=run_group([sys.executable,'-u','-m','scripts.continue_cv_peptide_validation','--output-dir',str(root),'--coordinate'],max(.01,(a.minutes-6)*60-(time.monotonic()-started)))
        result=post_audit(root)
    except Exception as exc:
        traceback.print_exc();code=code or 1;result=dict(audit_complete=False,execution_complete=False,integrity_passed=False,problems=[repr(exc)])
        write(root/'audit.json',result)
    receipt=dict(total_minutes=a.minutes,elapsed_seconds=time.monotonic()-started,exit_code=code,timed_out=timed_out,
        all_steps_finished=bool(code==0 and result['execution_complete']),audit_complete=result['audit_complete'],
        validation_passed=bool(result.get('endpoint_checks_passed') and result['integrity_passed']),training_launched=False)
    write(root/'budget_receipt.json',receipt);print(json.dumps(receipt),flush=True)
    return 0 if receipt['all_steps_finished'] and result['integrity_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
