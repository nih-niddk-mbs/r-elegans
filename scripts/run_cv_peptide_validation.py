"""E123 fixed E122 endpoint validation/recovery/batching; no parameter fitting."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from scripts.run_cv_peptide_pilot import write,physical_cpus
from scripts.experiment_records import capture_run_record,file_sha256

REPO=Path(__file__).resolve().parents[1]
SOURCE='/home/carsonc/r-elegans-results/cv-peptide-pilot-20261123T211006Z-970232'
EXPECTED='1a62cbcb0496cad3be8cedc0ff9825b523597d13a41a4eec60251b6b78806ce3'


def compare(a,b):
    delta=float(np.max(abs(a-b)));tolerance=float(1e-7+.1*np.max(abs(b)))
    return dict(maximum_difference=delta,tolerance=tolerance,passed=delta<=tolerance)


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def state_delta(a,b):
    if set(a)!=set(b) or any(a[k].shape!=b[k].shape for k in a):raise ValueError('Continuation fields/shapes differ')
    return {k:float(np.max(abs(a[k]-b[k]))) for k in a}


def recovery(root,arm):
    from r_elegans.brain.peptide_checkpoint import load_model,load_state,save_state,make_rest
    import jax
    model=root/arm/'model.npz';p,f,c,pp,exc,inh,_,a=load_model(model)
    ids=a['neuron_ids'].tolist();rmd=[ids.index('RMDL'),ids.index('RMDR')]
    start=load_state(root/arm/'restart-input.npz')
    end,trace=jax.tree.map(np.asarray,make_rest(c,rmd,1.)(p,f,pp,exc,inh,start))
    save_state(root/arm/'recovered-state.npz',end)
    np.savez_compressed(root/arm/'recovered-trace.npz',rmd=trace[0])


def worker(root,arm):
    m=json.loads((root/'manifest.json').read_text());out=root/arm
    if file_sha256(out/'model.npz')!=m['model_sha256'][arm]:raise ValueError('Full model hash changed')
    for name,sha in m['source_sha256'].items():
        if file_sha256(REPO/name)!=sha:raise ValueError('Validation source changed: '+name)
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,m['worker_cpus'][arm])
    import jax
    import jax.numpy as jnp
    from r_elegans.brain.peptide_checkpoint import load_model,make_rest,save_state
    from r_elegans.brain.peptide_state_replay import make_engine
    from r_elegans.brain.rhythm_qualification import rhythm_metrics,compare_cycles
    if not jax.config.x64_enabled or jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('One visible float64 GPU required')
    capture_run_record(out,REPO,dict(arm=arm),phase='E123-worker')
    p,f,c,pp,exc,inh,stored,a=load_model(out/'model.npz');ids=a['neuron_ids'].tolist()
    rmd=[ids.index('RMDL'),ids.index('RMDR')];targets=m['targets']
    gs=np.zeros((len(targets)+2,len(ids)));gs[np.arange(len(targets)),[ids.index(t) for t in targets]]=m['stimulus_nS']
    endpoint=[];starts=[];replays=[]
    for level in range(3):
        before=time.monotonic();dt=.001/2**level;step_ms=.03125/2**level
        anchors,replay=make_engine(c,rmd,dt=dt,step_ms=step_ms)
        start,phase=anchors(p,f,exc,inh,pp.log_tau_s)
        means,final,trace=jax.tree.map(np.asarray,replay(p,f,exc,inh,pp.log_tau_s,start,jnp.asarray(gs)))
        response=means[:,:len(targets)]-means[:,-1:]
        checks=[rhythm_metrics(trace[0][i,-1,:,cell],dt) for i in range(4) for cell in range(2)]
        row=dict(level=level,dt=dt,step_ms=step_ms,elapsed_seconds=time.monotonic()-before,
            duplicate_sham_error=float(np.max(abs(means[:,-1]-means[:,-2]))),rhythms=checks,
            finite=all(np.isfinite(v).all() for v in jax.tree.leaves((means,final,trace))),phase=phase)
        np.savez_compressed(out/f'grid-{level}.npz',response=response,means=means,rmd=trace[0],peptide_mean=trace[1])
        save_state(out/f'grid-{level}-final.npz',final)
        endpoint.append(row);starts.append(start);replays.append(replay)
        write(out/'progress.json',dict(grids=endpoint));print(json.dumps(dict(arm=arm,grid=level,seconds=row['elapsed_seconds'])),flush=True)
    reference=arrays(out/'e122-state.npz')
    baseline=arrays(out/'grid-0.npz')['response']
    reproduction=float(np.max(abs(baseline-reference['response'])))
    clock_comparisons=[compare(arrays(out/f'grid-{i}.npz')['response'],arrays(out/f'grid-{i+1}.npz')['response']) for i in range(2)]
    # Same-model continuation in another interpreter, not optimizer/RNG resume.
    start=jax.tree.map(lambda x:x[0],stored)
    one=make_rest(c,rmd,1.);two=make_rest(c,rmd,2.)
    halfway,_=one(p,f,pp,exc,inh,start);save_state(out/'restart-input.npz',halfway)
    uninterrupted,trace=jax.tree.map(np.asarray,two(p,f,pp,exc,inh,start));save_state(out/'uninterrupted-state.npz',uninterrupted)
    np.savez_compressed(out/'uninterrupted-trace.npz',rmd=trace[0])
    subprocess.run([sys.executable,'-u','-m','scripts.run_cv_peptide_validation','--output-dir',str(root),'--recover',arm],check=True)
    delta=state_delta(arrays(out/'uninterrupted-state.npz'),arrays(out/'recovered-state.npz'))
    trace_error=float(np.max(abs(trace[0][1000:]-arrays(out/'recovered-trace.npz')['rmd'])))
    restart=dict(state_errors=delta,rmd_trace_error_mV=trace_error,passed=max([trace_error,*delta.values()])<=1e-10,
        scope='Serialized neural+peptide continuation in fresh process; not optimizer resume or machine-crash recovery')
    long=[]
    for level in (0,1):
        dt=.001/2**level
        initial=jax.tree.map(lambda x:x[0],starts[level])
        end,trace=jax.tree.map(np.asarray,make_rest(c,rmd,300.,dt=dt,step_ms=.03125/2**level)(p,f,pp,exc,inh,initial))
        save_state(out/f'long-{level}-final.npz',end)
        np.savez_compressed(out/f'long-{level}.npz',rmd=trace[0],voltage_min=trace[1],voltage_max=trace[2],peptide_mean=trace[3])
        long.append(dict(level=level,dt=dt,checks=[rhythm_metrics(trace[0][:,cell],dt) for cell in range(2)],
            finite=all(np.isfinite(v).all() for v in jax.tree.leaves((end,trace))),
            all_cell_voltage_bounded=bool(trace[1].min()>=-120 and trace[2].max()<=80)))
        write(out/'progress.json',dict(grids=endpoint,long=long,restart=restart))
    coarse=arrays(out/'long-0.npz')['rmd'];fine=arrays(out/'long-1.npz')['rmd'][1::2]
    cycles=[compare_cycles(coarse[:,cell],fine[:,cell],.001) for cell in range(2)]
    # Throughput-only duplicated stimuli; no claim of extra biological labels.
    benchmarks=[]
    for copies in (1,2,4):
        g=np.concatenate([gs[:-2]]*copies+[gs[-2:]])
        previous=benchmarks[-1] if benchmarks else None
        seconds=previous['warm_seconds'] if previous else endpoint[0]['elapsed_seconds']
        prior_streams=previous['streams'] if previous else 4*len(gs)
        projected=2.6*seconds*4*len(g)/prior_streams
        if time.time()+max(180.,projected)>m['worker_deadline_unix']:break
        before=time.monotonic();jax.block_until_ready(replays[0](p,f,exc,inh,pp.log_tau_s,starts[0],jnp.asarray(g)))
        first=time.monotonic()-before
        before=time.monotonic();jax.block_until_ready(replays[0](p,f,exc,inh,pp.log_tau_s,starts[0],jnp.asarray(g)))
        elapsed=time.monotonic()-before
        benchmarks.append(dict(target_slots=len(g)-2,streams=4*len(g),first_call_seconds=first,
            warm_seconds=elapsed,target_trials_per_second=4*(len(g)-2)/elapsed))
    result=dict(completed=True,arm=arm,grids=endpoint,response_comparisons=clock_comparisons,
        e122_response_reproduction_max_difference=reproduction,restart=restart,long=long,long_cycle_checks=cycles,batching=benchmarks,
        checks_passed=bool(reproduction<=1e-10 and all(r['finite'] and r['duplicate_sham_error']<1e-12 and all(x['passed'] for x in r['rhythms']) for r in endpoint)
            and all(x['passed'] for x in clock_comparisons) and restart['passed'] and
            all(r['finite'] and r['all_cell_voltage_bounded'] and all(x['passed'] for x in r['checks']) for r in long) and all(x['passed'] for x in cycles)),
        unavailable=['Full atlas/all five fold fits','Peptide gain/kinetic optimization','Body, swimming and sensory start/stop',
            'Exact optimizer resume, disk-full and machine-reboot fault recovery','Complete batch sweep' if len(benchmarks)<3 else 'Batch results apply only to this workload'])
    write(out/'summary.json',result)


def prepare(root,source):
    from r_elegans.brain.extrasynaptic import load_extrasynaptic_asset
    from r_elegans.brain.peptide_checkpoint import bundle_model
    s=json.loads((source/'manifest.json').read_text());audit=json.loads((source/'audit.json').read_text());budget=json.loads((source/'budget_receipt.json').read_text())
    if not (audit['integrity_passed'] and audit['execution_complete'] and budget['all_steps_finished'] and not budget['timed_out']):raise ValueError('Complete clean E122 audit required')
    cpus=physical_cpus()
    if len(cpus)<8:raise ValueError('Eight physical cores required')
    asset=load_extrasynaptic_asset();asset_sha=file_sha256(REPO/'r_elegans/assets/extrasynaptic_v1.npz')
    inputs={str(source/name):file_sha256(source/name) for name in ('manifest.json','audit.json','budget_receipt.json','fold.json')}
    for arm in ('network','peptide'):
        out=root/arm;out.mkdir();summary=source/arm/'summary.json';inputs[str(summary)]=file_sha256(summary)
        r=json.loads(summary.read_text())['selected'];network=source/arm/r['name']/'network.npz';state=network.with_name('state.npz')
        if file_sha256(network)!=EXPECTED or r['sha256'][str(network)]!=EXPECTED or r['peptide_log_gains']!=[0.,0.]:raise ValueError('Pinned E122 endpoint/prior mismatch')
        for path in (network,state):
            if file_sha256(path)!=r['sha256'][str(path)]:raise ValueError('E122 artifact changed')
            inputs[str(path)]=file_sha256(path)
        shutil.copy2(state,out/'e122-state.npz')
        bundle_model(network,state,out/'model.npz',enabled=arm=='peptide',asset=asset,asset_sha256=asset_sha,network_sha256=EXPECTED)
    m=dict(experiment='E123',source=str(source),targets=s['targets'],stimulus_nS=s['stimulus_nS'],input_sha256=inputs,
        worker_cpus=dict(network=cpus[:4],peptide=cpus[4:8]),worker_deadline_unix=time.time()+50*60,
        model_sha256={arm:file_sha256(root/arm/'model.npz') for arm in ('network','peptide')})
    m['source_sha256']={name:file_sha256(REPO/name) for name in ('scripts/run_cv_peptide_validation.py',
        'r_elegans/brain/peptide_checkpoint.py','r_elegans/brain/peptide_state_replay.py',
        'r_elegans/brain/extrasynaptic.py','r_elegans/brain/conductance_network.py','r_elegans/brain/single_compartment.py')}
    write(root/'manifest.json',m)


def coordinator(root):
    processes=[];logs=[]
    try:
        for gpu,arm in enumerate(('network','peptide')):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',JAX_ENABLE_X64='true',XLA_PYTHON_CLIENT_PREALLOCATE='false')
            log=(root/(arm+'.log')).open('w');logs.append(log)
            processes.append(subprocess.Popen([sys.executable,'-u','-m','scripts.run_cv_peptide_validation','--output-dir',str(root),'--worker',arm],env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            print(json.dumps(dict(running=[arm for arm,p in zip(('network','peptide'),processes) if p.poll() is None])),flush=True)
            try:
                result=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                with (root/'gpu_telemetry.jsonl').open('a') as f:f.write(json.dumps(dict(time_unix=time.time(),returncode=result.returncode,rows=result.stdout.splitlines()))+'\n')
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


def post_audit(root):
    from r_elegans.brain.peptide_checkpoint import validate
    from r_elegans.brain.rhythm_qualification import rhythm_metrics,compare_cycles
    m=json.loads((root/'manifest.json').read_text());problems=[];results={}
    for name,sha in m['input_sha256'].items():
        if file_sha256(name)!=sha:problems.append('Changed source input '+name)
    for name,sha in m['source_sha256'].items():
        if file_sha256(REPO/name)!=sha:problems.append('Changed runtime source '+name)
    for record in root.glob('**/run_records/*/source_record.json'):
        r=json.loads(record.read_text())
        for name,sha in r['source_sha256'].items():
            if file_sha256(record.parent/'source'/name)!=sha:problems.append('Source archive '+name)
    hashes={str(p):file_sha256(p) for p in root.rglob('*.npz')}
    for arm in ('network','peptide'):
        out=root/arm
        if file_sha256(out/'model.npz')!=m['model_sha256'][arm]:problems.append('Model changed '+arm)
        model=arrays(out/'model.npz');validate(model)
        if bool(model['extrasynaptic_enabled'])!=(arm=='peptide'):problems.append('Enabled flag '+arm)
        # Bundle must preserve every original wired field exactly.
        original=arrays(next(Path(p) for p in m['input_sha256'] if p.endswith('/'+arm+'/eval-0009/network.npz')))
        if any(not np.array_equal(model[k] if k!='schema_version' else model['network_schema_version'],v) for k,v in original.items()):problems.append('Wired/frozen fields '+arm)
        path=out/'summary.json'
        if not path.exists():problems.append('Incomplete '+arm);continue
        s=json.loads(path.read_text());responses=[arrays(out/f'grid-{i}.npz') for i in range(3)]
        if [compare(responses[i]['response'],responses[i+1]['response']) for i in range(2)]!=s['response_comparisons']:problems.append('Grid arithmetic '+arm)
        for i,r in enumerate(responses):
            checks=[rhythm_metrics(r['rmd'][phase,-1,:,cell],.001/2**i) for phase in range(4) for cell in range(2)]
            if checks!=s['grids'][i]['rhythms'] or np.max(abs(r['means'][:,-1]-r['means'][:,-2]))!=s['grids'][i]['duplicate_sham_error']:problems.append('Rhythm/control arithmetic '+arm)
            finite=all(np.isfinite(v).all() for v in list(r.values())+list(arrays(out/f'grid-{i}-final.npz').values()))
            if finite!=s['grids'][i]['finite']:problems.append('Finite grid check '+arm)
        delta=state_delta(arrays(out/'uninterrupted-state.npz'),arrays(out/'recovered-state.npz'))
        if delta!=s['restart']['state_errors']:problems.append('Recovery arithmetic '+arm)
        trace_error=float(np.max(abs(arrays(out/'uninterrupted-trace.npz')['rmd'][1000:]-arrays(out/'recovered-trace.npz')['rmd'])))
        restart_passed=max([trace_error,*delta.values()])<=1e-10
        if trace_error!=s['restart']['rmd_trace_error_mV'] or restart_passed!=s['restart']['passed']:problems.append('Recovery trace '+arm)
        reproduction=float(np.max(abs(responses[0]['response']-arrays(out/'e122-state.npz')['response'])))
        if reproduction!=s['e122_response_reproduction_max_difference']:problems.append('E122 endpoint reproduction '+arm)
        for i in (0,1):
            r=arrays(out/f'long-{i}.npz');checks=[rhythm_metrics(r['rmd'][:,cell],.001/2**i) for cell in range(2)]
            if checks!=s['long'][i]['checks'] or bool(r['voltage_min'].min()>=-120 and r['voltage_max'].max()<=80)!=s['long'][i]['all_cell_voltage_bounded']:problems.append('Long dynamics arithmetic '+arm)
            finite=all(np.isfinite(v).all() for v in list(r.values())+list(arrays(out/f'long-{i}-final.npz').values()))
            if finite!=s['long'][i]['finite']:problems.append('Long finite check '+arm)
        coarse=arrays(out/'long-0.npz')['rmd'];fine=arrays(out/'long-1.npz')['rmd'][1::2]
        if [compare_cycles(coarse[:,i],fine[:,i],.001) for i in range(2)]!=s['long_cycle_checks']:problems.append('Long cycle arithmetic '+arm)
        passed=bool(reproduction<=1e-10 and restart_passed and all(r['finite'] and r['duplicate_sham_error']<1e-12 and all(x['passed'] for x in r['rhythms']) for r in s['grids'])
            and all(x['passed'] for x in s['response_comparisons']) and all(r['finite'] and r['all_cell_voltage_bounded'] and all(x['passed'] for x in r['checks']) for r in s['long']) and all(x['passed'] for x in s['long_cycle_checks']))
        if passed!=s['checks_passed']:problems.append('Qualification arithmetic '+arm)
        results[arm]=s
    result=dict(audit_complete=True,execution_complete=len(results)==2,integrity_passed=not problems,problems=problems,
        endpoint_checks_passed=bool(len(results)==2 and all(s['checks_passed'] for s in results.values())),
        artifact_sha256=hashes,models=m['model_sha256'],training_launched=False,behavior_qualified=False,
        unavailable=['No optimizer/crash recovery tested','No fivefold fitting or full-atlas validation','No biological/behavior qualification'])
    write(root/'audit.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--source',type=Path,default=Path(SOURCE))
    p.add_argument('--worker',choices=('network','peptide'));p.add_argument('--recover',choices=('network','peptide'));p.add_argument('--coordinate',action='store_true');p.add_argument('--report',action='store_true')
    a=p.parse_args();root=a.output_dir.resolve()
    if a.report:
        print('Run:',root)
        for arm in ('network','peptide'):
            path=root/arm/'summary.json'
            if path.exists():
                s=json.loads(path.read_text());print(arm,{k:s[k] for k in ('completed','checks_passed','response_comparisons','restart','batching','unavailable')})
            else:print(arm,'Not complete; inspect '+str(root/(arm+'.log')))
        for name in ('audit.json','budget_receipt.json'):
            path=root/name;r=json.loads(path.read_text()) if path.exists() else {}
            print(name,{k:v for k,v in r.items() if k!='artifact_sha256'})
        return
    if a.worker:return worker(root,a.worker)
    if a.recover:return recovery(root,a.recover)
    if a.coordinate:return coordinator(root)
    if root==REPO or REPO in root.parents or root in REPO.parents:p.error('External output required')
    root.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    if shutil.disk_usage(root).free<5*1024**3:raise ValueError('At least5GiB free required')
    capture_run_record(root,REPO,vars(a),phase='E123-validation')
    code=1;timed_out=False
    try:
        prepare(root,a.source.resolve())
        from scripts.total_job_budget import run_group
        code,timed_out=run_group([sys.executable,'-u','-m','scripts.run_cv_peptide_validation','--output-dir',str(root),'--coordinate'],max(.01,54*60-(time.monotonic()-started)))
        result=post_audit(root)
    except Exception as exc:
        traceback.print_exc();code=code or 1
        result=dict(audit_complete=False,execution_complete=False,integrity_passed=False,problems=[repr(exc)])
        write(root/'audit.json',result)
    receipt=dict(total_minutes=60,elapsed_seconds=time.monotonic()-started,exit_code=code,timed_out=timed_out,
        all_steps_finished=bool(code==0 and result['execution_complete']),audit_complete=result['audit_complete'],
        validation_passed=bool(result.get('endpoint_checks_passed') and result['integrity_passed']),training_launched=False)
    write(root/'budget_receipt.json',receipt);print(json.dumps(receipt),flush=True)
    return 0 if receipt['all_steps_finished'] and result['integrity_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
