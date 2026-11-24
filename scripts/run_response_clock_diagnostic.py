"""E125 fixed-weight response clock/state diagnostic; default120min total, no fitting."""
import argparse
import fcntl
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
from scripts.experiment_records import capture_run_record,file_sha256
from scripts.run_cv_peptide_pilot import write,physical_cpus
from r_elegans.brain.response_clock_probe import CLOCKS,CASES,response_differences

REPO=Path(__file__).resolve().parents[1]
BASE=Path('/home/carsonc/r-elegans-results')
SOURCE=BASE/'cv-peptide-fit-20261124T042745Z-989175'
MODULE='scripts.run_response_clock_diagnostic'
MODELS={'network':(0,'bc411528999e8150fa48fa622058496d682031ea3a8fbd29ba630ac0cf206ca7'),
        'peptide':(1,'f626ded0494902820432046f1031cafe35b65216cb25ecc6ffb1cee2d829e652')}
TARGETS=['RIVL','RMDR','RMDL']
OBSERVED=['RIVL','RMDVL','RMDR','RMDL']


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def jobs():
    from scripts.run_cv_peptide_fit import active_jobs
    from scripts.cv_validation_job import ancestor_pids
    found=active_jobs();exclude=ancestor_pids(os.getpid())
    output=subprocess.check_output(['ps','-ww','-u',str(os.getuid()),'-o','pid=,etime=,args='],text=True)
    for line in output.splitlines():
        fields=line.split(None,2)
        if len(fields)!=3:continue
        pid,elapsed,cmd=fields
        if int(pid) in exclude:continue
        try:a=shlex.split(cmd)
        except ValueError:continue
        if any(a[i:i+2]==['-m',MODULE] for i in range(len(a)-1)) and not any(x in a for x in ('--report','--check','--status')):
            found.append(dict(pid=int(pid),elapsed=elapsed,command=cmd))
    return found


def prepare(root,source):
    inputs={}
    def record(path):
        path=Path(path);inputs[str(path)]=file_sha256(path);return path
    audit=json.loads(record(source/'audit.json').read_text());budget=json.loads(record(source/'budget_receipt.json').read_text())
    if not (audit['integrity_passed'] and audit['execution_complete'] and budget['all_steps_finished']):raise ValueError('Completed audited E124 required')
    cpus=physical_cpus()
    if len(cpus)<8:raise ValueError('Eight distinct physical cores required')
    models={}
    for arm,(fold,expected) in MODELS.items():
        folder=root/arm;folder.mkdir();original=source/arm/f'fold-{fold}'
        summary=json.loads(record(original/'summary.json').read_text());model=record(original/'selected-model.npz')
        if file_sha256(model)!=expected or summary['model_sha256']!=expected:raise ValueError('Pinned model mismatch')
        shutil.copy2(model,folder/'model.npz');record(folder/'model.npz')
        model_arrays=arrays(model);states={};responses={}
        paths=summary['prediction_paths']+[summary['quarter_path']]
        for level,path in enumerate(paths):
            path=Path(path);meta=json.loads(record(path/'metrics.json').read_text());prediction=record(path/'prediction.npz')
            if file_sha256(prediction)!=meta['sha256'][str(prediction)] or meta['level']!=level:raise ValueError('Source prediction integrity')
            source_network=record(path/'network.npz');base=arrays(source_network)
            if file_sha256(source_network)!=meta['sha256'][str(source_network)] or any(not np.array_equal(base[k],model_arrays[k]) for k in base if k!='schema_version'):
                raise ValueError('Anchor weights differ from pinned model')
            with np.load(prediction,allow_pickle=False) as z:
                start={k:z['start_'+k] for k in ('voltage_mV','gates','calcium_uM','indicator','peptide')}
                available=[t for t in TARGETS if t in meta['targets']]
                reference=z['response'][:,[meta['targets'].index(t) for t in available]]
            state_path=folder/f'start-{level}.npz';np.savez_compressed(state_path,**start);record(state_path)
            reference_path=folder/f'reference-{level}.npz';np.savez_compressed(reference_path,response=reference);record(reference_path)
            states[str(level)]=str(state_path);responses[str(level)]=dict(path=str(reference_path),targets=available,phase=meta['phase'])
        if any(len(responses[str(level)]['targets'])!=3 for level in (0,1)):raise ValueError('Full-target reproduction needs all three stimuli')
        models[arm]=dict(model_sha256=expected,source_fold=fold,starts=states,references=responses,cpus=cpus[:4] if arm=='network' else cpus[4:8])
    critical=['scripts/run_response_clock_diagnostic.py','r_elegans/brain/response_clock_probe.py',
        'r_elegans/brain/peptide_checkpoint.py','r_elegans/brain/peptide_state_replay.py',
        'r_elegans/brain/conductance_network.py','r_elegans/brain/single_compartment.py','r_elegans/brain/extrasynaptic.py']
    manifest=dict(experiment='E125',source=str(source),models=models,targets=TARGETS,observed=OBSERVED,
        cases=[dict(clock=c,start=s) for c,s in CASES],clocks=CLOCKS,input_sha256=inputs,
        source_sha256={p:file_sha256(REPO/p) for p in critical},
        common_state='Exact saved E124 quarter-clock four-phase neural+peptide state; identical arrays across all common-clock replays',
        own_state='Exact saved E124 per-clock anchor states; no fresh settling or phase reselection',
        protocol=dict(seconds=12,stimulus_seconds=.5,stimulus_nS=.03,observation_window_s=[1.5,11.5],export_dt_s=.001,reproduction_tolerance=1e-10),
        scope='Diagnostic only. Outer-clock refinement changes all outer update operations, not exclusively peptide kinetics. No weights, fitting, behavior or gate promotion.')
    write(root/'manifest.json',manifest)


def worker(root,arm):
    m=json.loads((root/'manifest.json').read_text());spec=m['models'][arm];folder=root/arm
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,spec['cpus'])
    import jax
    import jax.numpy as jnp
    from r_elegans.brain.peptide_checkpoint import load_model,load_state,save_state
    from r_elegans.brain.response_clock_probe import make_replay
    if not jax.config.x64_enabled or jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('One visible float64 GPU per worker required')
    for p,h in m['source_sha256'].items():
        if file_sha256(REPO/p)!=h:raise ValueError('Runtime source changed')
    capture_run_record(folder,REPO,dict(arm=arm),phase='E125-fixed-weight-worker')
    p,f,c,pp,exc,inh,_,model=load_model(folder/'model.npz');ids=model['neuron_ids'].tolist()
    gs=np.zeros((5,len(ids)));gs[np.arange(3),[ids.index(t) for t in TARGETS]]=.03
    kernels={};completed=[]
    for clock,kind in CASES:
        before=time.monotonic();out=folder/(kind+'-'+clock);out.mkdir()
        level=2 if kind=='common' else {'coarse':0,'joint-half':1,'joint-quarter':2}[clock]
        start=load_state(spec['starts'][str(level)]);dt,step=CLOCKS[clock]
        if clock not in kernels:kernels[clock]=make_replay(c,[ids.index(n) for n in OBSERVED],dt,step)
        write(folder/'progress.json',dict(case=out.name,completed=completed))
        means,final,trace=jax.tree.map(np.asarray,kernels[clock](p,f,pp,exc,inh,start,jnp.asarray(gs)))
        values=dict(response=means[:,:3]-means[:,-1:],means=means,voltage_mV=trace[0],calcium_uM=trace[1],indicator=trace[2],peptide=trace[3],voltage_min=trace[4],voltage_max=trace[5])
        np.savez_compressed(out/'trace.npz',**values);save_state(out/'final.npz',final)
        row=case_metrics(values,arrays(out/'final.npz'));row.update(clock=clock,start=kind,start_level=level,elapsed_seconds=time.monotonic()-before,
            start_sha256=file_sha256(spec['starts'][str(level)]),
            sha256={str(out/n):file_sha256(out/n) for n in ('trace.npz','final.npz')})
        write(out/'metrics.json',row);completed.append(out.name)
        print(json.dumps(dict(arm=arm,case=out.name,seconds=row['elapsed_seconds'],checks=row['checks'])),flush=True)
    write(folder/'summary.json',dict(completed=completed,model_sha256=spec['model_sha256']))


def case_metrics(data,final):
    finite=all(np.isfinite(v).all() for v in list(data.values())+list(final.values()))
    duplicate=float(abs(data['means'][:,-1]-data['means'][:,-2]).max())
    trace_duplicate=float(max(abs(data[k][:,-1]-data[k][:,-2]).max() for k in ('voltage_mV','calcium_uM','indicator','peptide')))
    bounded=bool(data['voltage_min'].min()>=-120 and data['voltage_max'].max()<=80)
    checks=dict(finite=finite,voltage_bounded=bounded,duplicate_sham_passed=duplicate<=1e-12 and trace_duplicate<=1e-10)
    return dict(checks=checks,duplicate_mean_error=duplicate,duplicate_trace_error=trace_duplicate,
        response_from_means_error=float(abs(data['response']-(data['means'][:,:3]-data['means'][:,-1:])).max()))


def coordinate(root,source):
    prepare(root,source);processes=[];logs=[]
    try:
        for gpu,arm in enumerate(MODELS):
            log=(root/(arm+'.log')).open('w');logs.append(log)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',JAX_ENABLE_X64='true')
            processes.append(subprocess.Popen([sys.executable,'-u','-m',MODULE,'--output-dir',str(root),'--worker',arm],env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            print(json.dumps(dict(running=[a for a,p in zip(MODELS,processes) if p.poll() is None])),flush=True);time.sleep(15)
        write(root/'worker_exit_codes.json',{a:p.returncode for a,p in zip(MODELS,processes)})
        return 0 if all(p.returncode==0 for p in processes) else 2
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()


def audit(root):
    problems=[];unavailable=[];arms={};mp=root/'manifest.json'
    if not mp.exists():
        result=dict(audit_complete=True,execution_complete=False,integrity_passed=False,problems=['Preparation incomplete'],unavailable=['All forward diagnostic evidence']);write(root/'audit.json',result);return result
    m=json.loads(mp.read_text())
    for p,h in m['input_sha256'].items():
        if not Path(p).exists() or file_sha256(p)!=h:problems.append('Input changed '+p)
    for p,h in m['source_sha256'].items():
        if file_sha256(REPO/p)!=h:problems.append('Runtime source changed '+p)
    for arm,spec in m['models'].items():
        folder=root/arm;responses={};all_checks=[];reproduction=[];pair_responses=[]
        ids=arrays(folder/'model.npz')['neuron_ids'].tolist()
        if file_sha256(folder/'model.npz')!=spec['model_sha256']:problems.append('Model changed '+arm)
        for clock,kind in CASES:
            name=kind+'-'+clock;out=folder/name
            if not (out/'metrics.json').exists():unavailable.append(arm+' '+name+' incomplete');continue
            row=json.loads((out/'metrics.json').read_text());data=arrays(out/'trace.npz');final=arrays(out/'final.npz')
            expected_level=2 if kind=='common' else {'coarse':0,'joint-half':1,'joint-quarter':2}[clock]
            if (row['clock']!=clock or row['start']!=kind or row['start_level']!=expected_level or
                row['start_sha256']!=file_sha256(spec['starts'][str(expected_level)])):problems.append('Starting-state metadata '+str(out))
            if data['response'].shape!=(4,3,302) or data['voltage_mV'].shape!=(4,5,12000,4):problems.append('Trace dimensions '+str(out))
            computed=case_metrics(data,final)
            for key,value in computed.items():
                if row[key]!=value:problems.append('Metric arithmetic '+str(out)+' '+key)
            if computed['response_from_means_error']>1e-15:problems.append('Response arithmetic '+str(out))
            for p,h in row['sha256'].items():
                if file_sha256(p)!=h:problems.append('Artifact hash '+p)
            responses[name]=data['response'];all_checks.append(all(computed['checks'].values()))
            pair_responses.append(dict(case=name,pairs=[dict(target=t,receiver=r,
                phases=data['response'][:,TARGETS.index(t),ids.index(r)].tolist(),
                phase_mean=float(data['response'][:,TARGETS.index(t),ids.index(r)].mean()))
                for t,r in [('RIVL','RMDVL'),('RMDR','RMDL')]]))
            if kind=='own':
                level={'coarse':0,'joint-half':1,'joint-quarter':2}[clock];ref=spec['references'][str(level)]
                ix=[TARGETS.index(t) for t in ref['targets']]
                error=float(abs(data['response'][:,ix]-arrays(ref['path'])['response']).max()) if ix else None
                if not ix:unavailable.append(arm+' '+clock+' original target panel contains none of the three diagnostic stimuli; no original-response reproduction for that case')
                reproduction.append(dict(clock=clock,available_targets=ref['targets'],maximum_difference=error,passed=None if error is None else error<=m['protocol']['reproduction_tolerance']))
        differences=[]
        for a,b,label in [('common-coarse','common-neural-half','neural-only'),('common-coarse','common-outer-half','outer-only'),
            ('common-coarse','common-joint-half','joint-first'),('common-joint-half','common-joint-quarter','joint-second'),
            ('common-joint-quarter','common-joint-eighth','joint-third')]+[(f'common-{c}',f'own-{c}','anchor-effect') for c in ('coarse','joint-half','joint-quarter')]:
            if a in responses and b in responses:differences.append(dict(pair=[a,b],kind=label,**response_differences(responses[a],responses[b])))
        repeated_error=(float(abs(responses['common-joint-quarter']-responses['own-joint-quarter']).max())
            if all(k in responses for k in ('common-joint-quarter','own-joint-quarter')) else None)
        arms[arm]=dict(completed_cases=len(responses),case_checks_passed=len(all_checks)==len(CASES) and all(all_checks),
            original_reproduction=reproduction,reproduction_passed=len(reproduction)==3 and all(r['passed'] for r in reproduction if r['maximum_difference'] is not None),
            same_state_same_clock_error=repeated_error,same_state_same_clock_passed=repeated_error is not None and repeated_error<=1e-12,
            differences=differences,pair_responses=pair_responses,model_sha256=spec['model_sha256'])
    for path in root.glob('**/run_records/*/source_record.json'):
        source=json.loads(path.read_text())
        for p,h in source['source_sha256'].items():
            if file_sha256(path.parent/'source'/p)!=h:problems.append('Archived source '+p)
    complete=all(r['completed_cases']==len(CASES) for r in arms.values())
    result=dict(audit_complete=True,execution_complete=complete,integrity_passed=not problems,problems=problems,arms=arms,
        diagnostic_execution_valid=complete and not problems and all(r['case_checks_passed'] and r['reproduction_passed'] and r['same_state_same_clock_passed'] for r in arms.values()),
        artifact_sha256={str(p):file_sha256(p) for p in root.rglob('*.npz')},
        unavailable=unavailable+['No new60s settling/300s dynamics, full target domain or behavior test','No model fitting, optimizer recovery or biological qualification','Outer-only changes all outer update operations, not exclusively peptide memory','Only four cells have dense state traces; all-cell outer-voltage bounds and final full state retained'],acceptance_gates_changed=False,model_promoted=False)
    write(root/'audit.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path);p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--worker',choices=tuple(MODELS));p.add_argument('--coordinate',action='store_true');p.add_argument('--report',action='store_true');p.add_argument('--check',action='store_true');p.add_argument('--status',action='store_true');a=p.parse_args()
    if a.check or a.status:
        active=jobs();print('READ-ONLY; no job started. Active jobs:',json.dumps(active));return 3 if active and a.check else 0
    if not a.output_dir:p.error('--output-dir required')
    root=a.output_dir.resolve()
    if a.report:
        for name in ('audit.json','budget_receipt.json'):
            path=root/name;value=json.loads(path.read_text()) if path.exists() else {};print(name,json.dumps({k:v for k,v in value.items() if k!='artifact_sha256'}))
        return
    if a.worker:return worker(root,a.worker)
    if a.coordinate:
        capture_run_record(root,REPO,dict(source=str(a.source)),phase='E125-execution');return coordinate(root,a.source.resolve())
    if jobs():p.error('An existing validation/fit/diagnostic is active; nothing started')
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'.cv-peptide-validation.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('Shared GPU lock held; nothing started')
        if jobs():p.error('GPU job detected after locking; nothing started')
        if root==REPO or REPO in root.parents or root in REPO.parents:p.error('External output required')
        root.mkdir(parents=True,exist_ok=False);started=time.monotonic();code=1;timed_out=False
        write(root/'budget_receipt.json',dict(status='running',total_minutes=120,all_steps_finished=False,audit_complete=False))
        try:
            if shutil.disk_usage(root).free<5*1024**3:raise ValueError('Requires5GiB free')
            from scripts.total_job_budget import run_group
            code,timed_out=run_group([sys.executable,'-u','-m',MODULE,'--output-dir',str(root),'--source',str(a.source.resolve()),'--coordinate'],110*60)
        except Exception as exc:traceback.print_exc();write(root/'failure.json',dict(error=repr(exc)))
        try:result=audit(root)
        except Exception as exc:traceback.print_exc();result=dict(audit_complete=False,execution_complete=False,integrity_passed=False,problems=[repr(exc)]);write(root/'audit.json',result)
        receipt=dict(status='finished',total_minutes=120,elapsed_seconds=time.monotonic()-started,exit_code=code,timed_out=timed_out,
            all_steps_finished=bool(code==0 and result['execution_complete'] and result['audit_complete']),audit_complete=result['audit_complete'],
            diagnostic_execution_valid=bool(result.get('diagnostic_execution_valid')),training_launched=False,model_promoted=False)
        write(root/'budget_receipt.json',receipt);print(json.dumps(receipt));return 0 if receipt['all_steps_finished'] and result['integrity_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
