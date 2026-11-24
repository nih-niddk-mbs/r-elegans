"""E126 opt-in midpoint solver and multi-cycle all-target parent validation; no fitting."""
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
from scripts import run_response_clock_diagnostic as prior
from scripts.run_cv_peptide_pilot import write,REFERENCE_SHA
from scripts.experiment_records import capture_run_record,file_sha256
from r_elegans.brain.response_clock_probe import CLOCKS

REPO=Path(__file__).resolve().parents[1];BASE=prior.BASE;SOURCE=prior.SOURCE
MODULE='scripts.run_midpoint_validation'
BENCH=[('legacy-own','legacy',0,0),('legacy-coarse','legacy',0,2),('legacy-reference','legacy',3,2)]+[(f'midpoint-{i}','midpoint',i,2) for i in range(3)]


def clock(level):return (.001/2**level,.03125/2**level)


def jobs():
    from scripts.cv_validation_job import ancestor_pids
    found=prior.jobs();exclude=ancestor_pids(os.getpid())
    for line in subprocess.check_output(['ps','-ww','-u',str(os.getuid()),'-o','pid=,etime=,args='],text=True).splitlines():
        fields=line.split(None,2)
        if len(fields)!=3:continue
        pid,elapsed,cmd=fields
        if int(pid) in exclude:continue
        try:a=shlex.split(cmd)
        except ValueError:continue
        if any(a[i:i+2]==['-m',MODULE] for i in range(len(a)-1)) and not any(x in a for x in ('--status','--report','--check')):found.append(dict(pid=int(pid),elapsed=elapsed,command=cmd))
    return found


def check_response(a,b):
    error=float(abs(a-b).max());tolerance=float(1e-7+.1*abs(b).max())
    return dict(maximum_difference=error,tolerance=tolerance,passed=bool(np.isfinite(error) and error<=tolerance))


def cycle_sampling(response):
    if response.ndim!=3 or response.shape[0]!=16:raise ValueError('Four cycles x four phases required')
    cycles=response.reshape(4,4,*response.shape[1:]).mean(1);mean=cycles.mean(0)
    error=float(max(abs(np.delete(cycles,i,0).mean(0)-mean).max() for i in range(4)))
    tolerance=float(1e-7+.1*abs(mean).max())
    return dict(leave_one_cycle_out_max_difference=error,tolerance=tolerance,passed=error<=tolerance,
        scope='Prospective four-cycle sampling stability screen, not confidence interval or biological test')


def choose_clock(data,checks):
    reference=data['legacy-reference']['response'];legacy_error=float(abs(data['legacy-coarse']['response']-reference).max())
    comparisons=[];selected=None
    for level in (0,1):
        a=data[f'midpoint-{level}']['response'];b=data[f'midpoint-{level+1}']['response']
        cross=check_response(a,reference);adjacent=check_response(a,b)
        improves=float(abs(a-reference).max())<legacy_error
        safe=all(all(checks[n]['checks'].values()) for n in (f'midpoint-{level}',f'midpoint-{level+1}','legacy-reference','legacy-coarse','legacy-own'))
        passed=bool(cross['passed'] and adjacent['passed'] and improves and safe)
        comparisons.append(dict(level=level,reference=cross,adjacent=adjacent,improves_over_legacy=improves,safe=safe,passed=passed))
        if selected is None and passed:selected=level
    return selected,comparisons


def prepare(root,source):
    prior.prepare(root,source);m=json.loads((root/'manifest.json').read_text())
    original=source/'parent.npz'
    if file_sha256(original)!=REFERENCE_SHA:raise ValueError('Clean wired parent mismatch')
    parent=prior.arrays(original);pp=prior.arrays(source/'peptide/fold-1/selected-model.npz')
    if file_sha256(source/'peptide/fold-1/selected-model.npz')!=prior.MODELS['peptide'][1]:raise ValueError('Peptide parent changed')
    for k in parent:
        if k!='schema_version' and not np.array_equal(parent[k],pp[k]):raise ValueError('Parent wired constants differ')
    targets=json.loads((source/'manifest.json').read_text())['targets']
    m['input_sha256'].update({str(p):file_sha256(p) for p in (original,source/'manifest.json',source/'peptide/fold-1/selected-model.npz')})
    for arm in m['models']:
        values={k:v.copy() for k,v in pp.items()};values['extrasynaptic_enabled']=np.array(arm=='peptide')
        path=root/arm/'parent-model.npz';np.savez_compressed(path,**values)
        m['models'][arm]['parent_sha256']=file_sha256(path);m['input_sha256'][str(path)]=file_sha256(path)
    m.update(experiment='E126',full_targets=targets,policy=dict(cycles=4,phases_per_cycle=4,warmup_seconds=60,record_seconds=4,batch_targets=4,
        benchmark='Coarsest level0/1 passing same-state adjacent/reference response bounds and improving error versus legacy; numerical selection only.',
        parent='Clean pre-atlas wired parent with fixed peptide prior enabled/disabled; fresh own-state settling per selected clock.',
        numeric='Phase-mean full-target response error <=1e-7+0.1max_abs_fine; each level cycle jackknife <=1e-7+0.1max_abs_mean.',
        scope='New prospective screen; does not reclassify legacy E124 gates or qualify behavior. Neural midpoint only; held outer peptide input remains first order.'))
    for name in ('scripts/run_midpoint_validation.py','r_elegans/brain/midpoint_network.py','r_elegans/brain/midpoint_replay.py'):
        m['source_sha256'][name]=file_sha256(REPO/name)
    write(root/'manifest.json',m)


def full_metrics(data,final):
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    finite=all(np.isfinite(a).all() for a in list(data.values())+list(final.values()))
    controls=data['controls'];sham=float(abs(controls-controls[0,:,0][None,:,None,:]).max())
    v=data['sham_voltage'];voltage_sham=float(abs(v[:,0]-v[:,1]).max())
    rhythms=[rhythm_metrics(v[i,0,:,j],.001) for i in range(16) for j in range(2)]
    bounded=bool(data['voltage_min'].min()>=-120 and data['voltage_max'].max()<=80)
    return dict(finite=finite,duplicate_mean_error=sham,duplicate_voltage_error=voltage_sham,rhythms=rhythms,
        bounded=bounded,passed=bool(finite and bounded and sham<=1e-12 and voltage_sham<=1e-10 and all(r['passed'] for r in rhythms)))


def worker(root,arm):
    m=json.loads((root/'manifest.json').read_text());spec=m['models'][arm];folder=root/arm
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,spec['cpus'])
    import jax
    import jax.numpy as jnp
    from r_elegans.brain.peptide_checkpoint import load_model,load_state,save_state
    from r_elegans.brain.midpoint_replay import make_replay,make_anchors
    if not jax.config.x64_enabled or jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('One visible float64 GPU required')
    for p,h in m['source_sha256'].items():
        if file_sha256(REPO/p)!=h:raise ValueError('Source changed')
    capture_run_record(folder,REPO,dict(arm=arm),phase='E126-midpoint-worker')
    p,f,c,pp,exc,inh,_,model=load_model(folder/'model.npz');ids=model['neuron_ids'].tolist()
    gs=np.zeros((5,302));gs[np.arange(3),[ids.index(t) for t in prior.TARGETS]]=.03
    data={};rows={}
    for name,method,level,start_level in BENCH:
        path=folder/'benchmark'/name;path.mkdir(parents=True);write(folder/'progress.json',dict(stage='benchmark',case=name));tick=time.monotonic()
        start=load_state(spec['starts'][str(start_level)]);dt,step=clock(level)
        means,final,trace=jax.tree.map(np.asarray,make_replay(c,[ids.index(n) for n in prior.OBSERVED],dt,step,method=method)(p,f,pp,exc,inh,start,jnp.asarray(gs)))
        values=dict(response=means[:,:3]-means[:,-1:],means=means,voltage_mV=trace[0],calcium_uM=trace[1],indicator=trace[2],peptide=trace[3],voltage_min=trace[4],voltage_max=trace[5])
        np.savez_compressed(path/'trace.npz',**values);save_state(path/'final.npz',final)
        row=prior.case_metrics(values,prior.arrays(path/'final.npz'));row.update(seconds=time.monotonic()-tick)
        row['sha256']={str(path/n):file_sha256(path/n) for n in ('trace.npz','final.npz')}
        write(path/'metrics.json',row);data[name]=values;rows[name]=row
        print(json.dumps(dict(arm=arm,stage='benchmark',case=name,seconds=row['seconds'])),flush=True)
    original=prior.arrays(spec['references']['0']['path'])['response']
    reproduction=float(abs(data['legacy-own']['response']-original).max())
    selected,comparisons=choose_clock(data,rows)
    if not np.isfinite(reproduction) or reproduction>1e-10:selected=None
    result=dict(benchmark_reproduction_error=reproduction,selected_level=selected,benchmark_comparisons=comparisons,full_target_complete=False,parent_validation_passed=False)
    write(folder/'summary.json',result)
    if selected is None:
        result['stop_reason']='benchmark_gate_not_met; no all-target run';write(folder/'summary.json',result);return
    p,f,c,pp,exc,inh,_,parent=load_model(folder/'parent-model.npz')
    for level in (selected,selected+1):
        dt,step=clock(level);out=folder/f'full-{level}';out.mkdir();tick=time.monotonic()
        write(folder/'progress.json',dict(stage='own_state_multicycle',level=level))
        start,phases=make_anchors(c,ids.index('RMDL'),dt,step)(p,f,pp,exc,inh);save_state(out/'starts.npz',start);write(out/'phases.json',phases)
        kernel=make_replay(c,[ids.index('RMDL'),ids.index('RMDR')],dt,step,trace_mode='shams')
        responses=[];controls=[];lo=[];hi=[];finals={};sham_voltage=None
        for begin in range(0,len(m['full_targets']),4):
            targets=m['full_targets'][begin:begin+4];stim=np.zeros((6,302));stim[np.arange(len(targets)),[ids.index(t) for t in targets]]=.03
            write(folder/'progress.json',dict(stage='all_targets',level=level,targets_done=begin,targets_total=len(m['full_targets'])))
            means,end,traces=jax.tree.map(np.asarray,kernel(p,f,pp,exc,inh,start,jnp.asarray(stim)))
            responses.append(means[:,:len(targets)]-means[:,-1:]);controls.append(means[:,-2:]);lo.append(traces[4].min(axis=-1));hi.append(traces[5].max(axis=-1))
            if sham_voltage is None:sham_voltage=traces[0]
            for k,v in dict(end.neural._asdict(),peptide=end.peptide).items():finals.setdefault(k,[]).append(v)
        values=dict(response=np.concatenate(responses,axis=1),controls=np.stack(controls),sham_voltage=sham_voltage,voltage_min=np.stack(lo),voltage_max=np.stack(hi))
        final={k:np.stack(v) for k,v in finals.items()};np.savez_compressed(out/'response.npz',**values);np.savez_compressed(out/'final.npz',**final)
        row=dict(metrics=full_metrics(values,final),sampling=cycle_sampling(values['response']),seconds=time.monotonic()-tick)
        row['sha256']={str(out/n):file_sha256(out/n) for n in ('response.npz','final.npz','starts.npz','phases.json')}
        write(out/'metrics.json',row)
    low=prior.arrays(folder/f'full-{selected}'/'response.npz')['response'];high=prior.arrays(folder/f'full-{selected+1}'/'response.npz')['response']
    convergence=check_response(low.mean(0),high.mean(0))
    full_rows=[json.loads((folder/f'full-{i}'/'metrics.json').read_text()) for i in (selected,selected+1)]
    result.update(full_target_complete=True,full_target_response_check=convergence,parent_validation_passed=bool(convergence['passed'] and all(r['metrics']['passed'] and r['sampling']['passed'] for r in full_rows)),stop_reason='planned_checks_finished')
    write(folder/'summary.json',result)


def coordinate(root,source):
    prepare(root,source);processes=[];logs=[]
    try:
        for gpu,arm in enumerate(prior.MODELS):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',JAX_ENABLE_X64='true')
            log=(root/(arm+'.log')).open('w');logs.append(log)
            processes.append(subprocess.Popen([sys.executable,'-u','-m',MODULE,'--output-dir',str(root),'--worker',arm],env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            progress={}
            for arm in prior.MODELS:
                path=root/arm/'progress.json';progress[arm]=json.loads(path.read_text()) if path.exists() else 'starting'
            print(json.dumps(progress),flush=True);time.sleep(15)
        write(root/'worker_exit_codes.json',{a:p.returncode for a,p in zip(prior.MODELS,processes)})
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
        result=dict(audit_complete=True,execution_complete=False,integrity_passed=False,problems=['Preparation incomplete']);write(root/'audit.json',result);return result
    m=json.loads(mp.read_text())
    if m['full_targets']!=json.loads((Path(m['source'])/'manifest.json').read_text())['targets']:problems.append('Target domain changed')
    for path,h in m['input_sha256'].items():
        if not Path(path).exists() or file_sha256(path)!=h:problems.append('Input hash '+path)
    for path,h in m['source_sha256'].items():
        if file_sha256(REPO/path)!=h:problems.append('Source hash '+path)
    for arm,spec in m['models'].items():
        folder=root/arm;data={};rows={}
        failure=folder/'failure.json'
        if failure.exists():unavailable.append(arm+' worker failure: '+json.loads(failure.read_text())['error'])
        for file,key in (('model.npz','model_sha256'),('parent-model.npz','parent_sha256')):
            if file_sha256(folder/file)!=spec[key]:problems.append('Model hash '+arm+' '+file)
        for name,method,level,start_level in BENCH:
            path=folder/'benchmark'/name
            if not (path/'metrics.json').exists():unavailable.append(arm+' incomplete '+name);continue
            data[name]=prior.arrays(path/'trace.npz');rows[name]=json.loads((path/'metrics.json').read_text())
            expected=prior.case_metrics(data[name],prior.arrays(path/'final.npz'))
            for p,h in rows[name]['sha256'].items():
                if file_sha256(p)!=h:problems.append('Benchmark artifact hash '+p)
            for key,v in expected.items():
                if rows[name][key]!=v:problems.append('Benchmark arithmetic '+str(path))
        summary_path=folder/'summary.json'
        if not summary_path.exists() or len(data)!=len(BENCH):arms[arm]=dict(completed=False);continue
        s=json.loads(summary_path.read_text());reproduction=float(abs(data['legacy-own']['response']-prior.arrays(spec['references']['0']['path'])['response']).max())
        selected,comparisons=choose_clock(data,rows)
        if not np.isfinite(reproduction) or reproduction>1e-10:selected=None
        if selected!=s['selected_level'] or comparisons!=s['benchmark_comparisons'] or reproduction!=s['benchmark_reproduction_error']:problems.append('Benchmark gate/selection '+arm)
        passed=False;convergence=None;sampling=[];complete=selected is None;level_checks=[]
        if selected is not None:
            responses=[];checks=[]
            for level in (selected,selected+1):
                path=folder/f'full-{level}'
                if not (path/'metrics.json').exists():unavailable.append(arm+' missing full-target '+str(level));continue
                values=prior.arrays(path/'response.npz');final=prior.arrays(path/'final.npz');row=json.loads((path/'metrics.json').read_text())
                metrics=full_metrics(values,final);sample=cycle_sampling(values['response'])
                level_checks.append(dict(level=level,finite=metrics['finite'],bounded=metrics['bounded'],
                    duplicate_mean_error=metrics['duplicate_mean_error'],duplicate_voltage_error=metrics['duplicate_voltage_error'],
                    failed_rhythms=[dict(index=i,metrics=r) for i,r in enumerate(metrics['rhythms']) if not r['passed']],passed=metrics['passed']))
                for p,h in row['sha256'].items():
                    if file_sha256(p)!=h:problems.append('Full artifact hash '+p)
                if metrics!=row['metrics'] or sample!=row['sampling']:problems.append('Full-target arithmetic '+str(path))
                starts=prior.arrays(path/'starts.npz');phases=json.loads((path/'phases.json').read_text())
                from r_elegans.brain.midpoint_replay import cycle_phases
                indices,metadata=cycle_phases(phases['reference_voltage_mV'],clock(level)[0])
                if metadata!={k:v for k,v in phases.items() if k!='reference_voltage_mV'}:problems.append('Cycle selection arithmetic '+str(path))
                ids=prior.arrays(folder/'parent-model.npz')['neuron_ids'].tolist()
                if (values['response'].shape!=(16,len(m['full_targets']),302) or
                    not np.array_equal(starts['voltage_mV'][:,ids.index('RMDL')],np.asarray(phases['reference_voltage_mV'])[indices])):problems.append('Phase/domain correspondence '+str(path))
                if starts['voltage_mV'].shape!=(16,302) or len(phases['indices'])!=16:problems.append('Phase/state dimensions '+str(path))
                responses.append(values['response']);checks.append(metrics['passed']);sampling.append(sample)
            complete=len(responses)==2
            if complete:
                convergence=check_response(responses[0].mean(0),responses[1].mean(0));passed=bool(convergence['passed'] and all(checks) and all(r['passed'] for r in sampling))
                if convergence!=s['full_target_response_check']:problems.append('Response convergence arithmetic '+arm)
        if passed!=s['parent_validation_passed']:problems.append('Parent qualification arithmetic '+arm)
        arms[arm]=dict(completed=complete,selected_level=selected,benchmark_comparisons=comparisons,legacy_reproduction_error=reproduction,
            full_target_response_check=convergence,cycle_sampling=sampling,parent_validation_passed=passed,
            level_checks=level_checks,
            benchmark_gate_blocked=selected is None)
    for record in root.glob('**/run_records/*/source_record.json'):
        source=json.loads(record.read_text())
        for name,h in source['source_sha256'].items():
            if file_sha256(record.parent/'source'/name)!=h:problems.append('Archived source '+name)
    result=dict(audit_complete=True,execution_complete=len(arms)==2 and all(r['completed'] for r in arms.values()),integrity_passed=not problems,
        problems=problems,arms=arms,parent_validation_passed=not problems and len(arms)==2 and all(r.get('parent_validation_passed') for r in arms.values()),
        artifact_sha256={str(p):file_sha256(p) for p in root.rglob('*.npz')},
        unavailable=unavailable+['No full-target third clock or300s embodiment','Four-cycle jackknife is a deterministic stability screen, not a confidence interval','Neural midpoint is second-order locally; held outer peptide coupling remains first-order','No fitting, optimizer recovery, sensory start/stop, swimming or biological qualification'],legacy_gates_changed=False,model_promoted=False)
    write(root/'audit.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path);p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--worker',choices=tuple(prior.MODELS));p.add_argument('--coordinate',action='store_true');p.add_argument('--status',action='store_true');p.add_argument('--check',action='store_true');p.add_argument('--report',action='store_true');a=p.parse_args()
    if a.status or a.check:
        active=jobs();print('READ-ONLY; no work started. Active jobs:',json.dumps(active));return 3 if active and a.check else 0
    if not a.output_dir:p.error('--output-dir required')
    root=a.output_dir.resolve()
    if a.report:
        for name in ('audit.json','budget_receipt.json'):
            path=root/name;r=json.loads(path.read_text()) if path.exists() else {};print(name,json.dumps({k:v for k,v in r.items() if k!='artifact_sha256'}))
        return
    if a.worker:
        try:return worker(root,a.worker)
        except Exception as exc:
            write(root/a.worker/'failure.json',dict(error=repr(exc)));raise
    if a.coordinate:
        capture_run_record(root,REPO,dict(source=str(a.source)),phase='E126-fixed-weight');return coordinate(root,a.source.resolve())
    if jobs():p.error('Existing GPU work active; nothing started')
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'.cv-peptide-validation.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('Shared GPU lock held; nothing started')
        if jobs():p.error('GPU work detected after lock; nothing started')
        if root==REPO or REPO in root.parents or root in REPO.parents:p.error('External output required')
        root.mkdir(parents=True,exist_ok=False);started=time.monotonic();code=1;timed_out=False
        write(root/'budget_receipt.json',dict(status='running',total_minutes=240,all_steps_finished=False,audit_complete=False))
        try:
            if shutil.disk_usage(root).free<10*1024**3:raise ValueError('Requires10GiB free')
            from scripts.total_job_budget import run_group
            code,timed_out=run_group([sys.executable,'-u','-m',MODULE,'--output-dir',str(root),'--source',str(a.source.resolve()),'--coordinate'],225*60)
        except Exception as exc:traceback.print_exc();write(root/'failure.json',dict(error=repr(exc)))
        try:result=audit(root)
        except Exception as exc:traceback.print_exc();result=dict(audit_complete=False,execution_complete=False,integrity_passed=False,problems=[repr(exc)]);write(root/'audit.json',result)
        receipt=dict(status='finished',total_minutes=240,elapsed_seconds=time.monotonic()-started,exit_code=code,timed_out=timed_out,
            all_steps_finished=bool(code==0 and result['execution_complete'] and result['audit_complete']),audit_complete=result['audit_complete'],
            parent_validation_passed=bool(result.get('parent_validation_passed')),training_launched=False,model_promoted=False)
        write(root/'budget_receipt.json',receipt);print(json.dumps(receipt));return 0 if receipt['all_steps_finished'] and result['integrity_passed'] else 2


if __name__=='__main__':main()
