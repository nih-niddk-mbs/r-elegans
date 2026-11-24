"""E127 three-hour additional cycle sampling; checkpointed stages, validation finishes."""
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
from scripts import continue_midpoint_validation as resume
from scripts.experiment_records import capture_run_record,file_sha256
from r_elegans.brain.cycle_ensemble import ensemble_metrics,extension_phases

REPO=resume.REPO;BASE=resume.BASE;MODULE='scripts.run_cycle_ensemble'
SOURCE=BASE/'midpoint-continuation-20261124T210840Z-1129532'
STAGES=(8,16,32)


def jobs():
    from scripts.cv_validation_job import ancestor_pids
    found=resume.active_jobs();excluded=ancestor_pids(os.getpid())
    for line in subprocess.check_output(['ps','-ww','-u',str(os.getuid()),'-o','pid=,etime=,args='],text=True).splitlines():
        row=line.split(None,2)
        if len(row)!=3:continue
        pid,elapsed,cmd=row
        if int(pid) in excluded:continue
        try:a=shlex.split(cmd)
        except ValueError:continue
        if any(a[i:i+2]==['-m',MODULE] for i in range(len(a)-1)) and not any(x in a for x in ('--status','--report','--check','--select-output')):found.append(dict(pid=int(pid),elapsed=elapsed,command=cmd))
    return found


def select_output(source):
    found=[]
    for name in ('manifest.json','job.json'):
        for p in BASE.glob('cycle-ensemble-*/'+name):
            if json.loads(p.read_text()).get('source_run')==str(source.resolve()):found.append(p.parent)
    return max(found) if found else BASE/('cycle-ensemble-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(os.getpid()))


def affected_targets(response,targets,tolerance):
    c=response.reshape(4,4,len(targets),302).mean(1);mean=c.mean(0)
    error=np.stack([abs(np.delete(c,i,0).mean(0)-mean) for i in range(4)]).max(0)
    return [t for t,row in zip(targets,error) if (row>tolerance).any()]


def prepare(root,source):
    source=source.resolve();audit=json.loads((source/'audit.json').read_text());m=json.loads((source/'manifest.json').read_text())
    if not all(audit.get(k) for k in ('audit_complete','execution_complete','integrity_passed')):raise ValueError('Complete integrity-passed source audit required')
    if any(m['policy'].get(k)!=v for k,v in dict(cycles=4,phases_per_cycle=4,warmup_seconds=60).items()):raise ValueError('Expected declared four-cycle/60s source protocol')
    inputs={str(p):file_sha256(p) for p in (source/'audit.json',source/'manifest.json',source/'budget_receipt.json')};panel=set();cases=[]
    originals={}
    for arm in ('network','peptide'):
        if not audit['arms'][arm]['full_target_response_check']['passed'] or not all(r['passed'] for r in audit['arms'][arm]['level_checks']):raise ValueError('Source timestep/control checks must pass before expanded sampling')
        if audit['arms'][arm]['selected_level']!=0:raise ValueError('Expected source levels0/1')
        for level in (0,1):
            folder=source/arm/f'full-{level}';row=json.loads((folder/'metrics.json').read_text());path=folder/'response.npz'
            if file_sha256(path)!=audit['artifact_sha256'][str(path)]:raise ValueError('Source response hash')
            inputs[str(path)]=file_sha256(path);inputs[str(folder/'metrics.json')]=file_sha256(folder/'metrics.json')
            y=resume.old.prior.arrays(path)['response'];originals[(arm,level)]=y
            panel.update(affected_targets(y,m['full_targets'],row['sampling']['tolerance']))
    panel=sorted(panel)
    if not panel:raise ValueError('No affected source targets; no new sampling justified')
    for arm in ('network','peptide'):
        src=source/arm/'parent-model.npz';dest=root/arm/'model.npz';dest.parent.mkdir(parents=True,exist_ok=True)
        if file_sha256(src)!=m['models'][arm]['parent_sha256']:raise ValueError('Parent identity')
        shutil.copy2(src,dest);inputs[str(src)]=file_sha256(src);inputs[str(dest)]=file_sha256(dest)
        for level in (0,1):
            folder=root/arm/f'level-{level}';folder.mkdir(exist_ok=True);old=source/arm/f'full-{level}'
            for name in ('starts.npz','phases.json','metrics.json'):
                src=old/name;out=folder/('base-'+name);shutil.copy2(src,out);inputs[str(src)]=file_sha256(src);inputs[str(out)]=file_sha256(out)
                if src.suffix=='.npz' and inputs[str(src)]!=audit['artifact_sha256'][str(src)]:raise ValueError('Saved state hash')
            ix=[m['full_targets'].index(t) for t in panel];path=folder/'base-response.npz'
            resume.atomic_npz(path,response=originals[(arm,level)][:,ix]);inputs[str(path)]=file_sha256(path)
            old_meta=json.loads((folder/'base-phases.json').read_text());dt=resume.old.clock(level)[0]
            times=(60.+(np.asarray(old_meta['indices']).reshape(4,4)+1)*dt).mean(1).tolist()
            coarse_seconds=json.loads((source/arm/'full-0'/'metrics.json').read_text()).get('seconds',120.*((len(m['full_targets'])+3)//4))
            level_seconds=json.loads((folder/'base-metrics.json').read_text()).get('seconds',coarse_seconds*2**level)
            cases.append(dict(arm=arm,level=level,base_cycle_times=times,extension_start_s=60.+(old_meta['indices'][-1]+1)*dt,
                tolerance=json.loads((folder/'base-metrics.json').read_text())['sampling']['tolerance'],
                baseline_batch_seconds=level_seconds/((len(m['full_targets'])+3)//4),
                starts_sha256=file_sha256(folder/'base-starts.npz'),model_sha256=m['models'][arm]['parent_sha256']))
    critical=list(m['source_sha256'])+['scripts/run_cycle_ensemble.py','r_elegans/brain/cycle_ensemble.py']
    for p,h in m['source_sha256'].items():
        if file_sha256(REPO/p)!=h:raise ValueError('Original numerical source changed '+p)
    manifest=dict(experiment='E127',source_run=str(source),panel=panel,cases=cases,input_sha256=inputs,
        source_sha256={p:file_sha256(REPO/p) for p in critical},worker_cpus=[m['models'][arm]['cpus'] for arm in ('network','peptide')],
        numeric_tolerance={arm:audit['arms'][arm]['full_target_response_check']['tolerance'] for arm in ('network','peptide')},
        stages=STAGES,extension_seconds=32.,policy='Fixed source tolerances; nested4old+4/12/28new cycles, four phases each. Temporal/prefix/leave-out stability plus two-clock panel checks. No full-target qualification or fitting.')
    resume.atomic_json(root/'manifest.json',manifest)


def folder_for(root,case):return root/case['arm']/f"level-{case['level']}"


def tasks_for(m,cycles):
    tasks=[]
    for case in m['cases']:
        for group in range((cycles-4)//4):
            for begin in range(0,len(m['panel']),4):
                tasks.append(dict(arm=case['arm'],level=case['level'],group=group,begin=begin,targets=m['panel'][begin:begin+4],model_sha256=case['model_sha256']))
    return tasks


def task_path(root,task):return root/task['arm']/f"level-{task['level']}"/'batches'/f"group-{task['group']:02d}-target-{task['begin']:03d}.npz"


def read_task(root,task):
    path=task_path(root,task);a=resume.old.prior.arrays(path)
    if json.loads(str(a.pop('provenance_json')))!=task:raise ValueError('Task provenance '+str(path))
    shapes={'means':(16,6,302),'voltage_min':(16,6),'voltage_max':(16,6),'sham_voltage':(16,2,12000,2) if task['begin']==0 else (0,),
        'final_voltage_mV':(16,6,302),'final_gates':(16,6,302,35),'final_calcium_uM':(16,6,302),'final_indicator':(16,6,302),'final_peptide':(16,6,302)}
    if set(a)!=set(shapes) or any(a[k].shape!=shape for k,shape in shapes.items()) or not all(np.isfinite(v).all() for v in a.values()):raise ValueError('Invalid batch '+str(path))
    side=path.with_suffix('.json')
    if side.exists() and json.loads(side.read_text())['sha256']!=file_sha256(path):raise ValueError('Batch hash '+str(path))
    return a


def materialize_extension(folder,case):
    saved=resume.old.prior.arrays(folder/'extension.npz');metadata=json.loads(str(saved.pop('metadata_json')))
    if metadata['model_sha256']!=case['model_sha256'] or metadata['base_starts_sha256']!=case['starts_sha256']:raise ValueError('Extension provenance mismatch')
    states={k:v for k,v in saved.items() if not k.startswith('end_')};end={k[4:]:v for k,v in saved.items() if k.startswith('end_')}
    if states['voltage_mV'].shape!=(112,302) or not all(np.isfinite(a).all() for a in saved.values()):raise ValueError('Invalid extension checkpoint')
    for name,values in (('extension-states.npz',states),('extension-end.npz',end)):
        path=folder/name
        if path.exists():
            existing=resume.old.prior.arrays(path)
            if set(existing)!=set(values) or any(not np.array_equal(existing[k],v) for k,v in values.items()):raise ValueError('Materialized extension changed')
        else:resume.atomic_npz(path,**values)
    metadata['sha256']={str(folder/n):file_sha256(folder/n) for n in ('extension.npz','extension-states.npz','extension-end.npz')}
    resume.atomic_json(folder/'extension.json',metadata)


def extend_case(root,case):
    import jax
    from r_elegans.brain.peptide_checkpoint import load_model,load_state
    from r_elegans.brain.cycle_ensemble import make_extension
    folder=folder_for(root,case)
    if (folder/'extension.json').exists():
        row=json.loads((folder/'extension.json').read_text())
        for p,h in row['sha256'].items():
            if file_sha256(p)!=h:raise ValueError('Extension checkpoint changed')
        return
    if (folder/'extension.npz').exists():materialize_extension(folder,case);return
    model=root/case['arm']/'model.npz';p,f,c,pp,exc,inh,_,a=load_model(model)
    start=load_state(folder/'base-starts.npz');initial=jax.tree.map(lambda a:a[-1],start)
    dt,step=resume.old.clock(case['level']);ids=a['neuron_ids'].tolist()
    states,end,metadata=make_extension(c,ids.index('RMDL'),dt,step)(p,f,pp,exc,inh,initial)
    if not metadata['finite'] or metadata['voltage_min_mV']< -120 or metadata['voltage_max_mV']>80:raise ValueError('Extension dynamics failed; no response work admitted')
    values=jax.tree.map(np.asarray,states);last=jax.tree.map(np.asarray,end)
    metadata.update(model_sha256=case['model_sha256'],base_starts_sha256=case['starts_sha256'])
    resume.atomic_npz(folder/'extension.npz',metadata_json=np.array(json.dumps(metadata)),**values.neural._asdict(),peptide=values.peptide,
        **{'end_'+k:v for k,v in dict(last.neural._asdict(),peptide=last.peptide).items()})
    materialize_extension(folder,case)


def worker(root,slot,stage):
    import jax
    import jax.numpy as jnp
    from r_elegans.brain.peptide_checkpoint import load_model,load_state
    from r_elegans.brain.midpoint_replay import make_replay
    m=json.loads((root/'manifest.json').read_text())
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,m['worker_cpus'][slot])
    if not jax.config.x64_enabled or jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('One visible float64 GPU required')
    for p,h in m['source_sha256'].items():
        if file_sha256(REPO/p)!=h:raise ValueError('Runtime source changed')
    for case in m['cases']:
        if file_sha256(root/case['arm']/'model.npz')!=case['model_sha256'] or file_sha256(folder_for(root,case)/'base-starts.npz')!=case['starts_sha256']:raise ValueError('Model/starting states changed')
    capture_run_record(root/f'worker-{slot}',REPO,dict(stage=stage),phase='E127-cycle-sampling')
    if stage==0:
        for case in m['cases'][slot::2]:
            resume.atomic_json(root/f'progress-{slot}.json',dict(stage='extending_saved_trajectory',**case));extend_case(root,case)
        return
    cache={};kernels={}
    for task in tasks_for(m,stage)[slot::2]:
        path=task_path(root,task)
        if path.exists():read_task(root,task);continue
        folder=root/task['arm']/f"level-{task['level']}";key=(task['arm'],task['level'])
        if key not in cache:
            model=root/task['arm']/'model.npz'
            if file_sha256(model)!=task['model_sha256']:raise ValueError('Model changed')
            p,f,c,pp,exc,inh,_,a=load_model(model);ids=a['neuron_ids'].tolist();states=load_state(folder/'extension-states.npz')
            cache[key]=(p,f,pp,exc,inh,ids,states)
            if task['level'] not in kernels:
                dt,step=resume.old.clock(task['level']);kernels[task['level']]=make_replay(c,[ids.index('RMDL'),ids.index('RMDR')],dt,step,trace_mode='shams')
        p,f,pp,exc,inh,ids,states=cache[key];g=task['group'];start=jax.tree.map(lambda a:a[g*16:(g+1)*16],states)
        stimuli=np.zeros((6,302));stimuli[np.arange(len(task['targets'])),[ids.index(t) for t in task['targets']]]=.03
        resume.atomic_json(root/f'progress-{slot}.json',dict(stage='sampling_batch',cycles=stage,**task));tick=time.monotonic()
        means,end,traces=jax.tree.map(np.asarray,kernels[task['level']](p,f,pp,exc,inh,start,jnp.asarray(stimuli)))
        data=dict(means=means,voltage_min=traces[4].min(axis=-1),voltage_max=traces[5].max(axis=-1),sham_voltage=traces[0] if task['begin']==0 else np.empty(0),**{'final_'+k:v for k,v in dict(end.neural._asdict(),peptide=end.peptide).items()})
        resume.atomic_json(root/f'progress-{slot}.json',dict(stage='checkpointing_batch',cycles=stage,**task))
        resume.atomic_npz(path,provenance_json=np.array(json.dumps(task,sort_keys=True)),**data);read_task(root,task)
        resume.atomic_json(path.with_suffix('.json'),dict(sha256=file_sha256(path),seconds=time.monotonic()-tick))
        print(json.dumps(dict(event='batch_saved_verified',cycles=stage,seconds=time.monotonic()-tick,**task)),flush=True)
    resume.atomic_json(root/f'progress-{slot}.json',dict(stage='assigned_batches_complete',cycles=stage))


def evaluate(root,m,cycles):
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    cases=[];responses={};all_safe=True
    for case in m['cases']:
        folder=folder_for(root,case);parts=[resume.old.prior.arrays(folder/'base-response.npz')['response']];safe=True;control_errors=[];voltage_errors=[];failed_rhythms=[];lo=[];hi=[]
        for group in range((cycles-4)//4):
            values=[];shams=[]
            for task in [t for t in tasks_for(m,cycles) if t['arm']==case['arm'] and t['level']==case['level'] and t['group']==group]:
                a=read_task(root,task);means=a['means'];values.append(means[:,:len(task['targets'])]-means[:,-1:]);shams.append(means[:,-2:])
                lo.append(float(a['voltage_min'].min()));hi.append(float(a['voltage_max'].max()))
                safe=bool(safe and a['voltage_min'].min()>=-120 and a['voltage_max'].max()<=80)
                if task['begin']==0:
                    v=a['sham_voltage'];ve=float(abs(v[:,0]-v[:,1]).max());voltage_errors.append(ve)
                    rhythms=[dict(group=group,phase=i,neuron=('RMDL','RMDR')[j],metrics=rhythm_metrics(v[i,0,:,j],.001)) for i in range(16) for j in range(2)]
                    failed_rhythms.extend(r for r in rhythms if not r['metrics']['passed']);safe=bool(safe and ve<=1e-10 and all(r['metrics']['passed'] for r in rhythms))
            controls=np.stack(shams);error=float(abs(controls-controls[0,:,0][None,:,None,:]).max());control_errors.append(error);safe=bool(safe and error<=1e-12)
            parts.append(np.concatenate(values,axis=1))
        y=np.concatenate(parts,axis=0);responses[(case['arm'],case['level'])]=y
        metadata=json.loads((folder/'extension.json').read_text());n=cycles-4
        times=case['base_cycle_times']+[case['extension_start_s']+t for t in metadata['cycle_mean_time_s'][:n]]
        previous=y[:cycles//2*4]
        stats=ensemble_metrics(y,times,previous,case['tolerance'])
        cases.append(dict(arm=case['arm'],level=case['level'],sampling=stats,controls_and_dynamics_passed=safe,duplicate_mean_errors=control_errors,
            duplicate_voltage_errors=voltage_errors,voltage_min_mV=min(lo),voltage_max_mV=max(hi),failed_rhythms=failed_rhythms,final_states_finite=True));all_safe=all_safe and safe
    numeric=[]
    for arm in ('network','peptide'):
        error=float(abs(responses[(arm,0)].mean(0)-responses[(arm,1)].mean(0)).max());tol=m['numeric_tolerance'][arm]
        numeric.append(dict(arm=arm,maximum_difference=error,tolerance=tol,passed=error<=tol))
    return dict(cycles=cycles,cases=cases,numerical=numeric,panel_qualified=bool(all_safe and all(r['sampling']['passed'] for r in cases) and all(r['passed'] for r in numeric)))


def estimate_stage(root,m,cycles):
    costs={}
    for path in root.glob('*/level-*/batches/*.json'):
        a=json.loads(path.read_text());level=int(path.parent.parent.name.split('-')[-1]);costs.setdefault(level,[]).append(a['seconds'])
    missing=[t for t in tasks_for(m,cycles) if not task_path(root,t).exists()]
    if not missing:return 0.
    baseline={(c['arm'],c['level']):c['baseline_batch_seconds'] for c in m['cases']}
    return 1.3*sum(float(np.median(costs[t['level']])) if t['level'] in costs else baseline[(t['arm'],t['level'])] for t in missing)/2+120.


def run_workers(root,stage):
    processes=[];logs=[]
    try:
        for slot in range(2):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(slot),JAX_PLATFORMS='cuda',JAX_ENABLE_X64='true')
            log=(root/f'worker-{slot}.log').open('a');logs.append(log)
            processes.append(subprocess.Popen([sys.executable,'-u','-m',MODULE,'--output-dir',str(root),'--worker',str(slot),'--stage',str(stage)],env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            progress={}
            for slot in range(2):
                path=root/f'progress-{slot}.json';progress[slot]=json.loads(path.read_text()) if path.exists() else 'starting'
            print(json.dumps(dict(cycles=stage,workers=progress)),flush=True);time.sleep(15)
        return 0 if all(p.returncode==0 for p in processes) else 2
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()


def coordinate(root):
    m=json.loads((root/'manifest.json').read_text());job=json.loads((root/'job.json').read_text())
    state_path=root/'sampling-state.json';state=json.loads(state_path.read_text()) if state_path.exists() else dict(admitted=[],completed=[],stop_reason=None)
    state['stop_reason']=None
    if run_workers(root,0):return 2
    for cycles in STAGES:
        if (root/f'stage-{cycles}.json').exists():
            result=json.loads((root/f'stage-{cycles}.json').read_text())
            if cycles not in state['completed']:state['completed'].append(cycles)
            if result['panel_qualified']:state['stop_reason']='panel_stable';break
            continue
        if cycles not in state['admitted']:
            estimate=estimate_stage(root,m,cycles)
            if time.time()+estimate>=job['sampling_deadline_unix']:
                state['stop_reason']='sampling_allowance_reached; no new stage admitted';break
            state['admitted'].append(cycles);resume.atomic_json(state_path,state)
        # Once admitted, finish the stage and validation even after sampling allowance expires.
        if run_workers(root,cycles):return 2
        result=evaluate(root,m,cycles);resume.atomic_json(root/f'stage-{cycles}.json',result)
        state['completed'].append(cycles);resume.atomic_json(state_path,state)
        if result['panel_qualified']:state['stop_reason']='panel_stable';break
    if state['stop_reason'] is None:state['stop_reason']='maximum32cycles_reached'
    resume.atomic_json(state_path,state);return 0


def audit(root):
    m=json.loads((root/'manifest.json').read_text());problems=[];results=[];unavailable=[]
    for path,h in m['input_sha256'].items():
        if file_sha256(path)!=h:problems.append('Input hash '+path)
    for path,h in m['source_sha256'].items():
        if file_sha256(REPO/path)!=h:problems.append('Source hash '+path)
    source=Path(m['source_run']);original=json.loads((source/'manifest.json').read_text());expected_panel=set()
    for case in m['cases']:
        old_folder=source/case['arm']/f"full-{case['level']}";old_y=resume.old.prior.arrays(old_folder/'response.npz')['response']
        tolerance=json.loads((old_folder/'metrics.json').read_text())['sampling']['tolerance']
        expected_panel.update(affected_targets(old_y,original['full_targets'],tolerance))
        if case['tolerance']!=tolerance:problems.append('Changed original tolerance')
        ix=[original['full_targets'].index(t) for t in m['panel']]
        if not np.array_equal(resume.old.prior.arrays(folder_for(root,case)/'base-response.npz')['response'],old_y[:,ix]):problems.append('Reused response arithmetic')
    if sorted(expected_panel)!=m['panel']:problems.append('Affected-target panel arithmetic')
    extensions_complete=True
    for case in m['cases']:
        folder=folder_for(root,case);path=folder/'extension.json'
        if not path.exists():extensions_complete=False;unavailable.append(f"Missing extension: {case['arm']} level{case['level']}");continue
        metadata=json.loads(path.read_text());ix,recomputed=extension_phases(metadata['reference_voltage_mV'],resume.old.clock(case['level'])[0])
        if any(metadata[k]!=v for k,v in recomputed.items()):problems.append('Extension phase arithmetic '+str(folder))
        for p,h in metadata['sha256'].items():
            if file_sha256(p)!=h:problems.append('Extension hash '+p)
        states=resume.old.prior.arrays(folder/'extension-states.npz');ids=resume.old.prior.arrays(root/case['arm']/'model.npz')['neuron_ids'].tolist()
        if states['voltage_mV'].shape!=(112,302) or not np.array_equal(states['voltage_mV'][:,ids.index('RMDL')],np.asarray(metadata['reference_voltage_mV'])[ix]):problems.append('Extension/state correspondence '+str(folder))
        if not metadata['finite'] or metadata['voltage_min_mV']< -120 or metadata['voltage_max_mV']>80:problems.append('Extension dynamics '+str(folder))
    state=json.loads((root/'sampling-state.json').read_text()) if (root/'sampling-state.json').exists() else dict(admitted=[],completed=[],stop_reason='preparation_or_extension_incomplete')
    for cycles in state['admitted']:
        if not (root/f'stage-{cycles}.json').exists():unavailable.append('Admitted stage incomplete: '+str(cycles));continue
        expected=evaluate(root,m,cycles);actual=json.loads((root/f'stage-{cycles}.json').read_text())
        if expected!=actual:problems.append('Stage arithmetic '+str(cycles))
        results.append(actual)
    for path in root.glob('*/level-*/batches/*.npz'):
        with np.load(path,allow_pickle=False) as z:task=json.loads(str(z['provenance_json']))
        read_task(root,task)
    for path in root.glob('**/run_records/*/source_record.json'):
        source=json.loads(path.read_text())
        for name,h in source['source_sha256'].items():
            if file_sha256(path.parent/'source'/name)!=h:problems.append('Archived source '+name)
    complete=extensions_complete and len(results)==len(state['admitted'])
    result=dict(audit_complete=True,execution_complete=complete,integrity_passed=not problems,problems=problems,panel=m['panel'],
        stages=results,stop_reason=state['stop_reason'],panel_qualified=bool(not problems and results and results[-1]['panel_qualified']),
        saved_batches=sum(1 for _ in root.glob('*/level-*/batches/*.npz')),artifact_sha256={str(p):file_sha256(p) for p in root.rglob('*.npz')},
        worker_error_history=[json.loads(p.read_text()) for p in root.glob('failure-worker-*.json')],
        unavailable=unavailable+['Panel only, not all135-target qualification at expanded ensemble','Four old cycles retained; nested extension changes time support/weights as N grows','Deterministic stability screens, not confidence intervals or biological validation','Full32s all-cell trajectory not archived; selected full states/end state, reference RMDL voltage and recorded all-cell bounds/finite reductions retained','No fitting, optimizer resume, body/start-stop/swimming or biological qualification'],legacy_gates_changed=False,model_promoted=False)
    resume.atomic_json(root/'audit.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path);p.add_argument('--source-run',type=Path,default=SOURCE)
    p.add_argument('--worker',type=int,choices=(0,1));p.add_argument('--stage',type=int,choices=(0,8,16,32),default=0);p.add_argument('--coordinate',action='store_true')
    p.add_argument('--status',action='store_true');p.add_argument('--check',action='store_true');p.add_argument('--select-output',action='store_true');p.add_argument('--report',action='store_true');p.add_argument('--extend-sampling',action='store_true');a=p.parse_args()
    if a.select_output:print(select_output(a.source_run));return
    if a.status or a.check:
        found=jobs();print('READ-ONLY; no job started. Active jobs:',json.dumps(found));return 3 if a.check and found else 0
    if not a.output_dir:p.error('--output-dir required')
    root=a.output_dir.resolve()
    if a.report:
        print('READ-ONLY REPORT:',root)
        for name in ('audit.json','budget_receipt.json'):
            path=root/name;r=json.loads(path.read_text()) if path.exists() else {};print(name,json.dumps({k:v for k,v in r.items() if k!='artifact_sha256'}))
        return
    if a.worker is not None:
        try:return worker(root,a.worker,a.stage)
        except Exception as exc:
            resume.atomic_json(root/f'failure-worker-{a.worker}-{time.time_ns()}.json',dict(slot=a.worker,stage=a.stage,error=repr(exc)));raise
    if a.coordinate:return coordinate(root)
    if jobs():p.error('Active GPU work detected; nothing started')
    import fcntl
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'.cv-peptide-validation.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('Shared GPU lock held; nothing started')
        if jobs():p.error('GPU work detected after lock; nothing started')
        if root==REPO or REPO in root.parents or root in REPO.parents:p.error('External output required')
        root.mkdir(parents=True,exist_ok=True);started=time.monotonic();now=time.time();job_path=root/'job.json'
        job=json.loads(job_path.read_text()) if job_path.exists() else dict(source_run=str(a.source_run.resolve()),sampling_deadline_unix=now+180*60)
        if job['source_run']!=str(a.source_run.resolve()):p.error('Source mismatch')
        receipt=root/'budget_receipt.json'
        if receipt.exists() and json.loads(receipt.read_text()).get('all_steps_finished') and not a.extend_sampling:
            print('Already finished allowance; nothing started. Use --report; --extend-sampling explicitly adds another allowance.');return 0
        state_path=root/'sampling-state.json'
        if a.extend_sampling and state_path.exists() and json.loads(state_path.read_text()).get('stop_reason') in ('panel_stable','maximum32cycles_reached'):
            print('Planned sampling protocol already finished; nothing started. Use --report.');return 0
        if a.extend_sampling:job['sampling_deadline_unix']=now+180*60
        resume.atomic_json(job_path,job)
        if not (root/'manifest.json').exists():
            if shutil.disk_usage(root).free<10*1024**3:raise ValueError('Requires10GiB free')
            print('Verifying source and deriving affected-target panel; no completed responses will be replayed',flush=True);prepare(root,a.source_run)
        m=json.loads((root/'manifest.json').read_text())
        for path,h in m['source_sha256'].items():
            if file_sha256(REPO/path)!=h:raise ValueError('Pinned source changed; no silent resume')
        for path,h in m['input_sha256'].items():
            if file_sha256(path)!=h:raise ValueError('Input changed; no silent resume: '+path)
        capture_run_record(root,REPO,dict(source_run=str(a.source_run)),phase='E127-three-hour-sampling')
        resume.atomic_json(receipt,dict(status='running',sampling_minutes=180,validation_allowed_to_finish=True,all_steps_finished=False,audit_complete=False))
        code=1
        try:
            from scripts.total_job_budget import run_group
            code,_=run_group([sys.executable,'-u','-m',MODULE,'--output-dir',str(root),'--coordinate'],None)
        except Exception as exc:traceback.print_exc();resume.atomic_json(root/'failure.json',dict(error=repr(exc)))
        try:result=audit(root)
        except Exception as exc:traceback.print_exc();result=dict(audit_complete=False,execution_complete=False,integrity_passed=False,problems=[repr(exc)]);resume.atomic_json(root/'audit.json',result)
        outcome=dict(status='finished',exit_code=code,timed_out=False,elapsed_seconds=time.monotonic()-started,sampling_minutes=180,validation_allowed_to_finish=True,
            all_steps_finished=bool(code==0 and result['execution_complete'] and result['audit_complete']),audit_complete=result['audit_complete'],panel_qualified=bool(result.get('panel_qualified')),
            completed_cycle_counts=[r['cycles'] for r in result.get('stages',[])],stop_reason=result.get('stop_reason'),training_launched=False,model_promoted=False)
        resume.atomic_json(receipt,outcome);print(json.dumps(outcome));return 0 if outcome['all_steps_finished'] and result['integrity_passed'] else 2


if __name__=='__main__':main()
