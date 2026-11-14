"""Sensory stop/restart first; physical swimming only after full control gate."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from scripts.experiment_records import capture_run_record,file_sha256
from r_elegans.behavior_control import VALIDATION_CASES,CONTROL_GROUPS,control_report,qualify_behavior
from r_elegans.wall_budget import check_deadline,control_search_deadline

PARENT='8e26b5d0b776edd29c1ded289c1d3aedc2897e4cce1159b08d98e0b75cc0ce2a'


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def snapshot(source_inputs,out,worker,reference_protocol,gate=None):
    from scripts.compare_rhythm_worker import verify_inputs
    verify_inputs(source_inputs)
    shutil.copytree(source_inputs,out)
    m=json.loads((out/'manifest.json').read_text())
    shutil.copy2(out/'manifest.json',out/'previous_manifest.json')
    summary=json.loads((worker/'summary.json').read_text())
    tag=summary.get('selected') or f"iteration-{summary['selected_iteration']:06d}"
    selected=worker/'selected'/tag
    copies={'previous_manifest.json':source_inputs/'manifest.json','continuation_parent.npz':worker/'network_best.npz',
            'parent_best_state.npz':worker/'best_state.npz',
            'parent_training_trace.npz':worker/'best_training_trace.npz' if (worker/'best_training_trace.npz').exists() else selected/'trajectory.npz',
            'parent_behavior_summary.json':worker/'summary.json','behavior_parent_selected_rng.json':selected/'rng_state.json',
            'behavior_parent_configuration.json':worker/'configuration.json',
            'behavior_parent_final_optimizer.npz':worker/'optimizer_state.npz','behavior_parent_final_rng.json':worker/'rng_state.json'}
    before={str(path):file_sha256(path) for path in copies.values()}
    for name,path in copies.items():
        shutil.copy2(path,out/name)
        if file_sha256(out/name)!=before[str(path)]:raise ValueError('Input snapshot mismatch')
        m['sha256'][name]=before[str(path)];m['sources'][name]=str(path)
    if gate:
        write(out/'control_gate.json',gate);m['sha256']['control_gate.json']=file_sha256(out/'control_gate.json')
        m['sources']['control_gate.json']=str(worker.parent.parent/'control_gate.json')
    m.update(consumer_experiment='E073' if gate else 'E072',continuation_parent_sha256=file_sha256(out/'continuation_parent.npz'),
             reference_protocol=reference_protocol)
    write(out/'manifest.json',m);verify_inputs(out)
    if before!={str(path):file_sha256(path) for path in copies.values()}:raise ValueError('Parent changed during snapshot')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--source-run',type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();source=a.source_run.resolve();repo=Path(__file__).resolve().parents[1]
    if out==repo or repo in out.parents or out==source or source in out.parents:raise ValueError('Separate external output required')
    budget=json.loads((out/'deadline.json').read_text());finish=budget['coordinator_finish_by'];check_deadline(finish)
    parent=source/'seed-6901';s=json.loads((parent/'summary.json').read_text())
    receipt=json.loads((source/'budget_receipt.json').read_text())
    if (not s['passed'] or s['status']!='completed' or s['checkpoint_sha256']!=PARENT
            or file_sha256(parent/'network_best.npz')!=PARENT or not receipt['all_steps_finished']
            or not receipt['validation_complete']):raise ValueError('Pinned qualified E0716901 with complete validation required')
    capture_run_record(out,repo,a)
    snapshot(source/'inputs',out/'control_inputs',parent,'no-cue')
    from scripts.benchmark_cpu import physical_core_cpus
    from scripts.benchmark_cpu_parallel import worker_cpu_groups
    cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0)))
    if not known or not shutil.which('taskset'):raise ValueError('Physical CPU affinity required')
    groups=worker_cpu_groups(cpus,4,4)
    # Validation is one selected model across four workers, rather than four
    # copies of the entire suite. Scale measured parent cost by simulated work.
    measured=s['total_elapsed_seconds']-s['search_elapsed_seconds']
    reserve=max(45*60.,2*measured*(1890/660)/4+600)
    if finish-time.monotonic()<2*reserve+20*60:raise ValueError('Not enough total budget for both gated stages and validation reserves')
    env=os.environ.copy();env.update(JAX_PLATFORMS='cpu',JAX_ENABLE_X64='false',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    write(out/'configuration.json',dict(parent_sha256=PARENT,total_minutes=budget['total_minutes'],validation_reserve_seconds_each=reserve,
        phases=['sensory-control','controlled-propulsion'],automatic_swimming_only_after_verified_control=True,
        controller_groups=CONTROL_GROUPS,engineering_forward_speed_BL_s=.01))

    def pool(stage,mode,inputs,until,cases=None,checkpoint=None,coordinates=None):
        destination=out/stage/mode;destination.mkdir(parents=True)
        tasks=list(cases) if mode=='validate' else list(range(4))
        running={};finished=[];stop=destination/'stop-search.flag';heartbeat=time.monotonic()
        def start(task,slot):
            name=str(task) if mode=='validate' else f'seed-{(7200 if stage=="control" else 7300)+task}'
            child=destination/name
            command=['taskset','-c',','.join(map(str,groups[slot])),sys.executable,'-u','-m','scripts.behavior_control_worker',
                '--inputs',str(inputs),'--output-dir',str(child),'--mode',mode,'--stage',stage,'--until',str(until)]
            if mode=='fit':command+=['--seed',str((7200 if stage=='control' else 7300)+task),'--stop-flag',str(stop)]
            else:command+=['--case',task,'--checkpoint',str(checkpoint),'--coordinates',str(coordinates)]
            log=(destination/(name+'.log')).open('w');process=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            running[slot]=(process,log,child,task)
            print(json.dumps(dict(stage=stage,mode=mode,starting=name,pid=process.pid,cpus=groups[slot])),flush=True)
        try:
            while tasks or running:
                check_deadline(finish-30)
                for slot in range(4):
                    if slot in running and running[slot][0].poll() is not None:
                        process,log,child,task=running.pop(slot);log.close();path=child/'summary.json'
                        result=json.loads(path.read_text()) if path.exists() else None
                        finished.append(dict(path=str(child),exit_code=process.returncode,result=result))
                        print(json.dumps(dict(stage=stage,mode=mode,finished=str(task),exit_code=process.returncode,
                                              nominal_passed=result.get('nominal_passed') if result else None)),flush=True)
                        if mode=='fit' and result and result['nominal_passed']:stop.touch()
                    if slot not in running and tasks:start(tasks.pop(0),slot)
                if time.monotonic()-heartbeat>=30:
                    print(json.dumps(dict(stage=stage,mode=mode,remaining_minutes=(finish-time.monotonic())/60,running=len(running))),flush=True);heartbeat=time.monotonic()
                if running:time.sleep(.5)
        finally:
            for process,_,_,_ in running.values():
                if process.poll() is None:process.terminate()
            for process,log,_,_ in running.values():
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                log.close()
        return finished

    def validate(stage,inputs,chosen):
        selected=Path(chosen['path']);model=selected/'network_best.npz';coords=selected/'best_state.npz'
        capsule=out/stage/'validation_inputs';shutil.copytree(inputs,capsule)
        shutil.copy2(selected/'baseline.json',capsule/'behavior_baseline.json')
        m=json.loads((capsule/'manifest.json').read_text());m['sha256']['behavior_baseline.json']=file_sha256(capsule/'behavior_baseline.json');write(capsule/'manifest.json',m)
        results=pool(stage,'validate',capsule,finish-90,cases=VALIDATION_CASES,checkpoint=model,coordinates=coords)
        cases=[r['result'] for r in results if r['exit_code']==0 and r['result']]
        rows={r['case']:r for r in cases};refinement=[];swim_refinement=[]
        if 'nominal' in rows:
            nominal=Path(next(r['path'] for r in results if r['result'] and r['result']['case']=='nominal'))
            with np.load(nominal/'trajectory.npz') as z:t=z['dense_time_s'];q=z['dense_joints']
            for name in ('chemical-blocked','no-cue','start-blocked','stop-blocked'):
                if name in rows:
                    path=Path(next(r['path'] for r in results if r['result'] and r['result']['case']==name))
                    with np.load(path/'trajectory.npz') as z:
                        end=58 if name=='start-blocked' else 40
                        key='pre_start_maximum_difference_rad' if name=='start-blocked' else 'pre_cue_maximum_difference_rad'
                        rows[name][key]=float(np.max(np.abs(q[t<end]-z['dense_joints'][z['dense_time_s']<end])))
            for name in ('neural-half','body-half'):
                if name not in rows:continue
                path=Path(next(r['path'] for r in results if r['result'] and r['result']['case']==name))
                with np.load(path/'trajectory.npz') as z:
                    qt=np.column_stack([np.interp(t,z['dense_time_s'],column) for column in z['dense_joints'].T])
                error=float(np.max(abs(q-qt)))
                scale=max(float(np.max(np.ptp(q,axis=0))),.003)
                original=rows['nominal']['metrics'];refined=rows[name]['metrics']
                a0=np.asarray([r['amplitudes'] for r in original['on']]);a1=np.asarray([r['amplitudes'] for r in refined['on']])
                amplitudes=bool(np.all(abs(a0-a1)<=np.maximum(.2*a0,1e-5)))
                f0=[f for r in original['on'] for f in r['frequency_hz']];f1=[f for r in refined['on'] for f in r['frequency_hz']]
                frequencies=all(x is not None and y is not None and abs(x-y)<=.1*x for x,y in zip(f0,f1))
                latencies=True
                for old,new in zip(original['transition_latencies'],refined['transition_latencies']):
                    for key in ('settled_stop_latency_s','first_detectable_movement_latency_s'):
                        x,y=old[key],new[key]
                        latencies=latencies and x is not None and y is not None and abs(x-y)<=max(.5,.2*x)
                # Raw trajectory phase can drift with a small frequency change;
                # require independently positive state gates and feature/latency convergence.
                refinement.append(dict(case=name,maximum_raw_waveform_difference_rad_descriptive=error,
                    amplitudes_consistent=amplitudes,frequencies_consistent=frequencies,latencies_consistent=bool(latencies),
                    passed=bool(amplitudes and frequencies and latencies and refined['control_passed'])))
                speeds0=[r['propulsion']['speed_BL_s'] for r in rows['nominal']['metrics']['on']]
                speeds1=[r['propulsion']['speed_BL_s'] for r in rows[name]['metrics']['on']]
                speed_ok=all(x is not None and y is not None and abs(x-y)<=max(1e-4,.2*max(abs(x),abs(y))) for x,y in zip(speeds0,speeds1))
                swim_refinement.append(dict(case=name,passed=speed_ok))
        gate=control_report(file_sha256(model),cases,len(refinement)==2 and all(r['passed'] for r in refinement))
        gate['refinement']=refinement;gate['cases']=cases;gate['swim_refinement']=swim_refinement
        gate['execution_valid']=len(results)==len(VALIDATION_CASES) and all(r['exit_code']==0 for r in results)
        gate['swimming_passed']=bool(gate['passed'] and len(swim_refinement)==2 and all(r['passed'] for r in swim_refinement) and all(rows[n]['metrics']['swimming_passed'] for n in
                ('nominal','repeat','long-rest','onset-shift','dose-low','dose-high','redundant','neural-half','body-half')))
        write(out/stage/'control_gate.json',gate)
        return gate

    phases=[];searches={}
    control_until=control_search_deadline(finish,reserve)
    write(out/'phase_deadlines.json',dict(control_search_until_monotonic=control_until,
        control_search_minutes_available=(control_until-time.monotonic())/60,
        validation_reserve_seconds=reserve,report_margin_seconds=600,
        policy='Use available time for control; finish early on nominal pass; validate best even if training gate fails'))
    fits=pool('control','fit',out/'control_inputs',control_until)
    searches['control']=fits
    eligible=sorted((r for r in fits if r['exit_code']==0 and r['result'] and r['result']['nominal_passed']),key=lambda r:r['result']['best']['loss'])
    control=None;swim=None;qualified=False;complete=False
    candidates=eligible or sorted((r for r in fits if r['exit_code']==0 and r['result']),key=lambda r:r['result']['best']['loss'])
    if candidates:
        chosen=candidates[0];control=validate('control',out/'control_inputs',chosen)
        phases.append(dict(stage='control',selected=chosen['path'],report=str(out/'control/control_gate.json'),passed=control['passed']))
        if qualify_behavior(control['checkpoint_sha256'],control['swimming_passed'],control):
            qualified=True;complete=True
            phases.append(dict(stage='swim',selected=chosen['path'],report=str(out/'control/control_gate.json'),passed=True,
                               reused_same_checkpoint_motion_validation=True))
        elif control['passed'] and finish-time.monotonic()>reserve+15*60:
            selected=Path(chosen['path'])
            snapshot(out/'control_inputs',out/'swim_inputs',selected,'nominal',control)
            fit_until=finish-reserve-600
            fits=pool('swim','fit',out/'swim_inputs',fit_until)
            searches['swim']=fits
            candidates=sorted((r for r in fits if r['exit_code']==0 and r['result']),key=lambda r:r['result']['best']['loss'])
            if candidates:
                chosen=candidates[0];swim=validate('swim',out/'swim_inputs',chosen)
                qualified=qualify_behavior(swim['checkpoint_sha256'],swim['swimming_passed'],swim)
                phases.append(dict(stage='swim',selected=chosen['path'],report=str(out/'swim/control_gate.json'),passed=qualified));complete=swim['complete']
    valid=all(r['exit_code']==0 and r['result'] for runs in searches.values() for r in runs)
    valid=bool(valid and (control is None or control['execution_valid']) and (swim is None or swim['execution_valid']))
    summary=dict(experiment='E072-E073',execution_valid=valid,all_stages_finished=bool(complete),
        validation_complete=bool(complete),phases=phases,searches=searches,control_qualified=bool(control and control['passed']),
        control_validation_complete=bool(control and control['complete']),swim_validation_complete=bool(complete),
        swimming_qualified=qualified,swimming_blocked=not bool(control and control['passed']),
        scope='Learned cue association and first physical-propulsion engineering rung; biological fidelity not established')
    if not candidates:summary['stop_reason']='no_successful_control_fit_process; swimming not started'
    elif control and not control['passed']:summary['stop_reason']='sensory_control_validation_failed; swimming not started'
    elif not complete:summary['stop_reason']='insufficient_remaining_budget_or_incomplete_swim_validation'
    write(out/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='searches'}),flush=True)


if __name__=='__main__':main()
