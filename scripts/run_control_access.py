"""E076 bounded parallel interneuron control-access screen and refinement."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from scripts.experiment_records import capture_run_record,file_sha256
from r_elegans.control_access import cases,compare
from r_elegans.wall_budget import check_deadline

PARENT='2a2b30b4324163aaba5e6c70d4d549cf0c689732d03ebb65f03aec297ee0924f'


def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def read_trace(folder):
    report=json.loads((folder/'summary.json').read_text());path=folder/'trajectory.npz'
    if report['checkpoint_sha256']!=PARENT or file_sha256(path)!=report['trajectory_sha256']:raise ValueError('Probe artifact mismatch')
    with np.load(path) as z:return {k:z[k] for k in ('dense_time_s','dense_joints','dense_voltage_mV','dense_muscles','dense_command','dense_filtered_drive')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--source-run',type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();source=a.source_run.resolve();repo=Path(__file__).resolve().parents[1]
    for root in (repo,source):
        if out==root or root in out.parents or out in root.parents:raise ValueError('Separate external output required')
    budget=json.loads((out/'deadline.json').read_text());finish=budget['coordinator_finish_by'];check_deadline(finish)
    capture_run_record(out,repo,a)
    summary=json.loads((source/'summary.json').read_text());selected=Path(next(x['selected'] for x in summary['phases'] if x['stage']=='control')).resolve()
    if not selected.is_relative_to(source/'control/fit') or file_sha256(selected/'network_best.npz')!=PARENT:raise ValueError('Pinned E074 selected checkpoint required')
    from scripts.compare_rhythm_worker import verify_inputs
    verify_inputs(source/'control_inputs');inputs=out/'inputs';shutil.copytree(source/'control_inputs',inputs)
    manifest=json.loads((inputs/'manifest.json').read_text())
    copies={'access_source_manifest.json':source/'control_inputs/manifest.json','access_model.npz':selected/'network_best.npz',
        'access_coordinates.npz':selected/'best_state.npz','access_parent_summary.json':selected/'summary.json',
        'access_reference.npz':source/'control/validate/no-cue/trajectory.npz',
        'access_reference_summary.json':source/'control/validate/no-cue/summary.json'}
    reference=json.loads(copies['access_reference_summary.json'].read_text())
    if reference['checkpoint_sha256']!=PARENT or file_sha256(copies['access_reference.npz'])!=reference['trajectory_sha256']:raise ValueError('Reference trajectory hash mismatch')
    for name,path in copies.items():
        before=file_sha256(path);shutil.copy2(path,inputs/name)
        if before!=file_sha256(inputs/name) or before!=file_sha256(path):raise ValueError('Input changed while copied')
        manifest['sha256'][name]=before;manifest['sources'][name]=str(path)
    manifest['consumer_experiment']='E076';write(inputs/'manifest.json',manifest);verify_inputs(inputs)
    from r_elegans.model import load_builtin_model
    neuron_ids=load_builtin_model().neuron_ids
    from scripts.benchmark_cpu import physical_core_cpus
    from scripts.benchmark_cpu_parallel import worker_cpu_groups
    cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0)))
    if not known or not shutil.which('taskset'):raise ValueError('Known physical topology and taskset required')
    groups=worker_cpu_groups(cpus,4,4);all_cases=cases()
    write(out/'configuration.json',dict(experiment='E076',cases=all_cases,parent_sha256=PARENT,no_training=True,
        duration_s=60,probe_on_s=30,probe_off_s=40,screening_deadline_monotonic=finish-15*60,
        refinement_reserve_seconds=15*60,qualification='Diagnostic reversible suppression only; never learned stopping'))
    env=os.environ.copy();env.update(JAX_PLATFORMS='cpu',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def pool(tasks,phase,until):
        folder=out/phase;folder.mkdir();waiting=list(tasks);running={};done=[];heartbeat=time.monotonic()
        try:
            while waiting or running:
                check_deadline(finish-30)
                for slot in range(4):
                    if slot in running and running[slot][0].poll() is not None:
                        proc,log,name=running.pop(slot);log.close();record=folder/name/'summary.json'
                        result=json.loads(record.read_text()) if record.exists() else None
                        done.append(dict(name=name,path=str(folder/name),exit_code=proc.returncode,result=result))
                        print(json.dumps(dict(finished=name,exit_code=proc.returncode,phase=phase)),flush=True)
                    if slot not in running and waiting and time.monotonic()<until-60:
                        case,grid=waiting.pop(0);name=case['name']+'--'+grid;log=(folder/(name+'.log')).open('w')
                        command=['taskset','-c',','.join(map(str,groups[slot])),sys.executable,'-u','-m','scripts.control_access_worker',
                            '--inputs',str(inputs),'--output-dir',str(folder/name),'--case',case['name'],'--grid',grid,'--until',str(until)]
                        proc=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT);running[slot]=(proc,log,name)
                        print(json.dumps(dict(starting=name,pid=proc.pid,cpus=groups[slot],phase=phase)),flush=True)
                if time.monotonic()-heartbeat>=30:
                    print(json.dumps(dict(phase=phase,running=len(running),pending=len(waiting),remaining_minutes=(finish-time.monotonic())/60)),flush=True);heartbeat=time.monotonic()
                if not running and waiting:break
                if running:time.sleep(.5)
        finally:
            for proc,log,_ in running.values():
                if proc.poll() is None:proc.terminate()
                try:proc.wait(timeout=2)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                log.close()
        return done,waiting
    screen,pending=pool([(c,'ordinary') for c in all_cases],'screen',finish-15*60)
    good={r['result']['case']['name']:r for r in screen if r['exit_code']==0 and r['result']}
    comparisons=[];refinement=[];ref_results=[];ref_pending=[]
    if 'sham' in good:
        sham=read_trace(Path(good['sham']['path']))
        for case,row in good.items():
            if case=='sham':continue
            trace=read_trace(Path(row['path']));m=compare(trace,sham)
            indices=[neuron_ids.index(n) for n in row['result']['targets']]
            use=(trace['dense_time_s']>=35)&(trace['dense_time_s']<40)
            m['output_chain_during_probe']={k:dict(maximum_raw_range=float(np.max(np.ptp(trace[k][use],axis=0))),sham_maximum_raw_range=float(np.max(np.ptp(sham[k][use],axis=0)))) for k in ('dense_muscles','dense_command','dense_filtered_drive')}
            m['target_sham_mean_voltage_mV']=np.mean(sham['dense_voltage_mV'][use][:,indices],axis=0).tolist()
            m['target_mean_voltage_mV']=np.mean(trace['dense_voltage_mV'][use][:,indices],axis=0).tolist()
            comparisons.append(dict(case=row['result']['case'],metrics=m))
        ranked=sorted((r for r in comparisons if r['metrics']['pre_cue_match'] and r['metrics']['safety_passed']),key=lambda r:r['metrics']['during_to_sham_raw_range_ratio'])
        if ranked and finish-time.monotonic()>5*60:
            best=ranked[0];case=best['case'];sham_case=all_cases[0]
            ref_results,ref_pending=pool([(c,g) for g in ('neural-half','body-half') for c in (sham_case,case)],'refinement',finish-60)
            ref_good={r['name']:r for r in ref_results if r['exit_code']==0 and r['result']}
            for grid in ('neural-half','body-half'):
                keys=[c['name']+'--'+grid for c in (case,sham_case)]
                if not all(k in ref_good for k in keys):continue
                m=compare(read_trace(Path(ref_good[keys[0]]['path'])),read_trace(Path(ref_good[keys[1]]['path'])))
                original=best['metrics'];a0=original['during_to_sham_raw_range_ratio'];a1=m['during_to_sham_raw_range_ratio']
                consistent=abs(a0-a1)<=max(.02,.2*abs(a0))
                refinement.append(dict(grid=grid,case=case,metrics=m,ratio_consistent=consistent,
                    passed=bool(consistent and m['reversible_diagnostic_suppression'])))
    verify_inputs(inputs)
    full=not pending and len(screen)==len(all_cases) and all(r['exit_code']==0 for r in screen)
    access=bool(full and len(refinement)==2 and all(r['passed'] for r in refinement))
    complete=bool(full and len(refinement)==2 and not ref_pending and all(r['exit_code']==0 for r in ref_results))
    report=dict(experiment='E076',checkpoint_sha256=PARENT,execution_valid=bool(full and all(r['exit_code']==0 for r in ref_results)),
        all_stages_finished=complete,validation_complete=complete,diagnostic_control_access=access,
        learned_stopping_qualified=False,source_run=str(source),screen_results=screen,comparisons=comparisons,
        refinement_results=ref_results,refinement=refinement,unstarted_screen=[(c['name'],g) for c,g in pending],
        scope='Imposed interneuron conductance tests; no training, learned stop or biological qualification',
        interpretation='Negative results exclude only tested groups/strengths/timing, not all possible control mechanisms')
    write(out/'summary.json',report)
    print('CONTROL_ACCESS_REPORT:',out/'summary.json',flush=True)
    for row in sorted(comparisons,key=lambda r:r['metrics']['during_to_sham_raw_range_ratio']):
        m=row['metrics'];print(json.dumps(dict(case=row['case']['name'],range_ratio=m['during_to_sham_raw_range_ratio'],quiet=m['quiet_during_probe'],recovery=m['recovery_amplitude_preserved'],pre_match=m['pre_cue_match'],safe=m['safety_passed'],target_mean_voltage_mV=m['target_mean_voltage_mV'])),flush=True)
    print(json.dumps(dict(diagnostic_control_access=access,learned_stopping_qualified=False,validation_complete=complete,refinement=refinement)),flush=True)


if __name__=='__main__':main()
