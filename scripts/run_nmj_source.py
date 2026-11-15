"""E080/E081 bounded diagnostic screen of rhythm-carrying NMJ populations."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from scipy.special import expit
from scripts.experiment_records import capture_run_record,file_sha256
from r_elegans.nmj_source import protocol_settings,targets_output,refinement_target,split_controls
from r_elegans.control_access import compare
from r_elegans.wall_budget import check_deadline

SELECTED='dd42e3f62bc00befc6052fae60c102bc89aaf9266ae88da1f78f3be30c2766a8'


def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--protocol',choices=('population','head-split'),default='population')
    a=p.parse_args();out=a.output_dir.resolve();source=a.source_run.resolve();repo=Path(__file__).resolve().parents[1]
    experiment,cases,_=protocol_settings(a.protocol)
    for root in (repo,source):
        if out==root or root in out.parents or out in root.parents:raise ValueError('Separate output required')
    finish=json.loads((out/'deadline.json').read_text())['coordinator_finish_by'];check_deadline(finish)
    capture_run_record(out,repo,a)
    summary=json.loads((source/'summary.json').read_text())
    ordinary=next(r for r in summary['validation'] if r['variant']=='ordinary' and r['grid']=='ordinary')
    folder=Path(ordinary['output']).resolve()
    if source not in folder.parents or ordinary['checkpoint_sha256']!=SELECTED:raise ValueError('Pinned E078 selected ordinary evaluation required')
    reference=folder/'sham.npz'
    if file_sha256(reference)!=ordinary['trajectory_sha256']['sham'] or file_sha256(folder/'network.npz')!=SELECTED:raise ValueError('Source hash mismatch')
    from scripts.compare_rhythm_worker import verify_inputs
    verify_inputs(source/'inputs');inputs=out/'inputs';shutil.copytree(source/'inputs',inputs)
    manifest=json.loads((inputs/'manifest.json').read_text())
    for name,path in [('diagnostic_model.npz',folder/'network.npz'),('diagnostic_coordinates.npz',folder/'coordinates.npz'),('source_summary.json',source/'summary.json')]:
        sha=file_sha256(path);shutil.copy2(path,inputs/name)
        if sha!=file_sha256(inputs/name) or sha!=file_sha256(path):raise ValueError('Changing input')
        manifest['sha256'][name]=sha;manifest['sources'][name]=str(path)
    with np.load(inputs/'diagnostic_model.npz',allow_pickle=False) as z:
        threshold=z['fixed_release_threshold_mV'];slope=z['fixed_release_slope_mV'];ids=z['neuron_ids'].tolist()
    with np.load(reference,allow_pickle=False) as z:
        use=(z['dense_time_s']>=20)&(z['dense_time_s']<30)
        if use.sum()<4:raise ValueError('Missing baseline')
        mean=expit((z['dense_voltage_mV'][use]-threshold)/slope).mean(axis=0)
    np.savez_compressed(inputs/'baseline_release.npz',mean=mean)
    write(inputs/'reference.json',dict(path=str(reference),sha256=ordinary['trajectory_sha256']['sham']))
    for name in ('baseline_release.npz','reference.json'):
        manifest['sha256'][name]=file_sha256(inputs/name);manifest['sources'][name]='Generated from hash-verified selected sham'
    manifest['consumer_experiment']=experiment;write(inputs/'manifest.json',manifest);verify_inputs(inputs)
    write(out/'configuration.json',dict(experiment=experiment,protocol=a.protocol,checkpoint_sha256=SELECTED,cases=cases,
        groups={c:[n for n in ids if targets_output(n,c)] for c in cases if c!='sham'},
        onset_s=30,offset_s=40,simulation_seconds=60,baseline_mean_window=[20,30],no_training=True,
        corrected_motor_identity_used=True,scope='NMJ rhythm transmission, not isolated oscillator identity or learned stopping'))
    from scripts.benchmark_cpu import physical_core_cpus
    from scripts.benchmark_cpu_parallel import worker_cpu_groups
    cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0)))
    if not known or not shutil.which('taskset'):raise ValueError('Physical topology and taskset required')
    groups=worker_cpu_groups(cpus,4,4)
    env=os.environ.copy();env.update(JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='1')
    def pool(tasks,phase,until):
        dest=out/phase;dest.mkdir();waiting=list(tasks);running={};done=[];last=time.monotonic()
        try:
            while waiting or running:
                check_deadline(until)
                for slot in range(4):
                    if slot in running and running[slot][0].poll() is not None:
                        proc,log,name=running.pop(slot);log.close();path=dest/name/'summary.json'
                        r=json.loads(path.read_text()) if path.exists() else None
                        done.append(dict(name=name,path=str(dest/name),exit_code=proc.returncode,result=r))
                        print(json.dumps(dict(finished=name,exit_code=proc.returncode)),flush=True)
                    if slot not in running and waiting:
                        case,grid=waiting.pop(0);name=case+'--'+grid;log=(dest/(name+'.log')).open('w')
                        cmd=['taskset','-c',','.join(map(str,groups[slot])),sys.executable,'-u','-m','scripts.nmj_source_worker',
                             '--inputs',str(inputs),'--output-dir',str(dest/name),'--case',case,'--grid',grid,'--experiment',experiment,'--until',str(until-15)]
                        proc=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT);running[slot]=(proc,log,name)
                        print(json.dumps(dict(starting=name,pid=proc.pid,cpus=groups[slot])),flush=True)
                if time.monotonic()-last>30:
                    print(json.dumps(dict(phase=phase,running=len(running),pending=len(waiting),remaining_minutes=(finish-time.monotonic())/60)),flush=True);last=time.monotonic()
                if running:time.sleep(.5)
        finally:
            for proc,log,_ in running.values():
                if proc.poll() is None:proc.terminate()
                try:proc.wait(timeout=2)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                log.close()
        return done
    def trace(row):
        r=row['result'];path=Path(row['path'])/'trajectory.npz'
        if r['checkpoint_sha256']!=SELECTED or file_sha256(path)!=r['trajectory_sha256']:raise ValueError('Diagnostic trace mismatch')
        with np.load(path,allow_pickle=False) as z:
            return {k:z[k] for k in ('dense_time_s','dense_voltage_mV','dense_joints','dense_muscles','dense_command','dense_filtered_drive')}
    def compare_rows(rows,grid):
        good={r['result']['case']:r for r in rows if r['exit_code']==0 and r['result'] and r['result']['grid']==grid}
        if 'sham' not in good:return []
        sham=trace(good['sham']);result=[]
        for case,row in good.items():
            if case=='sham':continue
            tested=trace(row);m=compare(tested,sham)
            use=(tested['dense_time_s']>=35)&(tested['dense_time_s']<40)
            m['NMJ_muscle_range_ratio']=float(np.max(np.ptp(tested['dense_muscles'][use],axis=0))/max(1e-12,np.max(np.ptp(sham['dense_muscles'][use],axis=0))))
            result.append(dict(case=case,grid=grid,metrics=m))
        return result
    failure=None;screen=[];refined=[];comparisons=[];refcomparisons=[]
    try:
        screen=pool([(c,'ordinary') for c in cases],'screen',finish-15*60)
        comparisons=compare_rows(screen,'ordinary')
        best=refinement_target(comparisons,a.protocol)
        if best is not None:
            refined=pool([(c,g) for g in ('neural-half','body-half') for c in ('sham',best)],'refinement',finish-45)
            for grid in ('neural-half','body-half'):refcomparisons+=compare_rows(refined,grid)
    except Exception as e:failure=repr(e);print('NMJ_AUDIT_ERROR:',failure,flush=True)
    verify_inputs(inputs)
    complete=bool(failure is None and len(screen)==len(cases) and len(refined)==4 and len(refcomparisons)==2 and all(r['exit_code']==0 and r['result'] for r in screen+refined))
    checks=[]
    for row in refcomparisons:
        ordinary=next(r for r in comparisons if r['case']==row['case'])
        delta=abs(ordinary['metrics']['during_to_sham_raw_range_ratio']-row['metrics']['during_to_sham_raw_range_ratio'])
        checks.append(dict(case=row['case'],grid=row['grid'],suppression_ratio_difference=delta,passed=bool(delta<=.05 and row['metrics']['pre_cue_match'] and row['metrics']['safety_passed'])))
    result=dict(experiment=experiment,protocol=a.protocol,checkpoint_sha256=SELECTED,execution_valid=complete,all_stages_finished=complete,validation_complete=complete,
        failure=failure,screen=screen,comparisons=comparisons,refinement=refined,refinement_comparisons=refcomparisons,numerical_checks=checks,
        learned_stopping_qualified=False,swimming_qualified=False,scope='Imposed NMJ release holds localize rhythm transmission; do not prove isolated pacemaker origin or learned stopping')
    if a.protocol=='head-split':result['diagnostic_controls']=split_controls(comparisons)
    write(out/'summary.json',result)
    for r in comparisons+refcomparisons:
        print(json.dumps(dict(case=r['case'],grid=r['grid'],body_range_ratio=r['metrics']['during_to_sham_raw_range_ratio'],muscle_range_ratio=r['metrics']['NMJ_muscle_range_ratio'],recovery=r['metrics']['recovery_amplitude_preserved'])),flush=True)
    print('Report:',out/'summary.json',flush=True)


if __name__=='__main__':main()
