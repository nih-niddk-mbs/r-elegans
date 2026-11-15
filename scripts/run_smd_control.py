"""E082 bounded SMD neural inhibition dose/group screen and confirmation."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from scripts.experiment_records import capture_run_record,file_sha256
from r_elegans.smd_control import cases,compare,choose
from r_elegans.wall_budget import check_deadline

SELECTED='dd42e3f62bc00befc6052fae60c102bc89aaf9266ae88da1f78f3be30c2766a8'


def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--source-run',type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();source=a.source_run.resolve();repo=Path(__file__).resolve().parents[1]
    experiment='E082';all_cases=cases()
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
    with np.load(inputs/'diagnostic_model.npz',allow_pickle=False) as z:ids=z['neuron_ids'].tolist()
    write(inputs/'reference.json',dict(path=str(reference),sha256=ordinary['trajectory_sha256']['sham']))
    for name in ('reference.json',):
        manifest['sha256'][name]=file_sha256(inputs/name);manifest['sources'][name]='Generated from hash-verified selected sham'
    manifest['consumer_experiment']=experiment;write(inputs/'manifest.json',manifest);verify_inputs(inputs)
    write(out/'configuration.json',dict(experiment=experiment,checkpoint_sha256=SELECTED,cases=all_cases,
        no_training=True,membrane_reversal_mV=-80.,screen_variant='nominal',confirmation_variants=['shift','repeat','long-hold'],
        simulation_seconds_screen=60,total_minutes=json.loads((out/'deadline.json').read_text())['total_minutes'],
        scope='Imposed neural inhibition/release, not learned stopping or sensory behavior'))
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
                        case,grid,variant=waiting.pop(0);name=variant+'--'+case+'--'+grid;log=(dest/(name+'.log')).open('w')
                        cmd=['taskset','-c',','.join(map(str,groups[slot])),sys.executable,'-u','-m','scripts.smd_control_worker',
                             '--inputs',str(inputs),'--output-dir',str(dest/name),'--case',case,'--grid',grid,'--variant',variant,'--until',str(until-15)]
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
    def compare_rows(rows,grid,variant='nominal'):
        good={r['result']['case']['name']:r for r in rows if r['exit_code']==0 and r['result'] and r['result']['grid']==grid and r['result']['variant']==variant}
        if 'sham' not in good:return []
        sham=trace(good['sham']);result=[]
        for name,row in good.items():
            if name=='sham':continue
            tested=trace(row);case=row['result']['case'];m=compare(tested,sham,ids,case,variant)
            result.append(dict(case=case,grid=grid,variant=variant,metrics=m))
        return result
    failure=None;screen=[];refined=[];confirmation=[];comparisons=[];refcomparisons=[];confirmed=[];best=None
    try:
        # Preserve at least90min for longer/repeated/shifted trials and refinements.
        screen=pool([(c['name'],'ordinary','nominal') for c in all_cases],'screen',finish-90*60)
        comparisons=compare_rows(screen,'ordinary');best=choose(comparisons)
        if best is not None:
            confirmation=pool([(c,'ordinary',v) for v in ('shift','repeat','long-hold') for c in ('sham',best['name'])],
                              'confirmation',finish-30*60)
            for variant in ('shift','repeat','long-hold'):confirmed+=compare_rows(confirmation,'ordinary',variant)
            refined=pool([(c,g,'nominal') for g in ('neural-half','body-half') for c in ('sham',best['name'])],
                         'refinement',finish-45)
            for grid in ('neural-half','body-half'):refcomparisons+=compare_rows(refined,grid)
    except Exception as e:failure=repr(e);print('SMD_CONTROL_ERROR:',failure,flush=True)
    verify_inputs(inputs)
    complete=bool(failure is None and len(screen)==len(all_cases) and len(confirmation)==6 and len(refined)==4
                  and len(confirmed)==3 and len(refcomparisons)==2
                  and all(r['exit_code']==0 and r['result'] for r in screen+confirmation+refined))
    checks=[]
    for row in refcomparisons:
        ordinary=next(r for r in comparisons if r['case']['name']==row['case']['name'])
        aa=ordinary['metrics']['off_windows'][0];bb=row['metrics']['off_windows'][0]
        delta=abs(aa['body_range_ratio']-bb['body_range_ratio'])
        nr0=aa['target_voltage_rms_ratio'];nr1=bb['target_voltage_rms_ratio']
        neural_error=abs(nr0-nr1) if nr0 is not None and nr1 is not None else None
        checks.append(dict(case=row['case']['name'],grid=row['grid'],body_ratio_difference=delta,
            neural_ratio_difference=neural_error,passed=bool(delta<=.05 and neural_error is not None and neural_error<=.05
                and row['metrics']['pre_match'] and row['metrics']['safe'])))
    selected=next((r for r in comparisons if best and r['case']['name']==best['name']),None)
    access=bool(complete and selected and all(r['metrics']['reversible_neural_target'] for r in [selected]+confirmed+refcomparisons)
                and len(checks)==2 and all(r['passed'] for r in checks))
    result=dict(experiment=experiment,checkpoint_sha256=SELECTED,execution_valid=complete,all_stages_finished=complete,validation_complete=complete,
        failure=failure,screen=screen,comparisons=comparisons,selected=best,confirmation=confirmation,confirmation_comparisons=confirmed,
        refinement=refined,refinement_comparisons=refcomparisons,numerical_checks=checks,reversible_smd_neural_access=access,
        learned_stopping_qualified=False,swimming_qualified=False,
        scope='Frozen neural inhibition diagnostic; no learned stopping, sensory policy or intrinsic oscillator proof')
    write(out/'summary.json',result)
    for r in comparisons+confirmed+refcomparisons:
        print(json.dumps(dict(case=r['case']['name'],variant=r['variant'],grid=r['grid'],metrics=r['metrics'])),flush=True)
    print('Report:',out/'summary.json',flush=True)


if __name__=='__main__':main()
