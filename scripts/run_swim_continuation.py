"""E071: verified E070 seed6802 continuation inside one absolute deadline."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from r_elegans.wall_budget import check_deadline,validation_reserve
from scripts.experiment_records import capture_run_record,file_sha256


def verify_parent(run):
    summary=json.loads((run/'summary.json').read_text())
    worker=run/'seed-6802';s=json.loads((worker/'summary.json').read_text())
    if (summary['experiment']!='E070' or not summary['execution_valid'] or summary['software_smoke']
            or s['status']!='completed' or s['seed']!=6802 or not s['source_inputs_unchanged']
            or not s['all_fixed_fields_unchanged'] or not s['held_voltage_control_passed']
            or not all(r['passed'] for r in s['refinement'])):
        raise ValueError('Completed numerically checked E070 seed6802 required')
    rows={r['case']:r['metrics'] for r in s['validation']}
    for name,row in rows.items():
        if name!='held-voltage' and not all(row[k] for k in ('neural_guard','safety_passed','timing_target_passed')):
            raise ValueError('Parent active guard failed: '+name)
    ordinary=rows['ordinary']
    if abs(ordinary['minimum_cycle_excursions_rad'][0]-.0027459793917179176)>1e-9:
        raise ValueError('Parent does not match the reported6802 tail amplitude')
    if file_sha256(worker/'network_best.npz')!=s['checkpoint_sha256']:
        raise ValueError('Selected checkpoint hash mismatch')
    selected=worker/'selected'/f"iteration-{s['selected_iteration']:06d}"
    if file_sha256(selected/'network.npz')!=s['checkpoint_sha256']:
        raise ValueError('Selected lineage checkpoint mismatch')
    return worker,selected,s


def find_parent(base):
    errors=[]
    for run in sorted((p for p in base.glob('neural-timing-*') if p.is_dir()),reverse=True):
        try:verify_parent(run);return run
        except (KeyError,ValueError,OSError) as error:errors.append(str(run)+': '+str(error))
    raise ValueError('No completed matching E070 seed6802; no random fallback. '+str(errors))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--source-run',type=Path)
    a=p.parse_args();out=a.output_dir.resolve();repo=Path(__file__).resolve().parents[1]
    budget=json.loads((out/'deadline.json').read_text());finish=budget['coordinator_finish_by']
    check_deadline(finish)
    source=a.source_run.resolve() if a.source_run else find_parent(Path('/home/carsonc/r-elegans-results'))
    if out==source or source in out.parents or out in source.parents:raise ValueError('Separate output required')
    worker,selected,parent=verify_parent(source)
    reserve=validation_reserve(parent['total_elapsed_seconds']-parent['search_elapsed_seconds'])
    if finish-time.monotonic()<reserve+15*60:raise ValueError('Insufficient total budget for setup and full validation reserve')
    capture_run_record(out,repo,a)
    original_manifest=source/'inputs/manifest.json';original_sha=file_sha256(original_manifest)
    shutil.copytree(source/'inputs',out/'inputs')
    root=out/'inputs';manifest=json.loads((root/'manifest.json').read_text())
    from scripts.compare_rhythm_worker import verify_inputs
    verify_inputs(root)
    copies={'continuation_parent.npz':worker/'network_best.npz','parent_best_state.npz':worker/'best_state.npz',
        'parent_summary.json':worker/'summary.json','parent_configuration.json':worker/'configuration.json',
        'parent_training_trace.npz':selected/'trajectory.npz','parent_selected_rng.json':selected/'rng_state.json',
        'parent_selected_metrics.json':selected/'metrics.json','parent_coordinate_reference.npz':worker/'coordinate_reference.npz',
        'parent_final_optimizer.npz':worker/'optimizer_state.npz','parent_final_rng.json':worker/'rng_state.json'}
    before={str(path):file_sha256(path) for path in copies.values()}
    shutil.copy2(root/'manifest.json',root/'inherited_manifest.json')
    manifest['sha256']['inherited_manifest.json']=file_sha256(root/'inherited_manifest.json')
    manifest['sources']['inherited_manifest.json']=str(original_manifest)
    for name,path in copies.items():
        shutil.copy2(path,root/name)
        if file_sha256(root/name)!=before[str(path)]:raise ValueError('Copied parent input mismatch')
        manifest['sha256'][name]=before[str(path)];manifest['sources'][name]=str(path)
    manifest.update(consumer_experiment='E071',continuation_parent_sha256=parent['checkpoint_sha256'],
                    continuation_source_run=str(source),continuation_source_worker=str(worker))
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    capture_run_record(root,repo,a,phase='continuation_snapshot')
    from scripts.benchmark_cpu import physical_core_cpus
    from scripts.benchmark_cpu_parallel import worker_cpu_groups
    if not hasattr(os,'sched_getaffinity') or not shutil.which('taskset'):raise ValueError('Linux CPU affinity required')
    cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0)))
    if not known:raise ValueError('Physical CPU topology unknown')
    groups=worker_cpu_groups(cpus,4,4);(out/'logs').mkdir()
    settings=dict(experiment='E071',source_run=str(source),parent_sha256=parent['checkpoint_sha256'],
        validation_reserve_seconds=reserve,finish_by_monotonic=finish,seeds=list(range(6900,6904)),
        total_minutes=budget['total_minutes'],fresh_optimizer=True,source_manifest_sha256=original_sha)
    (out/'configuration.json').write_text(json.dumps(settings,indent=2)+'\n')
    print(json.dumps(dict(selected_parent=str(worker),checkpoint_sha256=parent['checkpoint_sha256'],
                         validation_reserved_minutes=reserve/60,total_minutes=budget['total_minutes'])),flush=True)
    running={};finished=[];heartbeat=time.monotonic()
    env=os.environ.copy();env.update(JAX_PLATFORMS='cpu',JAX_ENABLE_X64='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='1')
    try:
        for seed,group in zip(range(6900,6904),groups):
            command=['taskset','-c',','.join(map(str,group)),sys.executable,'-u','-m','scripts.train_connection_amplitude',
                '--inputs',str(root),'--output-dir',str(out/f'seed-{seed}'),'--neural-timing','--tail-continuation',
                '--seed',str(seed),'--sigma','.035','--minutes',str(budget['total_minutes']),
                '--finish-by',str(finish),'--validation-reserve-seconds',str(reserve)]
            log=(out/'logs'/f'seed-{seed}.log').open('w')
            proc=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            running[seed]=(proc,log,group)
            print(json.dumps(dict(starting_seed=seed,pid=proc.pid,cpus=group)),flush=True)
        while running:
            for seed,(proc,log,group) in list(running.items()):
                if proc.poll() is None:continue
                log.close();del running[seed]
                path=out/f'seed-{seed}/summary.json';s=json.loads(path.read_text()) if path.exists() else None
                finished.append(dict(seed=seed,exit_code=proc.returncode,cpus=group,result=s))
                print(json.dumps(dict(finished_seed=seed,exit_code=proc.returncode,passed=s.get('passed') if s else None)),flush=True)
            if time.monotonic()-heartbeat>=30:
                progress={}
                for seed in running:
                    path=out/f'seed-{seed}/progress.json'
                    if path.exists():progress[seed]=json.loads(path.read_text())
                print(json.dumps(dict(minutes_remaining=(finish-time.monotonic())/60,progress=progress)),flush=True);heartbeat=time.monotonic()
            check_deadline(finish)
            if running:time.sleep(.5)
    finally:
        for proc,log,_ in running.values():
            if proc.poll() is None:proc.terminate()
        for proc,log,_ in running.values():
            try:proc.wait(timeout=2)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()
    verify_inputs(root)
    unchanged=before=={str(path):file_sha256(path) for path in copies.values()} and file_sha256(original_manifest)==original_sha
    valid=bool(len(finished)==4 and unchanged and all(r['exit_code']==0 and r['result'] and r['result'].get('validation_complete') for r in finished))
    summary=dict(experiment='E071',status='completed' if valid else 'incomplete_or_failed',execution_valid=valid,
                 parent_sha256=parent['checkpoint_sha256'],source_run=str(source),workers=finished,
                 source_files_unchanged=unchanged,total_minutes=budget['total_minutes'],validation_reserved_seconds=reserve,
                 any_qualified=any(r['result'] and r['result'].get('passed') for r in finished))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    for row in finished:
        s=row['result'] or {}
        print(json.dumps(dict(seed=row['seed'],exit_code=row['exit_code'],passed=s.get('passed'),validation_complete=s.get('validation_complete'),
            best_cycles=s.get('best',{}).get('metrics',{}).get('minimum_cycle_excursions_rad'),checkpoint_sha256=s.get('checkpoint_sha256'))),flush=True)
    print(json.dumps(dict(execution_valid=valid,report=str(out/'summary.json'))),flush=True)
    return 0 if valid else 124


if __name__=='__main__':raise SystemExit(main())
