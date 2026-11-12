"""Parallel independent E067 connection searches; no JAX in supervisor."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.benchmark_cpu import physical_core_cpus
from scripts.benchmark_cpu_parallel import worker_cpu_groups
from scripts.experiment_records import capture_run_record,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--expected-parent',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--minutes',type=float,default=240.)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--cores-per-worker',type=int,default=4)
    p.add_argument('--seed-start',type=int,default=6700)
    p.add_argument('--iterations',type=int,default=1000000)
    p.add_argument('--train-seconds',type=float,default=60.)
    p.add_argument('--window-seconds',type=float,default=15.)
    p.add_argument('--long-seconds',type=float,default=300.)
    p.add_argument('--software-smoke',action='store_true')
    p.add_argument('--no-affinity',action='store_true')
    a=p.parse_args()
    if (a.workers<1 or a.cores_per_worker<1 or a.seed_start<0 or a.iterations<1
            or not math.isfinite(a.minutes) or a.minutes<=0):p.error('Positive worker/core/budget settings required')
    repo=Path(__file__).resolve().parents[1];out=a.output_dir.resolve();source=a.source_run.resolve()
    for root in (repo,source):
        if out==root or root in out.parents or out in root.parents:p.error('Separate external output required')
    if a.no_affinity:
        groups=[None]*a.workers
    else:
        if not hasattr(os,'sched_getaffinity') or not shutil.which('taskset'):p.error('Linux affinity required, or explicit local --no-affinity')
        cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0)))
        if not known:p.error('Verified physical topology required')
        groups=worker_cpu_groups(cpus,a.workers,a.cores_per_worker)
    out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir()
    capture_run_record(out,repo,a)
    env=os.environ.copy();env.update(JAX_PLATFORMS='cpu',JAX_ENABLE_X64='false',JAX_DEFAULT_MATMUL_PRECISION='highest',
        OMP_NUM_THREADS=str(a.cores_per_worker),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    command=[sys.executable,'-u','-m','scripts.prepare_connection_amplitude','--source-run',str(source),
             '--expected-parent',a.expected_parent,'--output-dir',str(out/'inputs')]
    if a.software_smoke:command.append('--software-smoke')
    subprocess.run(command,cwd=repo,env=env,check=True)
    manifest_sha=file_sha256(out/'inputs/manifest.json')
    running={};finished=[];started=time.monotonic();heartbeat=started
    try:
        for slot,group in enumerate(groups):
            seed=a.seed_start+slot
            command=[sys.executable,'-u','-m','scripts.train_connection_amplitude',
                '--inputs',str(out/'inputs'),'--output-dir',str(out/f'seed-{seed}'),
                '--seed',str(seed),'--minutes',str(a.minutes),'--iterations',str(a.iterations),
                '--train-seconds',str(a.train_seconds),'--window-seconds',str(a.window_seconds),'--long-seconds',str(a.long_seconds)]
            if a.software_smoke:command.append('--software-smoke')
            if group is not None:command=['taskset','-c',','.join(map(str,group)),*command]
            log=(out/'logs'/f'seed-{seed}.log').open('w')
            try:process=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            except BaseException:log.close();raise
            running[seed]=(process,log,group)
            print(json.dumps(dict(starting_seed=seed,pid=process.pid,cpus=group)),flush=True)
        while running:
            for seed,(process,log,group) in list(running.items()):
                code=process.poll()
                if code is None:continue
                log.close();del running[seed]
                path=out/f'seed-{seed}/summary.json';result=json.loads(path.read_text()) if path.exists() else None
                finished.append(dict(seed=seed,cpus=group,exit_code=code,result=result))
                print(json.dumps(dict(finished_seed=seed,exit_code=code,passed=result['passed'] if result else None)),flush=True)
            if time.monotonic()-heartbeat>=30:
                progress={}
                for seed in running:
                    path=out/f'seed-{seed}/progress.json'
                    if path.exists():progress[str(seed)]=json.loads(path.read_text())
                print(json.dumps(dict(elapsed_minutes=(time.monotonic()-started)/60,running=list(running),progress=progress)),flush=True)
                heartbeat=time.monotonic()
            time.sleep(.5)
    finally:
        for process,log,_ in running.values():
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            log.close()
    valid=bool(len(finished)==a.workers and all(r['exit_code']==0 and r['result'] for r in finished)
               and file_sha256(out/'inputs/manifest.json')==manifest_sha)
    summary=dict(experiment='E067',status='completed',execution_valid=valid,requested_search_minutes=a.minutes,
        source_parent_sha256=a.expected_parent,workers=finished,elapsed_seconds=time.monotonic()-started,
        software_smoke=a.software_smoke,any_qualified=any(r['result'] and r['result']['passed'] for r in finished),
        scope='Independent fresh connection-only small amplitude rung; no biological/locomotion qualification')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    for r in sorted(finished,key=lambda r:r['seed']):
        s=r['result']
        print(json.dumps(dict(seed=r['seed'],exit_code=r['exit_code'],passed=s['passed'] if s else None,
            iterations=s['iterations'] if s else None,channels_frozen=s['channel_scales_unchanged'] if s else None,
            baseline_cycles=s['baseline']['minimum_cycle_excursions_rad'] if s else None,
            best_cycles=s['best']['metrics']['minimum_cycle_excursions_rad'] if s else None,
            validation=[dict(case=v['case'],cycles=v['metrics']['minimum_cycle_excursions_rad'],
                              neural_guard=v['metrics']['neural_guard'],target_passed=v['metrics']['target_passed']) for v in s['validation']] if s else None)),flush=True)
    print(json.dumps(dict(execution_valid=valid,any_qualified=summary['any_qualified'],report=str(out/'summary.json'))),flush=True)
    return 0 if valid else 2


if __name__=='__main__':raise SystemExit(main())
