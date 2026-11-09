"""Disjoint CPU searches for autonomous amplitude or motion, with bounded time."""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.benchmark_cpu import physical_core_cpus
from scripts.benchmark_cpu_parallel import worker_cpu_groups
from scripts.experiment_records import capture_run_record
from scripts.run_embodied_diagnostics import select_input_bundle,verify_input_bundle
from scripts.autonomous_warm_start import read_warm_start


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--inputs',type=Path)
    source.add_argument('--warm-start-workers',nargs='+',type=Path,help='Completed parents, round-robin; retain original anchors/bounds')
    parser.add_argument('--seed-start',type=int,default=0)
    parser.add_argument('--minutes',type=float,default=60.)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--cores-per-worker',type=int,default=4)
    parser.add_argument('--target-p2p-rad',type=float,default=.01)
    parser.add_argument('--objective',choices=['amplitude','movement','body-wave','cycle-movement','body-amplitude'],default='amplitude')
    parser.add_argument('--fit-scope',choices=['rmd','neurons-connections'])
    parser.add_argument('--target-speed',type=float,default=1e-4)
    args=parser.parse_args()
    if args.seed_start<0:parser.error('Nonnegative seed-start required')
    if args.fit_scope is None:args.fit_scope='rmd' if args.objective=='amplitude' else 'neurons-connections'
    parents=[]
    if args.warm_start_workers:
        if args.objective!='body-amplitude' or args.fit_scope!='neurons-connections':parser.error('Warm forks require joint body-amplitude objective')
        try:parents=[Path(read_warm_start(p)['parent_path']) for p in args.warm_start_workers]
        except (ValueError,OSError,KeyError) as e:parser.error(str(e))
    if sys.platform!='linux' or not hasattr(os,'sched_getaffinity'):parser.error('Linux CPU affinity required')
    if not all(math.isfinite(x) and x>0 for x in (args.minutes,args.target_p2p_rad,args.target_speed)):parser.error('Positive finite settings required')
    repo=Path(__file__).resolve().parents[1];base=args.output_dir.expanduser().resolve()
    if base==repo or repo in base.parents:parser.error('Keep results external')
    inputs=parents[0]/'source_inputs' if parents else select_input_bundle(args.inputs)
    if not parents and args.objective!='amplitude' and args.inputs is None:inputs=inputs.parent/'autonomous_wiggle_v1'
    manifest=verify_input_bundle(inputs)
    if args.objective=='cycle-movement' and not (args.inputs is not None and manifest.get('objective')=='body-wave' and manifest.get('passed') is True):
        parser.error('cycle-movement requires --inputs from a passed body-wave fitted_inputs checkpoint')
    cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0)))
    if not known:parser.error('Physical topology required')
    try:groups=worker_cpu_groups(cpus,args.workers,args.cores_per_worker)
    except ValueError as e:parser.error(str(e))
    check=subprocess.run([sys.executable,'-c','import jax, numpy, scipy'],capture_output=True,text=True)
    if check.returncode:parser.error('Use the dedicated r-elegans Python.\n'+check.stderr)
    prefix={'amplitude':'autonomous-wiggle-','movement':'autonomous-movement-',
        'body-wave':'autonomous-wave-','cycle-movement':'autonomous-cycle-movement-',
        'body-amplitude':'autonomous-body-amplitude-'}[args.objective]
    run=base/(prefix+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run.mkdir(parents=True,exist_ok=False);capture_run_record(run,repo,args)
    print('Run: '+str(run),flush=True)
    print('Fit scope: '+args.fit_scope,flush=True)
    print('Objective: '+args.objective,flush=True)
    jobs=[];stopped=False;start=time.monotonic()
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with ExitStack() as files:
        try:
            for slot,selected in enumerate(groups):
                seed=args.seed_start+slot
                env=os.environ.copy();env.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',
                    JAX_DEFAULT_MATMUL_PRECISION='highest',OMP_NUM_THREADS=str(len(selected)),
                    OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
                command=[sys.executable,'-u','-m','scripts.train_autonomous_wiggle','--output-dir',str(run/f'seed-{seed}'),
                    '--seed',str(seed),'--minutes',str(args.minutes),'--target-p2p-rad',str(args.target_p2p_rad),
                    '--objective',args.objective,'--fit-scope',args.fit_scope,'--target-speed',str(args.target_speed)]
                parent=parents[slot%len(parents)] if parents else None
                if parent:command.extend(['--warm-start-worker',str(parent)])
                else:command.extend(['--inputs',str(inputs)])
                log=files.enter_context((run/f'seed-{seed}.log').open('w'))
                process=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,
                    preexec_fn=lambda selected=selected:os.sched_setaffinity(0,selected))
                jobs.append(dict(seed=seed,cpus=selected,process=process,terminated=None,parent=str(parent) if parent else None))
                print(json.dumps(dict(seed=seed,pid=process.pid,cpus=selected,parent=str(parent) if parent else None,log=str(run/f'seed-{seed}.log'))),flush=True)
            last=start
            while any(j['process'].poll() is None for j in jobs):
                now=time.monotonic()
                for job in jobs:
                    p=job['process']
                    if p.poll() is not None:continue
                    if (stopped or now-start>args.minutes*60.+60.) and job['terminated'] is None:
                        p.terminate();job['terminated']=now
                    elif job['terminated'] is not None and now-job['terminated']>45.:p.kill()
                if now-last>30.:
                    print(json.dumps(dict(elapsed_minutes=(now-start)/60.,running=[j['seed'] for j in jobs if j['process'].poll() is None])),flush=True);last=now
                time.sleep(1.)
        finally:
            for job in jobs:
                if job['process'].poll() is None:job['process'].terminate()
            for job in jobs:
                try:job['process'].wait(timeout=45.)
                except subprocess.TimeoutExpired:job['process'].kill();job['process'].wait()
    workers=[]
    for job in jobs:
        path=run/f"seed-{job['seed']}/summary.json"
        result=json.loads(path.read_text()) if path.exists() else dict(status='missing_summary',passed=False)
        workers.append(dict(seed=job['seed'],cpus=job['cpus'],parent=job['parent'],exit_code=job['process'].returncode,result=result))
    experiment={'body-wave':'E035','cycle-movement':'E036','body-amplitude':'E037'}.get(args.objective,
        'E032' if args.fit_scope=='neurons-connections' else ('E031' if args.objective=='movement' else 'E030'))
    summary=dict(experiment=experiment,
        objective=args.objective,fit_scope=args.fit_scope,
        workers=workers,elapsed_seconds=time.monotonic()-start,interrupted=stopped)
    (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    passed=not stopped and all(w['exit_code']==0 and w['result']['passed'] for w in workers)
    label={'body-wave':'AUTONOMOUS_WAVE','cycle-movement':'AUTONOMOUS_CYCLE_MOVEMENT',
        'movement':'AUTONOMOUS_MOVEMENT','amplitude':'AUTONOMOUS_WIGGLE','body-amplitude':'AUTONOMOUS_BODY_AMPLITUDE'}[args.objective]
    print(label+'_EXIT_CODE='+str(0 if passed else 2),flush=True)
    return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
