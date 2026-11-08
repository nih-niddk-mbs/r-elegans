"""Bounded Linux CPU diagnostic queue, with explicit Python and disjoint cores.

No JAX imports in this supervisor. Launch under nohup; each worker captures its
environment/source and preserves streamed trajectories plus full state snapshots.
"""
import argparse
from datetime import datetime,timezone
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

if __package__:
    from .benchmark_cpu import physical_core_cpus
    from .benchmark_cpu_parallel import worker_cpu_groups
    from .experiment_records import capture_run_record
else:
    from benchmark_cpu import physical_core_cpus
    from benchmark_cpu_parallel import worker_cpu_groups
    from experiment_records import capture_run_record


def diagnostic_cases():
    cases=[dict(name='active-'+medium,medium=medium,condition='active',perturbation=0.,dt=.002,brain_step=.1)
           for medium in ('normalized','liquid','agar')]
    cases.extend([
        dict(name='liquid-perturb-plus',medium='liquid',condition='active',perturbation=1.,dt=.002,brain_step=.1),
        dict(name='liquid-perturb-minus',medium='liquid',condition='active',perturbation=-1.,dt=.002,brain_step=.1),
        dict(name='liquid-calcium-control',medium='liquid',condition='cca1_removed',perturbation=0.,dt=.002,brain_step=.1),
        dict(name='liquid-held-voltage',medium='liquid',condition='held_voltage',perturbation=0.,dt=.002,brain_step=.1),
        dict(name='liquid-body-refined',medium='liquid',condition='active',perturbation=0.,dt=.001,brain_step=.1),
        dict(name='liquid-brain-refined',medium='liquid',condition='active',perturbation=0.,dt=.002,brain_step=.05),
    ])
    return cases


def verify_input_bundle(root):
    manifest=json.loads((root/'inputs.json').read_text())
    for name in ('network.npz','parent_actor.npz'):
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=manifest['sha256'][name]:
            raise ValueError('Input checksum mismatch: '+name)
    return manifest


def select_input_bundle(inputs=None, parent_actor=None):
    """Use the pinned Mac reference unless the caller explicitly overrides it."""
    if inputs is not None:return Path(inputs).expanduser().resolve()
    if parent_actor is not None:return None
    from importlib.resources import files
    return Path(str(files('r_elegans.assets').joinpath('mac_rhythm_v1')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--inputs',type=Path,help='Exact input pair; default is the bundled Mac-tested reference')
    source.add_argument('--parent-actor',type=Path,help='Explicitly rebuild from this existing full-network actor instead of the bundled reference')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--minutes',type=float,default=45.)
    parser.add_argument('--case-minutes',type=float,default=12.)
    parser.add_argument('--duration',type=float,default=600.)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--cores-per-worker',type=int,default=4)
    args=parser.parse_args()
    if sys.platform!='linux' or not hasattr(os,'sched_getaffinity'):parser.error('Run this supervisor on Linux')
    if not all(math.isfinite(x) and x>0 for x in (args.minutes,args.case_minutes,args.duration)):
        parser.error('Finite positive time budgets required')
    repo=Path(__file__).resolve().parents[1];base=args.output_dir.expanduser().resolve()
    if base==repo or repo in base.parents:parser.error('Keep results outside the repository')
    inputs=select_input_bundle(args.inputs,args.parent_actor)
    if inputs:verify_input_bundle(inputs)
    else:
        parent=args.parent_actor.expanduser().resolve()
    # Catch the original /usr/bin/python/no-JAX failure before launching jobs.
    check=subprocess.run([sys.executable,'-c','import jax, numpy, optax'],capture_output=True,text=True)
    if check.returncode:parser.error('Activate r-elegans or use its absolute Python path.\n'+check.stderr)
    cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0)))
    if not known:parser.error('Physical CPU topology is required for disjoint allocations')
    try:groups=worker_cpu_groups(cpus,args.workers,args.cores_per_worker)
    except ValueError as e:parser.error(str(e))
    run=base/('embodied-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run.mkdir(parents=True,exist_ok=False);capture_run_record(run,repo,args)
    print('Run: '+str(run),flush=True)
    print('Python: '+sys.executable,flush=True)
    if inputs is not None:
        print('Using exact inputs: '+str(inputs),flush=True)
        shutil.copytree(inputs,run/'inputs')
        inputs=run/'inputs'
        print(json.dumps(verify_input_bundle(inputs)),flush=True)
    if inputs is None:
        inputs=run/'inputs'
        env=os.environ.copy();env.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',
            JAX_DEFAULT_MATMUL_PRECISION='highest',OMP_NUM_THREADS=str(len(groups[0])),OPENBLAS_NUM_THREADS='1')
        print('Rebuilding channel candidate from: '+str(parent),flush=True)
        result=subprocess.run([sys.executable,'-u','-m','scripts.bootstrap_embodied_inputs',
            '--parent-actor',str(parent),'--output-dir',str(inputs)],cwd=repo,env=env,
            preexec_fn=lambda:os.sched_setaffinity(0,groups[0]),timeout=min(120.,args.minutes*60.))
        if result.returncode:return result.returncode
        verify_input_bundle(inputs)
    pending=diagnostic_cases();planned=[c['name'] for c in pending];finished=[];active=[]
    start=time.monotonic();stopped=False;budget=False;last_status=start
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with ExitStack() as files:
        try:
            while pending or active:
                now=time.monotonic();budget=now-start>=args.minutes*60.
                busy={x['slot'] for x in active}
                for slot,selected in enumerate(groups):
                    if not pending or stopped or budget:break
                    if slot in busy:continue
                    case=pending.pop(0);output=run/case['name']
                    command=[sys.executable,'-u','-m','scripts.diagnose_channel_body',
                        '--network',str(inputs/'network.npz'),'--parent-actor',str(inputs/'parent_actor.npz'),
                        '--output-dir',str(output),'--medium',case['medium'],'--condition',case['condition'],
                        '--perturbation-mv',str(case['perturbation']),'--duration',str(args.duration),
                        '--minutes',str(min(args.case_minutes,max(.01,args.minutes-(now-start)/60.))),
                        '--dt',str(case['dt']),'--max-brain-step-ms',str(case['brain_step'])]
                    env=os.environ.copy();env.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',
                        JAX_DEFAULT_MATMUL_PRECISION='highest',OMP_NUM_THREADS=str(len(selected)),
                        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
                    log=files.enter_context((run/(case['name']+'.log')).open('w'))
                    process=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,
                        preexec_fn=lambda selected=selected:os.sched_setaffinity(0,selected))
                    active.append(dict(case=case,slot=slot,cpus=selected,process=process,output=output,
                                       started=time.monotonic(),termination_sent=None))
                    print(json.dumps(dict(starting=case['name'],pid=process.pid,cpus=selected)),flush=True)
                for job in list(active):
                    process=job['process']
                    if process.poll() is not None:
                        result=json.loads((job['output']/'summary.json').read_text()) if (job['output']/'summary.json').exists() else dict(status='missing_summary')
                        finished.append(dict(case=job['case']['name'],cpu_affinity=job['cpus'],exit_code=process.returncode,result=result))
                        print(json.dumps(dict(finished=job['case']['name'],exit_code=process.returncode,status=result['status'])),flush=True)
                        active.remove(job);continue
                    overrun=now-job['started']>(args.case_minutes*60.+60.)
                    if (budget or stopped or overrun) and job['termination_sent'] is None:
                        process.terminate();job['termination_sent']=now
                    elif job['termination_sent'] is not None and now-job['termination_sent']>45.:process.kill()
                snapshot=dict(experiment='E028',requested_minutes=args.minutes,elapsed_seconds=time.monotonic()-start,
                    planned_cases=planned,finished=finished,running=[j['case']['name'] for j in active],
                    pending=[c['name'] for c in pending],wall_budget_reached=budget,interrupted=stopped)
                temp=run/'summary.json.tmp';temp.write_text(json.dumps(snapshot,indent=2,allow_nan=False)+'\n');temp.replace(run/'summary.json')
                if now-last_status>=30.:
                    print(json.dumps(dict(elapsed_minutes=(now-start)/60.,running=snapshot['running'],
                        completed=len(finished),pending=len(pending))),flush=True)
                    last_status=now
                if (budget or stopped) and not active:break
                time.sleep(1.)
        finally:
            for job in active:
                if job['process'].poll() is None:job['process'].terminate()
            for job in active:
                try:job['process'].wait(timeout=45)
                except subprocess.TimeoutExpired:job['process'].kill();job['process'].wait()
    failed=stopped or not finished or any(r['exit_code']!=0 for r in finished)
    print('EMBODIED_DIAGNOSTIC_EXIT_CODE='+str(2 if failed else 0),flush=True)
    return 2 if failed else 0


if __name__=='__main__':raise SystemExit(main())
