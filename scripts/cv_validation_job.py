"""Read-only status and single-instance launcher for CV peptide validation jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

MODULES={'scripts.run_cv_peptide_validation','scripts.continue_cv_peptide_validation'}
BASE=Path('/home/carsonc/r-elegans-results')


def ancestor_pids(pid):
    result=set()
    while pid>1 and pid not in result:
        result.add(pid)
        value=subprocess.check_output(['ps','-o','ppid=','-p',str(pid)],text=True).strip()
        if not value:break
        pid=int(value)
    return result


def active_jobs(output=None,*,exclude_pid=None):
    """Use own-user ps only: no /proc access, PID guessing or broad pkill."""
    excluded={exclude_pid}
    if output is None:
        if exclude_pid is not None:excluded|=ancestor_pids(exclude_pid)
        output=subprocess.check_output(['ps','-ww','-u',str(os.getuid()),'-o','pid=,etime=,args='],text=True)
    jobs=[]
    for line in output.splitlines():
        parts=line.split(None,2)
        if len(parts)!=3:continue
        pid,elapsed,command=parts
        try:args=shlex.split(command)
        except ValueError:continue
        if int(pid) in excluded:continue
        modules=[args[i+1] for i,s in enumerate(args[:-1]) if s=='-m']
        wrapper='scripts.cv_validation_job' in modules and '--module' in args and not any(v in args for v in ('--status','--check'))
        if not (set(modules)&MODULES or wrapper):continue
        if '--report' in args:continue
        folder=args[args.index('--output-dir')+1] if '--output-dir' in args and args.index('--output-dir')+1<len(args) else '<unknown>'
        role='coordinate' if '--coordinate' in args else 'worker' if '--worker' in args else 'recover' if '--recover' in args else 'supervisor'
        jobs.append(dict(pid=int(pid),elapsed=elapsed,output_dir=folder,role=role))
    return jobs


def status(base=BASE):
    print('READ-ONLY STATUS — no job is started')
    jobs=active_jobs(exclude_pid=os.getpid())
    grouped={}
    for job in jobs:grouped.setdefault(job['output_dir'],[]).append(dict(pid=job['pid'],role=job['role'],elapsed=job['elapsed']))
    print('Active runs:',json.dumps(grouped))
    runs=sorted(set(base.glob('cv-peptide-validation-*'))|set(base.glob('cv-peptide-continuation-*')),reverse=True)
    for root in [p for p in runs if p.is_dir()][:4]:
        receipt=root/'budget_receipt.json'
        print(str(root),json.loads(receipt.read_text()) if receipt.exists() else 'No final receipt; see active processes and log')


def stop_run(root):
    root=Path(root).resolve()
    if root.parent!=BASE or not root.name.startswith(('cv-peptide-validation-','cv-peptide-continuation-')):
        raise ValueError('Explicit validation run directory under results required')
    def matching():return [r for r in active_jobs(exclude_pid=os.getpid()) if Path(r['output_dir']).resolve()==root]
    jobs=matching()
    if not jobs:
        print('No active process for this exact run; nothing stopped:',root);return 0
    coordinators=[r for r in jobs if r['role']=='coordinate']
    if not coordinators and all(r['role']=='supervisor' for r in jobs):
        print('Only supervisors remain; waiting briefly without sending a signal:',root,flush=True)
    else:
        if len(coordinators)!=1:raise ValueError('Cannot identify one private coordinator group; refusing to guess')
        pid=coordinators[0]['pid']
        if os.getpgid(pid)!=pid:raise ValueError('Coordinator is not a private process-group leader; nothing stopped')
        print('Stopping only the private worker group for:',root,flush=True)
        os.killpg(pid,signal.SIGTERM)
    # Keep the supervisor alive so it can audit partial outputs and write receipt.
    for _ in range(20):
        if not matching():
            print('Run stopped; original files preserved.');return 0
        time.sleep(1)
    print('Supervisor is still finalizing its audit. No replacement started.');return 3


def main():
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--status',action='store_true');modes.add_argument('--check',action='store_true');modes.add_argument('--stop-run',type=Path)
    p.add_argument('--module',choices=sorted(MODULES));p.add_argument('--minutes',type=int,default=60)
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.status:return status()
    if a.stop_run:return stop_run(a.stop_run)
    jobs=active_jobs(exclude_pid=os.getpid())
    if jobs:
        print('REFUSED: validation is already active. Nothing started.\n'+json.dumps(jobs),flush=True);return 3
    if a.check:
        print('No active validation processes detected; check only, nothing started.');return
    if not a.module or not 30<=a.minutes<=1440:p.error('Launch requires a module and30-1440total minutes')
    arguments=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
    if '--output-dir' not in arguments or any(v in arguments for v in ('--report','--worker','--coordinate','--recover')):
        p.error('Only top-level launch with --output-dir is allowed; reports do not use this launcher')
    BASE.mkdir(parents=True,exist_ok=True)
    # Kernel releases lock on process exit; old lock files do not imply active work.
    with (BASE/'.cv-peptide-validation.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('REFUSED: another validation launcher holds the lock. Nothing started.',flush=True);return 3
        jobs=active_jobs(exclude_pid=os.getpid())
        if jobs:
            print('REFUSED: active validation detected after locking. Nothing started.\n'+json.dumps(jobs),flush=True);return 3
        print('LAUNCHING one validation job; exclusive lock held until it exits.',flush=True)
        return subprocess.call(['timeout','--signal=TERM','--kill-after=5s',str(a.minutes*60-5)+'s',sys.executable,'-u','-m',a.module,*arguments])


if __name__=='__main__':raise SystemExit(main())
