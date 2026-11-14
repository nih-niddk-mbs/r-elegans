"""Run a private process group with a hard total wall-clock ceiling."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def run_group(command,seconds):
    process=subprocess.Popen(command,start_new_session=True)
    timed_out=False
    try:
        try:code=process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out=True;code=124
    finally:
        # Kill the private group, including orphaned descendants after leader exit.
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=2)
        except subprocess.TimeoutExpired:pass
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()
    return code,timed_out


def main():
    started=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--total-minutes',type=float,default=240.)
    p.add_argument('--source-run',type=Path)
    p.add_argument('--pipeline',choices=['tail','state-control','control-access'],default='tail')
    a=p.parse_args()
    if not math.isfinite(a.total_minutes) or a.total_minutes<5:p.error('At least five total minutes required')
    repo=Path(__file__).resolve().parents[1];out=a.output_dir.resolve()
    if out==repo or repo in out.parents or out in repo.parents:p.error('External output required')
    out.mkdir(parents=True,exist_ok=False)
    deadline=started+a.total_minutes*60
    # The coordinator finishes before the outer timeout; cleanup/reporting are inside total.
    budget=dict(experiment={'state-control':'E072-E073','control-access':'E076','tail':'E071'}[a.pipeline],total_minutes=a.total_minutes,started_utc=datetime.now(timezone.utc).isoformat(),
                hard_deadline_monotonic=deadline,coordinator_finish_by=deadline-60,
                phases_included=['preparation','baseline','search','all validation','cleanup','final reports'])
    (out/'deadline.json').write_text(json.dumps(budget,indent=2)+'\n')
    module={'state-control':'scripts.run_controlled_swimming','control-access':'scripts.run_control_access','tail':'scripts.run_swim_continuation'}[a.pipeline]
    command=[sys.executable,'-u','-m',module,'--output-dir',str(out)]
    if a.source_run:command+=['--source-run',str(a.source_run.resolve())]
    code,timed_out=run_group(command,max(.01,deadline-time.monotonic()-20))
    summary_path=out/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    result=dict(exit_code=code,timed_out=timed_out,total_elapsed_seconds=time.monotonic()-started,
                total_minutes=a.total_minutes,
                all_steps_finished=bool(code==0 and summary.get('execution_valid') and summary.get('all_stages_finished',True)),
                validation_complete=bool(summary.get('validation_complete',summary.get('execution_valid',False))),
                report=str(summary_path),scope='Timeout/incomplete validation never qualifies a model')
    (out/'budget_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
