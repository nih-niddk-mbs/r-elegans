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
    p.add_argument('--pipeline',choices=['tail','state-control','control-access','cma-curriculum','avb-training','nmj-source','head-split','smd-control','rial-control','rial-sensory','rial-release','relay-range'],default='tail')
    a=p.parse_args()
    if not math.isfinite(a.total_minutes) or a.total_minutes<5:p.error('At least five total minutes required')
    repo=Path(__file__).resolve().parents[1];out=a.output_dir.resolve()
    if out==repo or repo in out.parents or out in repo.parents:p.error('External output required')
    out.mkdir(parents=True,exist_ok=False)
    deadline=started+a.total_minutes*60
    automatic_audit=a.pipeline in ('rial-release','relay-range')
    audit_reserve=min(20*60,.15*a.total_minutes*60) if automatic_audit else 0
    # The coordinator finishes before the outer timeout; cleanup/reporting are inside total.
    budget=dict(experiment={'state-control':'E072-E073','control-access':'E076','cma-curriculum':'E077','avb-training':'E078','nmj-source':'E080','head-split':'E081','smd-control':'E082','rial-control':'E084','rial-sensory':'E085','rial-release':'E086','relay-range':'E087','tail':'E071'}[a.pipeline],total_minutes=a.total_minutes,started_utc=datetime.now(timezone.utc).isoformat(),
                hard_deadline_monotonic=deadline,coordinator_finish_by=deadline-audit_reserve-60,
                phases_included=['preparation','baseline','search','all validation','cleanup','final reports'])
    if automatic_audit:
        budget['phases_included'].append('complete saved-data post-run audit')
        budget['audit_reserve_seconds']=audit_reserve
    (out/'deadline.json').write_text(json.dumps(budget,indent=2)+'\n')
    module={'state-control':'scripts.run_controlled_swimming','control-access':'scripts.run_control_access','cma-curriculum':'scripts.run_cma_sensorimotor','avb-training':'scripts.run_avb_training','nmj-source':'scripts.run_nmj_source','head-split':'scripts.run_nmj_source','smd-control':'scripts.run_smd_control','rial-control':'scripts.run_rial_control','rial-sensory':'scripts.run_rial_sensory','rial-release':'scripts.run_rial_sensory','relay-range':'scripts.run_rial_sensory','tail':'scripts.run_swim_continuation'}[a.pipeline]
    command=[sys.executable,'-u','-m',module,'--output-dir',str(out)]
    if a.pipeline=='head-split':command+=['--protocol','head-split']
    if automatic_audit:command+=['--objective',a.pipeline]
    if a.source_run:command+=['--source-run',str(a.source_run.resolve())]
    code,timed_out=run_group(command,max(.01,deadline-audit_reserve-time.monotonic()-20))
    summary_path=out/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    audit_result={}
    if automatic_audit:
        audit_out=out.with_name(out.name+'-audit')
        audit_code,audit_timeout=run_group([sys.executable,'-u','-m','scripts.audit_rial_sensory',
            '--run',str(out),'--output-dir',str(audit_out)],max(.01,deadline-time.monotonic()-20))
        audit_summary=json.loads((audit_out/'summary.json').read_text()) if (audit_out/'summary.json').exists() else {}
        audit_result=dict(audit_exit_code=audit_code,audit_timed_out=audit_timeout,
            audit_complete=bool(audit_code==0 and audit_summary.get('audit_complete')),
            audit_integrity_passed=bool(audit_summary.get('integrity_passed')),audit_report=str(audit_out/'summary.json'))
        if code==0:code=audit_code
        timed_out=timed_out or audit_timeout
    result=dict(exit_code=code,timed_out=timed_out,total_elapsed_seconds=time.monotonic()-started,
                total_minutes=a.total_minutes,
                all_steps_finished=bool(code==0 and summary.get('execution_valid') and summary.get('all_stages_finished',True) and (not automatic_audit or audit_result['audit_complete'])),
                validation_complete=bool(summary.get('validation_complete',summary.get('execution_valid',False))),
                report=str(summary_path),scope='Timeout/incomplete validation never qualifies a model')
    result.update(audit_result)
    if automatic_audit:
        result['post_audit_component_qualified']=bool(result['all_steps_finished'] and
            result['validation_complete'] and result['audit_integrity_passed'] and
            summary.get('sensory_relay_range_component_qualified' if a.pipeline=='relay-range' else 'sensory_RIAL_release_component_qualified'))
    (out/'budget_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
