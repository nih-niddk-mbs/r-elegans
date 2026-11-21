"""Hard-budget E106 audit wrapper. Sources are never overwritten."""
import argparse,json,sys,time
from pathlib import Path
from scripts.experiment_records import capture_run_record
from scripts.selected_atlas_audit_checks import write
from scripts.total_job_budget import run_group


def main():
    started=time.monotonic();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--e104-sweep',type=Path,required=True);p.add_argument('--e105-sweep',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--minutes',type=float,default=120.)
    a=p.parse_args();out=a.output_dir.resolve();repo=Path(__file__).resolve().parents[1]
    if not 5<=a.minutes<=1440:p.error('Audit wall ceiling must be between5and1440minutes')
    for root in (repo,a.e104_sweep.resolve(),a.e105_sweep.resolve(),a.data_root.resolve()):
        if root==out or root in out.parents or out in root.parents:p.error('Separate external audit folder required')
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,repo,a)
    write(out/'deadline.json',dict(experiment='E106',total_minutes=a.minutes,hard_finish_monotonic=started+a.minutes*60))
    command=[sys.executable,'-u','-m','scripts.run_selected_atlas_audit','--e104-sweep',str(a.e104_sweep.resolve()),'--e105-sweep',str(a.e105_sweep.resolve()),'--data-root',str(a.data_root.resolve()),'--output-dir',str(out)]
    code,timeout=run_group(command,max(.01,started+a.minutes*60-time.monotonic()-20))
    path=out/'summary.json';summary=json.loads(path.read_text()) if path.exists() else {}
    receipt=dict(exit_code=code,timed_out=timeout,audit_complete=bool(code==0 and not timeout and summary.get('audit_complete')),total_elapsed_seconds=time.monotonic()-started,total_minutes=a.minutes,
        model_promoted=False,scope='Audit completion does not imply numerical, rhythm, biological or behavior acceptance; no training')
    write(out/'budget_receipt.json',receipt);print(json.dumps(receipt),flush=True);return code


if __name__=='__main__':raise SystemExit(main())
