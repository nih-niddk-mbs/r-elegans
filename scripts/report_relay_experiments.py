"""Report E092 experiments and their automatic audit."""
import argparse,json
from pathlib import Path


def latest_run(base):
    return max((p for p in base.glob('relay-experiments-*')
                if p.is_dir() and (p/'configuration.json').is_file()),default=None)


def audit_path(run):
    # A failed/incomplete newer re-audit remains visible; never fall back to a pass.
    candidates=list(run.parent.glob(run.name+'-audit-*'))
    candidates=[p for p in candidates if p.is_dir()]
    return (max(candidates) if candidates else run.with_name(run.name+'-audit'))/'summary.json'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);a=p.parse_args()
    run=a.run or latest_run(Path('/home/carsonc/r-elegans-results'))
    if run is None:raise SystemExit('No E092 run found')
    print('Run:',run)
    if not (run/'summary.json').exists():print('No final summary; inspect adjacent .log');return
    s=json.loads((run/'summary.json').read_text())
    print({k:s.get(k) for k in ('execution_valid','validation_complete','completed_matched_generations','stop_reason','failure')})
    for r in s['scan']:
        m=r.get('metrics',{});print(Path(r['output']).name,r['status'],dict(cap=r.get('cap'),loss=m.get('loss'),range=m.get('relay_operating_range_passed'),cue_changes=m.get('relay_response_mean_changes')))
    for name,a in s['arms'].items():
        r=a.get('primary') or {};m=r.get('metrics',{});print(name,'equal-cohort winner:',m.get('loss'),r.get('checkpoint_sha256'))
    for panel in s['panels']:print('Validation:',panel['labels'],panel['cap'],panel['assessment'])
    print('Original budget receipt is historical; corrected audit reports separately.')
    for path in (run/'budget_receipt.json',audit_path(run)):
        if path.exists():print(str(path),json.loads(path.read_text()))
        else:print('Pending:',path)


if __name__=='__main__':main()
