"""E093 movement suppression, recovery and audit report."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);p.add_argument('--details',action='store_true',help='Write separate read-only progress, bounds and sampled-current analysis');a=p.parse_args()
    run=a.run or max((p for p in Path('/home/carsonc/r-elegans-results').glob('sensory-motion-*') if (p/'configuration.json').is_file()),default=None)
    if run is None:raise SystemExit('No E093 run found')
    print('Run:',run)
    if not (run/'summary.json').exists():print('No final summary; inspect adjacent .log');return
    if a.details:
        from scripts.sensory_motion_details import analyze
        analyze(run);return
    s=json.loads((run/'summary.json').read_text())
    print({k:s.get(k) for k in ('execution_valid','validation_complete','completed_generations','target','stop_reason','failure','sensory_motion_component_qualified','rest_restart_component_qualified','full_start_stop_qualified')})
    for panel in s['panels']:
        print('Checkpoint:',panel['checkpoint_sha256'],'assessment:',panel['assessment'])
        for r in panel['validation']:
            print(r.get('variant'),r.get('grid'),r.get('mode'),r['status'],r.get('metrics'),r.get('mediation'))
    for path in (run/'budget_receipt.json',run.with_name(run.name+'-audit')/'summary.json'):
        print(str(path),json.loads(path.read_text()) if path.exists() else 'Pending')


if __name__=='__main__':main()
