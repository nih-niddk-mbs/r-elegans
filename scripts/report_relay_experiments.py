"""Report E092 experiments and their automatic audit."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);a=p.parse_args()
    run=a.run or max((p for p in Path('/home/carsonc/r-elegans-results').glob('relay-experiments-*') if p.is_dir() and not p.name.endswith('-audit')),default=None)
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
    for path in (run/'budget_receipt.json',run.with_name(run.name+'-audit')/'summary.json'):
        if path.exists():print(path.name,json.loads(path.read_text()))
        else:print('Pending:',path)


if __name__=='__main__':main()
