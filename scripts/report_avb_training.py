"""Summarize E078 without loading trajectories or importing JAX."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path)
    a=p.parse_args();run=a.run
    if run is None:
        runs=[p for p in Path('/home/carsonc/r-elegans-results').glob('avb-training-*') if p.is_dir()]
        if not runs:raise SystemExit('No AVB training run found')
        run=max(runs,key=lambda p:p.name)
    print('Run:',run)
    if not (run/'summary.json').exists():
        print('Final summary not yet present; inspect the run log.');return
    s=json.loads((run/'summary.json').read_text())
    for key in ('completed_generations','stop_reason','failure','validation_complete','avb_component_qualified','sensory_start_stop_qualified','swimming_qualified'):
        print(key+':',s.get(key))
    for label,r in [('baseline',s.get('baseline')),('selected',s.get('selected'))]+[(v['variant']+'/'+v['grid'],v) for v in s['validation']]:
        if r:print(label,r.get('status'),r.get('metrics'),r.get('checkpoint_sha256'))
    print('Numerical checks:',s['numerical_checks'])
    if (run/'budget_receipt.json').exists():print('Total budget:',json.loads((run/'budget_receipt.json').read_text()))


if __name__=='__main__':main()
