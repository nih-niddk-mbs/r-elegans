"""Read-only E077 progress and outcomes; no simulations or writes."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);a=p.parse_args()
    run=a.run or max((x for x in Path('/home/carsonc/r-elegans-results').glob('cma-cadence-*') if x.is_dir()),default=None)
    if run is None:p.error('No CMA cadence run found')
    print('Run:',run)
    if (run/'budget_receipt.json').exists():
        b=json.loads((run/'budget_receipt.json').read_text());print('Elapsed minutes:',b['total_elapsed_seconds']/60);print(b)
    if not (run/'summary.json').exists():
        print('Final summary not written yet. Completed evaluations:')
        for path in sorted((run/'evaluations').glob('*/summary.json')):
            r=json.loads(path.read_text());print(path.parent.name,r['status'],r.get('metrics',{}).get('loss'))
        return
    s=json.loads((run/'summary.json').read_text());print({k:v for k,v in s.items() if k not in ('selected','refinement','milestones')})
    selected=s.get('selected')
    if selected:
        print('Selected:',selected['output'],'checkpoint:',selected.get('checkpoint_sha256'))
        print('Training target/metrics:',selected.get('metrics'))
    for r in s.get('refinement',[]):print('Refinement:',r['grid'],r['status'],r.get('metrics'))


if __name__=='__main__':main()
