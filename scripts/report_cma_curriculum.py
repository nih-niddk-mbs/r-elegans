"""Read-only E077 progress and outcomes; no simulations or writes."""
import argparse,json
from pathlib import Path


def candidate_diagnostics(run):
    """Saved feature summaries suffice after excess raw trajectories are pruned."""
    rows=[]
    for path in sorted((run/'evaluations').glob('g-*-c-*/summary.json')):
        r=json.loads(path.read_text());m=r.get('metrics',{})
        rows.append(dict(name=path.parent.name,status=r.get('status'),loss=m.get('loss'),
            data_loss=m.get('data_loss'),constraint_loss=m.get('constraint_loss'),
            checkpoint_sha256=r.get('checkpoint_sha256'),
            safe=m.get('safe'),uncued=m.get('uncued_movement_preserved'),
            pre=m.get('pre_movement_preserved'),bends=m.get('bend_amplitude_preserved'),
            partial_pass=m.get('partial_cadence_passed'),ratios=m.get('cadence_ratios',[])))
    completed=[r for r in rows if r['status']=='completed']
    eligible=[r for r in completed if all(r[k] for k in ('safe','uncued','pre','bends'))]
    def best(items):
        candidates=[r for r in items if r['loss'] is not None]
        if not candidates:return None
        r=min(candidates,key=lambda r:r['loss'])
        return {k:v for k,v in r.items() if k!='ratios'}
    summary=dict(candidates=len(rows),completed=len(completed),eligible=len(eligible),
        failed_checks={k:sum(not bool(r[k]) for r in completed) for k in ('safe','uncued','pre','bends')},
        best_unconstrained=best(completed),best_eligible=best(eligible),cadence_ratio_ranges={})
    for condition in ('stop_only','stop_start','start_only'):
        for window in ('off','on'):
            key=condition+'/'+window
            values=[f for r in eligible for row in r['ratios'] if row['condition']==condition and row['window']==window for f in row['frequency_ratios'] if f is not None]
            if values:summary['cadence_ratio_ranges'][key]=dict(minimum=min(values),maximum=max(values))
    summary['generation_history']=[]
    history=run/'generation_history.jsonl'
    if history.exists():summary['generation_history']=[json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);p.add_argument('--diagnostics',action='store_true');a=p.parse_args()
    run=a.run or max((x for x in Path('/home/carsonc/r-elegans-results').glob('cma-cadence-*') if x.is_dir()),default=None)
    if run is None:p.error('No CMA cadence run found')
    print('Run:',run)
    if a.diagnostics:
        print(json.dumps(candidate_diagnostics(run),indent=2));return
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
