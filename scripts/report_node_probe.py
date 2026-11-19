"""E096 imposed-conductance dose-response report."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);a=p.parse_args()
    run=a.run or max((q for q in Path('/home/carsonc/r-elegans-results').glob('node-probe-*') if (q/'configuration.json').is_file()),default=None)
    if run is None:raise SystemExit('No E096 run found')
    print('Run:',run)
    if not (run/'summary.json').exists():print('No final summary; inspect adjacent .log');return
    s=json.loads((run/'summary.json').read_text())
    print({k:s.get(k) for k in ('execution_valid','validation_complete','stop_reason','failure','evaluation_count')})
    print('Minimum tested dose (nS) reaching 1% in every region with baseline/recovery preserved:',s['minimum_doses_reaching_target_nS'])
    if s.get('cue_reduction_percent') is not None:print('Actual cue run reduction % (tail/middle/head):',[round(x,4) for x in s['cue_reduction_percent']])
    print('group, sign, dose nS, reduction% tail/middle/head, safe, baseline+recovery, target reached')
    for t in s['probe_table']:
        if t['status']!='completed':print(t['group'],t['sign'],t['dose_nS'],t.get('grid',''),t['status']);continue
        print(t['group'],t['sign'],t['dose_nS'],t.get('grid',''),[round(x,3) for x in t['reduction_percent']],t['safe'],t['baseline_and_recovery_preserved'],t['target_reached'])
    for path in (run/'budget_receipt.json',run.with_name(run.name+'-audit')/'summary.json'):
        print(str(path),json.loads(path.read_text()) if path.exists() else 'Pending')


if __name__=='__main__':main()
