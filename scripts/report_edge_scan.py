"""E095 single-edge dose-response report."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);a=p.parse_args()
    run=a.run or max((q for q in Path('/home/carsonc/r-elegans-results').glob('edge-scan-*') if (q/'configuration.json').is_file()),default=None)
    if run is None:raise SystemExit('No E095 run found')
    print('Run:',run)
    if not (run/'summary.json').exists():print('No final summary; inspect adjacent .log');return
    s=json.loads((run/'summary.json').read_text())
    print({k:s.get(k) for k in ('execution_valid','validation_complete','stop_reason','failure','evaluation_count','any_motion_target_reached','finalists')})
    print('edge, factor, reduction% (weakest region), ratios tail/middle/head, safe, baseline+recovery, relay, target reached')
    for t in sorted((t for t in s['scan_table'] if t['status']=='completed'),key=lambda t:-t['reduction_percent']):
        print(t['edge'],t['factor'],round(t['reduction_percent'],4),[round(x,6) for x in t['regional_motion_ratios']],t['safe'],t['baseline_and_recovery_preserved'],t['relay_component_preserved'],t['motion_target_reached'])
    print('Not completed:',[(t['edge'],t['factor'],t['status']) for t in s['scan_table'] if t['status']!='completed'])
    print('Skipped (below original lower bound):',[(x['edge'],x['factor']) for x in s['skipped']])
    for c in s['finalist_checks']:print('finalist check',c)
    for c in s['controls']:print('control',Path(c['output']).name,c.get('grid'),c.get('mode'),c['status'],(c.get('metrics') or {}).get('regional_motion_ratios'),(c.get('metrics') or {}).get('relay_response_mean_changes'))
    for path in (run/'budget_receipt.json',run.with_name(run.name+'-audit')/'summary.json'):
        print(str(path),json.loads(path.read_text()) if path.exists() else 'Pending')


if __name__=='__main__':main()
