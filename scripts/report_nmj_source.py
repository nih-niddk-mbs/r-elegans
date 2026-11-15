"""Report E080 population-output diagnostics without importing JAX."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);a=p.parse_args()
    run=a.run
    if run is None:
        runs=[p for p in Path('/home/carsonc/r-elegans-results').glob('nmj-source-*') if p.is_dir()]
        if not runs:raise SystemExit('No NMJ source run found')
        run=max(runs,key=lambda p:p.name)
    print('Run:',run)
    if not (run/'summary.json').exists():print('No final summary; inspect adjacent .log');return
    s=json.loads((run/'summary.json').read_text())
    for key in ('execution_valid','validation_complete','failure','learned_stopping_qualified'):print(key+':',s[key])
    for r in s['comparisons']+s['refinement_comparisons']:
        m=r['metrics'];print(r['case'],r['grid'],dict(body_range_ratio=m['during_to_sham_raw_range_ratio'],muscle_range_ratio=m['NMJ_muscle_range_ratio'],recovery=m['recovery_amplitude_preserved'],pre_match=m['pre_cue_match'],safe=m['safety_passed']))
    print('Numerical checks:',s['numerical_checks'])
    if (run/'budget_receipt.json').exists():print('Budget:',json.loads((run/'budget_receipt.json').read_text()))


if __name__=='__main__':main()
