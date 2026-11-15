"""Summarize E082 without loading trajectories or importing JAX."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);a=p.parse_args();run=a.run
    if run is None:
        runs=[p for p in Path('/home/carsonc/r-elegans-results').glob('smd-control-*') if p.is_dir()]
        if not runs:raise SystemExit('No SMD control run found')
        run=max(runs,key=lambda p:p.name)
    print('Run:',run)
    if not (run/'summary.json').exists():print('No final summary; inspect adjacent .log');return
    s=json.loads((run/'summary.json').read_text())
    for key in ('execution_valid','validation_complete','failure','selected','reversible_smd_neural_access','learned_stopping_qualified'):
        print(key+':',s[key])
    for r in s['comparisons']+s['confirmation_comparisons']+s['refinement_comparisons']:
        m=r['metrics'];print(r['case']['name'],r['variant'],r['grid'],dict(body_ratios=[w['body_range_ratio'] for w in m['off_windows']],
            neural_ratios=[w['target_voltage_rms_ratio'] for w in m['off_windows']],
            target_mean_shifts_mV=[w['target_mean_shifts_mV'] for w in m['off_windows']],
            preserved=m['active_and_recovery_preserved'],safe=m['safe'],pre_match=m['pre_match'],access=m['reversible_neural_target'],strict_body_rest=m['strict_body_rest']))
    print('Numerical checks:',s['numerical_checks'])
    if (run/'budget_receipt.json').exists():print('Budget:',json.loads((run/'budget_receipt.json').read_text()))


if __name__=='__main__':main()
