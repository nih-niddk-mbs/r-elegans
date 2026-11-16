"""Report E084 without importing JAX or loading large trajectories."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);a=p.parse_args();run=a.run
    if run is None:
        runs=[p for p in Path('/home/carsonc/r-elegans-results').glob('rial-control-*') if p.is_dir()]
        if not runs:raise SystemExit('No RIAL control run found')
        run=max(runs,key=lambda p:p.name)
    print('Run:',run)
    if not (run/'summary.json').exists():print('No final summary; inspect adjacent .log');return
    s=json.loads((run/'summary.json').read_text())
    for k in ('execution_valid','validation_complete','failure','selected_membrane','attenuation','learned_stopping_qualified'):print(k+':',s[k])
    for r in s['comparisons']+s['confirmation_comparisons']+s['refinement_comparisons']:
        m=r['metrics'];print(r['case']['name'],r['variant'],r['grid'],dict(body_ratios=[w['body_range_ratio'] for w in m['ventral_SMD_metrics']],
            SMD_rms_ratios=[w['ventral_SMD_rms_ratio'] for w in m['ventral_SMD_metrics']],
            RIA_mean_voltage=[w['mean_voltage_mV'] for w in m['RIA_windows']],RIA_voltage_std=[w['voltage_std_mV'] for w in m['RIA_windows']],
            recovery=m['active_and_recovery_preserved'],pre_match=m['pre_match'],safe=m['safe'],strict_body_rest=m['strict_body_rest']))
    print('Numerical checks:',s['numerical_checks'])
    if (run/'budget_receipt.json').exists():print('Budget:',json.loads((run/'budget_receipt.json').read_text()))


if __name__=='__main__':main()
