"""Compact E083 report without JAX or large-array loading."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);a=p.parse_args();run=a.run
    if run is None:
        runs=[p for p in Path('/home/carsonc/r-elegans-results').glob('smd-drive-audit-*') if p.is_dir()]
        if not runs:raise SystemExit('No SMD drive audit found')
        run=max(runs,key=lambda p:p.name)
    print('Run:',run)
    if not (run/'summary.json').exists():print('No final summary; inspect adjacent .log');return
    s=json.loads((run/'summary.json').read_text());print('Status:',s['status']);print('Checks:',s['checks'])
    for label,r in s['results'].items():
        print('\nTrial:',label)
        for condition in ('sham','inhibited'):
            print(condition,'command:',r[condition]['command']['centered_rms'])
            for c in r[condition]['cells']:
                if c['neuron'] not in ('SMDVL','SMDVR'):continue
                print({k:c[k] for k in ('neuron','voltage_mean_mV','voltage_std_mV','release_mean','release_std','dominant_voltage_frequency_hz','source_harmonic_sum_pA')})
                print('Top inputs:',[{k:x[k] for k in ('neuron','source_harmonic_pA','chemical_harmonic_pA','gap_harmonic_pA')} for x in c['top_rhythmic_inputs'][:3]])
            print('Release hold command RMS ratios:',{x['group']:x['centered_command_rms_ratio'] for x in r[condition]['release_holds']})
    print('Figure:',s['figure']);print('Limitations:',s['limitations'])


if __name__=='__main__':main()
