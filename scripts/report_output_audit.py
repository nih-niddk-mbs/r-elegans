"""Print the existing output-path audit; no simulation, JAX or file writes."""
import argparse
import json
from pathlib import Path
import statistics


def vector(values):
    return '/'.join(f'{x:.6g}' for x in values)


def describe(values):
    values = sorted(values)
    return vector([values[0], statistics.median(values), values[-1]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path)
    p.add_argument('--all-cases', action='store_true')
    a = p.parse_args()
    if a.run:
        run = a.run.expanduser().resolve()
    else:
        base = Path('/home/carsonc/r-elegans-results')
        runs = [r for r in base.glob('output-path-audit-*') if r.is_dir() and (r/'summary.json').is_file()]
        if not runs:
            p.error('No completed output-path audit found; provide --run')
        run = max(runs, key=lambda r: r.name)
    s = json.loads((run/'summary.json').read_text())
    if s['experiment'] != 'E068' or not s['execution_valid'] or not s['source_files_unchanged']:
        p.error('Valid completed E068 report with unchanged input hashes required')
    print('Run:', run)
    print('Software smoke:', s['software_smoke'], '; frozen readout:', s['actor_sha256'])
    print('T/M/H = tail/middle/head; only joint cycles have units radians.')
    print('Drive stages are normalized activations, not forces; their ratios are descriptive.')
    focus = {'old-old-sham', 'hybrid-sham', 'hybrid-long', 'seed-6703-ordinary', 'seed-6703-long'}
    for r in s['cases']:
        if not a.all_cases and r['case'] not in focus:
            continue
        print('\n'+r['case']+'; minimum joint cycles T/M/H='+vector(r['minimum_joint_cycles_tail_middle_head']))
        print('Reconstruction error NMJ/muscle, projection:',
              r['nmj_activation_reconstruction_error'], r['projection_maximum_error'])
        for w in r['windows']:
            stages = ' '.join(k+'='+vector(w['cycles'][k]['tail_middle_head']) for k in
                              ('dorsal', 'ventral', 'net_command', 'filtered_drive', 'joints'))
            print(f"  {w['start_s']:g}-{w['end_s']:g}s T/M/H: {stages}")
            print('    Across95 muscles min/median/max: NMJdrive_p2p='+describe(w['signed_normalized_nmj_drive_raw_p2p'])+
                  ' activation_p2p='+describe(w['muscle_raw_p2p'])+
                  ' static_sigmoid_slope='+describe(w['muscle_mean_local_sigmoid_slope'])+
                  ' near_extremes_fraction='+describe(w['muscle_near_extremes_fraction']))
        print('  Motor frequency Hz:', r['windows'][-1]['balance']['frequency_hz'])
    print('\nSingle-frequency cancellation is not broadband energy loss. Static slopes do not establish causal gain.')
    print('Production windows: ordinary30-45/45-60s vs long150-225/225-300s; smoke windows differ. No new acceptance gate.')


if __name__ == '__main__':
    main()
