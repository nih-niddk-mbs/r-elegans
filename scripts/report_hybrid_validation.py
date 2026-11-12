"""Compact read-only E066 report; direct cycle comparisons are descriptive."""
import argparse
import json
from pathlib import Path


def cycle_comparisons(summary):
    rows = {w['case']['name']: w['result'] for w in summary['finished'] if w.get('result')}
    base = rows['hybrid-sham']['hybrid_assessment']['cycle_excursions']['regional_cycle_excursion_rad']
    results = []
    for name in ('hybrid-neural-half', 'hybrid-body-half', 'hybrid-long'):
        fine = rows[name]['hybrid_assessment']['cycle_excursions']['regional_cycle_excursion_rad']
        delta = [abs(a-b) for a, b in zip(base, fine)]
        results.append(dict(case=name, absolute_difference_rad=delta,
                            relative_difference=[d / a if a > 1e-8 else None for d, a in zip(delta, base)],
                            scope='Descriptive nonzero cyclic-amplitude comparison, not a replacement for original screening'))
    return results


def format_summary(run, summary):
    lines = [f'Run: {run}', f"Candidate: {summary.get('candidate_sha256')}",
             f"Execution valid: {summary['execution_valid']}; ordinary/long simulated seconds: "
             f"{summary['duration_s']}/{summary.get('long_duration_s')}"]
    for w in sorted(summary['finished'], key=lambda w: w['case']['name']):
        r = w.get('result')
        if not r:
            lines.append(f"{w['case']['name']}: exit={w['exit_code']}; missing result")
            continue
        a = r['hybrid_assessment']; c = a['coordination']; n = r['late_repeating_by_stage']
        cycles = '/'.join(f'{x:.6g}' for x in a['cycle_excursions']['regional_cycle_excursion_rad'])
        lines.append(f"{w['case']['name']}: seconds={r['simulation_seconds']:g}; "
                     f"V/muscles/joints={n['dense_voltage_mV']}/{n['dense_muscles']}/{n['dense_joints']}; "
                     f"cycle_rad(T/M/H)={cycles}; coverage={a['repeated_joints_per_region']}; "
                     f"eligible={c['eligible_joints']}; coherent={sum(p['coherence'] >= .8 for p in c['phases'])}; "
                     f"Hz={c['frequency_hz']}; PC1={c['detrended_shape_pc1_fraction']}; span={c['coherent_phase_span_rad']}")
    if all(w.get('result') for w in summary['finished']):
        for row in cycle_comparisons(summary):
            relative = '/'.join('undefined' if x is None else f'{100*x:.3g}%' for x in row['relative_difference'])
            lines.append(f"{row['case']} vs baseline cyclic-amplitude difference(T/M/H): {relative}")
    lines.append('Original screening: ' + json.dumps(summary.get('hybrid_screening')))
    lines.append('Window-gated zeros do not imply zero motion; PC1 with one eligible joint does not describe the whole body. Cycle comparisons add no new acceptance gate.')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    run = args.run.expanduser().resolve()
    summary = json.loads((run / 'summary.json').read_text())
    if summary.get('experiment') != 'E066' or summary.get('status') != 'completed':
        p.error('Completed E066 hybrid validation summary required')
    print(format_summary(run, summary))


if __name__ == '__main__':
    main()
