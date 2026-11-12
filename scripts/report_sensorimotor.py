"""Compact read-only sensory-motor summary, suitable for copying from a log."""
import argparse
import json
from pathlib import Path


def format_summary(root,summary,configuration):
    target=configuration['protocol']['bend_target_rad']
    target_label=f'bend target: {target:.9g} rad'
    if configuration['stage']=='alternation':
        target_label=f"post-cue peak-to-peak target: {configuration['arguments']['alternation_p2p_rad']:.9g} rad"
    lines=[f'Run: {root}',f"Passed: {summary['passed']}; stop: {summary.get('stop_reason')}",
        f"Stage: {configuration['stage']}; training cues: {configuration['arguments'].get('training_cues','nominal')}; precision: {configuration['arguments'].get('precision','float32')}; training margin: {configuration['arguments'].get('training_margin',1.)}",
        f"Proposals this run: {summary['iterations']}; global: {summary.get('initial_iteration',0)} -> {summary.get('last_iteration',summary['iterations'])}; selected: {summary['selected_iteration']}",
        f"Elapsed minutes: {summary['elapsed_seconds']/60:.2f}; {target_label}"]
    for label in ('baseline','best'):
        r=summary[label];m=r['metrics']
        if 'alternating_p2p_rad' in m:
            lines.append(f"{label}: loss={r['total']:.9g}; post-cue_p2p={m['alternating_p2p_rad']:.9g}; peaks/troughs={m['peaks']}/{m['troughs']}; window_ratio={m['window_amplitude_ratio']:.4f}; AWA={m['awa_depolarization_mV']:.6g}")
        else:
            lines.append(f"{label}: loss={r['total']:.9g}; bend={m['cue_specific_peak_rad']:.9g}; recovery={m['recovery_ratio']:.4f}; AWA={m['awa_depolarization_mV']:.6g} mV")
        for case in r.get('training_cases',[]):
            c=case['metrics']
            if 'alternating_p2p_rad' in c:
                lines.append(f"  {label}/{case['case']}: pattern_loss={c['total']:.9g}; post-cue_p2p={c['alternating_p2p_rad']:.9g}; peaks/troughs={c['peaks']}/{c['troughs']}")
            else:
                lines.append(f"  {label}/{case['case']}: response_loss={c['total']:.9g}; bend={c['cue_specific_peak_rad']:.9g}; recovery={c['recovery_ratio']:.4f}")
    for row in summary['validation']:
        m=row['metrics']
        if 'alternating_p2p_rad' in m:
            lines.append(f"{row['case']}: passed={m['passed']}; pattern_pass={m['pattern_pass']}; post-cue_p2p={m['alternating_p2p_rad']:.9g}; peaks/troughs={m['peaks']}/{m['troughs']}; window_ratio={m['window_amplitude_ratio']:.4f}; AWA={m['awa_depolarization_mV']:.6g}")
        else:
            lines.append(f"{row['case']}: passed={m['passed']}; bend={m['cue_specific_peak_rad']:.9g} ({100*m['cue_specific_peak_rad']/target:.1f}% target); recovery={m['recovery_ratio']:.4f}; AWA={m['awa_depolarization_mV']:.6g} mV; AWA_recovery={m.get('awa_recovery_ratio')}")
    if summary.get('unforced_pattern'):
        lines.append('Unforced short-window pattern pass: '+str(summary['unforced_pattern']['pattern_pass']))
    if summary.get('pattern_refinement_comparisons'):
        lines.append('Pattern refinement checks: '+json.dumps(summary['pattern_refinement_comparisons']))
    lines.extend([f"Blocked controls: {summary['blocked_controls_passed']}; refinement: {json.dumps(summary['refinement_comparison'])}",
        f"Sensory gains chemical/B/DVA: {summary['sensory_gain_multipliers']}",
        f"Changed neural coordinates this run: {summary.get('neuron_connection_parameters_changed_this_run','not recorded')}; versus original paired parent: {summary['neuron_connection_parameters_changed_from_parent']}",
        f"Checkpoint: {summary['checkpoint_sha256']}",
        f"Resume checkpoint: {summary.get('resume_checkpoint_sha256','not recorded')}"])
    if summary.get('warm_start_checkpoint_sha256'):
        lines.append(f"Warm-start checkpoint (fresh search): {summary['warm_start_checkpoint_sha256']}")
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path)
    p.add_argument('--base',type=Path,default=Path('/home/carsonc/r-elegans-results'))
    args=p.parse_args()
    root=args.run
    if root is None:
        roots=sorted((path for pattern in ('sensorimotor-*','sensorimotor64-*','alternation64-*') for path in args.base.glob(pattern) if path.is_dir()),
            key=lambda path:path.name.rsplit('-',2)[-2:])
        if not roots:p.error('No sensory-motor run directory found')
        root=roots[-1]
    if not (root/'summary.json').exists():
        print(f'Run: {root}\nNo final summary yet; check its log.')
        return
    summary=json.loads((root/'summary.json').read_text())
    configuration=json.loads((root/'configuration.json').read_text())
    print(format_summary(root,summary,configuration),flush=True)


if __name__=='__main__':main()
