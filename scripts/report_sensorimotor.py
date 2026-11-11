"""Compact read-only sensory-motor summary, suitable for copying from a log."""
import argparse
import json
from pathlib import Path


def format_summary(root,summary,configuration):
    target=configuration['protocol']['bend_target_rad']
    lines=[f'Run: {root}',f"Passed: {summary['passed']}; stop: {summary.get('stop_reason')}",
        f"Proposals this run: {summary['iterations']}; global: {summary.get('initial_iteration',0)} -> {summary.get('last_iteration',summary['iterations'])}; selected: {summary['selected_iteration']}",
        f"Elapsed minutes: {summary['elapsed_seconds']/60:.2f}; bend target: {target:.9g} rad"]
    for label in ('baseline','best'):
        r=summary[label];m=r['metrics']
        lines.append(f"{label}: loss={r['total']:.9g}; bend={m['cue_specific_peak_rad']:.9g}; recovery={m['recovery_ratio']:.4f}; AWA={m['awa_depolarization_mV']:.6g} mV")
    for row in summary['validation']:
        m=row['metrics']
        lines.append(f"{row['case']}: passed={m['passed']}; bend={m['cue_specific_peak_rad']:.9g} ({100*m['cue_specific_peak_rad']/target:.1f}% target); recovery={m['recovery_ratio']:.4f}; AWA={m['awa_depolarization_mV']:.6g} mV; AWA_recovery={m.get('awa_recovery_ratio')}")
    lines.extend([f"Blocked controls: {summary['blocked_controls_passed']}; refinement: {json.dumps(summary['refinement_comparison'])}",
        f"Sensory gains chemical/B/DVA: {summary['sensory_gain_multipliers']}",
        f"Changed neural coordinates this run: {summary.get('neuron_connection_parameters_changed_this_run','not recorded')}; versus original paired parent: {summary['neuron_connection_parameters_changed_from_parent']}",
        f"Checkpoint: {summary['checkpoint_sha256']}",
        f"Resume checkpoint: {summary.get('resume_checkpoint_sha256','not recorded')}"])
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path)
    p.add_argument('--base',type=Path,default=Path('/home/carsonc/r-elegans-results'))
    args=p.parse_args()
    root=args.run
    if root is None:
        roots=sorted(path for path in args.base.glob('sensorimotor-*') if path.is_dir())
        if not roots:p.error('No sensory-motor run directory found')
        root=roots[-1]
    if not (root/'summary.json').exists():
        print(f'Run: {root}\nNo final summary yet; check its log.')
        return
    summary=json.loads((root/'summary.json').read_text())
    configuration=json.loads((root/'configuration.json').read_text())
    print(format_summary(root,summary,configuration),flush=True)


if __name__=='__main__':main()
