"""Re-run an audited E093-E099 sensory-motion evaluation with a changed network fixed field (e.g. synapse polarity).

Takes the saved evaluation request of a pipeline run, replays it unchanged as a control (must reproduce the saved
summary), then replays it with the fixed fields of another checkpoint that has identical parameters and differs only in
`synapse_reversal_mV`. Reports gait/recovery/cue-response metrics side by side. Nothing is trained.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts.experiment_records import capture_run_record


def numeric_leaves(value, prefix=''):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return {}
    if isinstance(value, (int, float)):
        return {prefix: float(value)}
    if isinstance(value, dict):
        return {k: v for key, item in value.items() for k, v in numeric_leaves(item, f'{prefix}/{key}').items()}
    if isinstance(value, (list, tuple)):
        return {k: v for i, item in enumerate(value) for k, v in numeric_leaves(item, f'{prefix}[{i}]').items()}
    return {}


def max_relative_difference(a, b):
    x, y = numeric_leaves(a), numeric_leaves(b)
    if set(x) != set(y):
        return float('inf')
    return max((abs(x[k] - y[k]) / max(abs(y[k]), 1e-12) for k in x), default=0.)


def gait_rows(result):
    paired = result.get('paired_features') or {}
    return {condition: [dict(window=r['window'], frequency_hz=r['frequency_hz'], regional_cycles_rad=r['regional_cycles_rad'])
                        for r in rows['active']] for condition, rows in paired.items() if 'active' in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='pipeline run folder containing evaluations/ and inputs/')
    parser.add_argument('--evaluation', default='final-00')
    parser.add_argument('--fixed-network', type=Path, required=True, help='checkpoint whose fixed fields replace the evaluation network\'s')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--minutes', type=float, default=120., help='wall deadline per replay')
    parser.add_argument('--check-only', action='store_true', help='verify identity of parameters/fields and exit')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = args.output_dir.expanduser().resolve()
    run = args.run.expanduser().resolve()
    if out == repo or repo in out.parents or run in out.parents or out == run:
        parser.error('Keep artifacts external and outside the source run')
    out.mkdir(parents=True, exist_ok=False)
    capture_run_record(out, repo, args)
    import jax
    if not jax.config.jax_enable_x64:
        parser.error('Set JAX_ENABLE_X64=true')
    from r_elegans.brain.conductance_network import load_network

    source = run / 'evaluations' / args.evaluation
    job = json.loads((source / 'request.json').read_text())
    reference = json.loads((source / 'summary.json').read_text())
    base_params, base_fixed, _ = load_network(source / 'network.npz')
    new_params, new_fixed, _ = load_network(args.fixed_network)
    same = lambda a, b: np.array_equal(np.asarray(a), np.asarray(b))
    identity = dict(
        identical_parameters=all(same(a, b) for a, b in zip(base_params, new_params)),
        identical_cell_fields=all(same(a, b) for a, b in zip(base_fixed.cells, new_fixed.cells)),
        fixed_fields_that_differ=[k for k in base_fixed._fields if k != 'cells' and not same(getattr(base_fixed, k), getattr(new_fixed, k))],
        evaluation_network_sha256=reference.get('checkpoint_sha256'), fixed_network=str(args.fixed_network.resolve()))
    print(json.dumps(identity), flush=True)
    if not (identity['identical_parameters'] and identity['identical_cell_fields']
            and identity['fixed_fields_that_differ'] == ['synapse_reversal_mV']):
        raise SystemExit('Checkpoints must differ only in synapse_reversal_mV')
    (out / 'identity.json').write_text(json.dumps(identity, indent=1) + '\n')
    print(json.dumps(dict(job_keys=sorted(job), objective=job.get('objective'), variant=job.get('variant'), grid=job.get('grid'),
                          retain_traces=job.get('retain_traces'), reference=job.get('reference'))), flush=True)
    if args.check_only:
        return 0

    from scripts import rial_sensory_worker as core
    from scripts.sensory_motion_worker import evaluate, setup
    core._DATA = setup(run / 'inputs')
    results = {}
    for label in ('control', 'polarized'):
        replay = dict(job, output=str(out / label), until=time.monotonic() + args.minutes * 60., params_network=str(source / 'network.npz'))
        replay.pop('fixed_network', None)
        if label == 'polarized':
            replay.update(fixed_network=str(args.fixed_network), reference=False)
        else:
            replay['retain_traces'] = False
        results[label] = evaluate(replay)
        print(json.dumps(dict(label=label, status=results[label]['status'], elapsed_seconds=results[label].get('elapsed_seconds'),
                              checkpoint_sha256=results[label].get('checkpoint_sha256'))), flush=True)
    control, polarized = results['control'], results['polarized']
    flags = lambda r: {k: v for k, v in (r.get('metrics') or {}).items() if isinstance(v, (bool, str))}
    comparison = dict(
        evaluation=args.evaluation, saved_checkpoint_sha256=reference.get('checkpoint_sha256'),
        control_checkpoint_matches_saved=control.get('checkpoint_sha256') == reference.get('checkpoint_sha256'),
        control_metrics_max_relative_difference_from_saved=max_relative_difference(control.get('metrics'), reference.get('metrics')),
        control_status=control['status'], polarized_status=polarized['status'],
        saved_flags=flags(reference), control_flags=flags(control), polarized_flags=flags(polarized),
        polarized_checkpoint_sha256=polarized.get('checkpoint_sha256'),
        gait_windows=dict(saved=gait_rows(reference), control=gait_rows(control), polarized=gait_rows(polarized)),
        scope='Fixed reversal field only; gait/recovery/cue-response metrics of the same protocol; no retraining.')
    (out / 'comparison.json').write_text(json.dumps(comparison, indent=1, default=float) + '\n')
    print(json.dumps({k: v for k, v in comparison.items() if k != 'gait_windows'}, default=float), flush=True)
    return 0 if control['status'] == polarized['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
