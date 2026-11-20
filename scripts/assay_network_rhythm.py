"""Rhythm component check of conductance networks (RMD membrane oscillation, free run, no stimulus).

Applies the existing persistence gates (r_elegans.brain.rhythm_qualification) to every RMD cell and a coarse-vs-fine
integration comparison on cells that pass. It does not test locomotion, coordination, start/stop or any behavior.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.assets import load_asset_document
from r_elegans.brain.conductance_network import load_network, network_recording
from r_elegans.brain.rhythm_qualification import compare_cycles, rhythm_metrics

try:
    from .experiment_records import capture_run_record
except ImportError:
    from experiment_records import capture_run_record


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', action='append', required=True, metavar='LABEL=PATH')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=40.)
    parser.add_argument('--sample-dt-s', type=float, default=.002)
    parser.add_argument('--max-step-ms', type=float, default=1.)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.expanduser().resolve()
    if output == repo or repo in output.parents:
        parser.error('Keep artifacts external')
    output.mkdir(parents=True, exist_ok=False)
    capture_run_record(output, repo, args)
    neuron_ids = list(load_asset_document('runtime_model_v1.json')['neuron_ids'])
    rmd = [i for i, n in enumerate(neuron_ids) if n.startswith('RMD')]
    float_type = jnp.result_type(float)
    frames = int(round(args.seconds / args.sample_dt_s))
    substeps = math.ceil(args.sample_dt_s * 1000. / args.max_step_ms)
    report = {}
    for item in args.network:
        label, _, path = item.partition('=')
        params, fixed, config = load_network(path)
        params, fixed = jax.tree_util.tree_map(
            lambda a: a.astype(float_type) if jnp.issubdtype(a.dtype, jnp.floating) else a, (params, fixed))
        run = lambda s: np.asarray(jax.jit(lambda p: network_recording(p, fixed, args.sample_dt_s, frames, s, config).voltage_mV)(params))
        coarse, fine = run(substeps), run(2 * substeps)
        cells = {}
        for i in rmd:
            metrics = rhythm_metrics(coarse[:, i], args.sample_dt_s)
            if metrics['passed']:
                metrics['coarse_vs_fine'] = compare_cycles(coarse[:, i], fine[:, i], args.sample_dt_s)
            cells[neuron_ids[i]] = metrics
        late = coarse[len(coarse) // 2:]
        spread = late.std(axis=0)
        report[label] = dict(
            path=str(Path(path).resolve()), sha256=sha256(path), seconds=args.seconds, substeps=substeps,
            finite=bool(np.isfinite(coarse).all() and np.isfinite(fine).all()),
            rmd_cells_passing=[n for n, m in cells.items() if m['passed']],
            rmd_cells_passing_convergence=[n for n, m in cells.items() if m['passed'] and m['coarse_vs_fine'].get('passed')],
            late_window_std_mV=dict(neurons_above_1=int((spread > 1.).sum()), neurons_above_5=int((spread > 5.).sum()),
                                    max=float(spread.max()), rmd={neuron_ids[i]: round(float(spread[i]), 3) for i in rmd}),
            voltage_range_mV=[float(coarse.min()), float(coarse.max())], cells=cells)
        print(json.dumps({label: {k: v for k, v in report[label].items() if k != 'cells'}}, default=float), flush=True)
    (output / 'rhythm_check.json').write_text(json.dumps(report, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
