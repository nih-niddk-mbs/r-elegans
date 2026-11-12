"""Read saved E063 baseline/best/validation signals; no JAX or new simulation."""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

from r_elegans.signal_chain_audit import FIELDS, plot_raw_chain, summarize_trace
from scripts.experiment_records import capture_run_record, file_sha256


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_files(run, summary):
    """Require every declared validation and saved baseline/best training trace."""
    tasks = []
    for phase, iteration in (('baseline', summary['initial_iteration']), ('best', summary['selected_iteration'])):
        selected = run / 'selected' / f'iteration-{iteration:06d}'
        for label in ('active', 'sham', 'dose-low'):
            tasks.append((phase, label, selected / (label + '-trajectory.npz')))
    for row in summary['validation']:
        name = row['case']
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z-]*', name):
            raise ValueError('Unsafe validation case name')
        for label in ('active', 'sham'):
            tasks.append(('validation/' + name, label, run / name / (label + '-trajectory.npz')))
    for _, _, path in tasks:
        if not path.resolve().is_relative_to(run) or not path.is_file():
            raise ValueError('Required trace missing or outside source: ' + str(path))
    return tasks


def paired_difference(active, sham, begin_s=4.):
    if not np.array_equal(active['dense_time_s'], sham['dense_time_s']):
        raise ValueError('Active/sham saved time grids differ')
    use = active['dense_time_s'] >= begin_s
    result = {}
    for field in FIELDS:
        a, b = active[field], sham[field]
        if a.shape != b.shape:
            raise ValueError('Active/sham channel shapes differ')
        d = np.asarray(a[use], dtype=float) - np.asarray(b[use], dtype=float)
        result[field] = dict(maximum_absolute_cue_difference=float(np.abs(d).max()),
                             maximum_cue_difference_p2p=float(np.ptp(d, axis=0).max()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{64}', args.expected_sha256):
        parser.error('Explicit checkpoint SHA256 required')
    run = args.run.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if out == repo or repo in out.parents:
        parser.error('Scientific outputs must remain outside repository')
    if out == run or run in out.parents or out in run.parents:
        parser.error('Output must be separate from source run')
    summary = json.loads((run / 'summary.json').read_text())
    cfg = json.loads((run / 'configuration.json').read_text())
    if (summary['status'] != 'completed' or cfg['stage'] != 'alternation'
            or cfg['arguments']['precision'] != 'float64'
            or cfg['arguments']['training_cues'] != 'nominal-low-dose'):
        raise ValueError('Completed float64 nominal/low-dose alternation run required')
    network = run / 'network_best.npz'
    if file_sha256(network) != args.expected_sha256 or summary['checkpoint_sha256'] != args.expected_sha256:
        raise ValueError('Pinned checkpoint hash mismatch')
    selected = run / 'selected' / f"iteration-{summary['selected_iteration']:06d}" / 'network.npz'
    baseline = run / 'selected' / f"iteration-{summary['initial_iteration']:06d}" / 'network.npz'
    if file_sha256(selected) != args.expected_sha256:
        raise ValueError('Selected checkpoint differs from network_best')
    with np.load(network, allow_pickle=False) as z:
        neuron_ids = z['neuron_ids'].tolist()
    tasks = source_files(run, summary)
    inputs = {run / 'summary.json', run / 'configuration.json', network, selected, baseline}
    inputs.update(path for _, _, path in tasks)
    for path in inputs:
        if not path.resolve().is_relative_to(run):
            raise ValueError('Input reference outside source')
    before = {str(path.relative_to(run)): file_sha256(path) for path in sorted(inputs)}
    out.mkdir(parents=True, exist_ok=False)
    capture_run_record(out, repo, args)
    write_json(out / 'input_hashes_before.json', before)
    report = dict(source_run=str(run), checkpoint_sha256=args.expected_sha256,
                  scope='Descriptive saved-signal analysis; no simulation, fitting, model changes or new acceptance gate',
                  neuron_ids=neuron_ids, traces=[], paired_cue_differences=[],
                  limitations=[
                      'Engineering floors and finite-window repetitions do not establish a limit cycle or biology.',
                      'Named groups are explicit neuron-name filters, not a functional classification inferred from signals.',
                      'No per-edge NMJ trace was recorded; muscle activation is the first available downstream signal.',
                      'Observed signal loss between stages is descriptive and does not prove causality.',
                      'Input hashes establish analysis-time integrity, not original source revision or model reproduction.',
                  ])
    for phase, label, path in tasks:
        destination = out / phase / label
        destination.mkdir(parents=True)
        with np.load(path, allow_pickle=False) as z:
            if z['dense_voltage_mV'].dtype != np.float64:
                raise ValueError('Expected saved float64 voltage')
            stats = summarize_trace(z, neuron_ids)
            write_json(destination / 'channels.json', stats)
            if label == 'active' and phase in ('baseline', 'best', 'validation/active'):
                plot_raw_chain(z, stats, neuron_ids, destination / 'raw-signals.png', phase)
        counts = {key: dict(reversing=len(row['reversing_indices']), repeating=len(row['repeating_indices']),
                            channels=row['channels'], maximum_raw_p2p=max(row['raw_p2p']))
                  for key, row in stats.items()}
        entry = dict(phase=phase, label=label, source=str(path.relative_to(run)), counts=counts,
                     repeating_neurons=stats['dense_voltage_mV']['repeating_neurons'],
                     voltage_named_groups=stats['dense_voltage_mV']['named_groups'])
        report['traces'].append(entry)
        print(json.dumps(dict(phase=phase, label=label, counts=counts)), flush=True)
    for phase, label, path in tasks:
        if label != 'active':
            continue
        sham = next(p for ph, lab, p in tasks if ph == phase and lab == 'sham')
        with np.load(path, allow_pickle=False) as a, np.load(sham, allow_pickle=False) as b:
            report['paired_cue_differences'].append(dict(phase=phase, fields=paired_difference(a, b)))
    after = {str(path.relative_to(run)): file_sha256(path) for path in sorted(inputs)}
    write_json(out / 'input_hashes_after.json', after)
    if before != after:
        raise ValueError('Source files changed during analysis')
    report['input_files_unchanged'] = True
    report['input_file_count'] = len(before)
    report['jax_imported'] = 'jax' in sys.modules
    if report['jax_imported']:
        raise RuntimeError('Saved-file diagnostic unexpectedly imported JAX')
    write_json(out / 'summary.json', report)
    print('Input hashes unchanged. Report: ' + str(out / 'summary.json'), flush=True)
    for row in report['traces']:
        if row['phase'] in ('baseline', 'best', 'validation/active', 'validation/held-voltage') and row['label'] == 'active':
            print(json.dumps(dict(phase=row['phase'], repeating_by_stage={k: v['repeating'] for k, v in row['counts'].items()},
                                  repeating_neurons=row['repeating_neurons'],
                                  voltage_groups={k: dict(repeating_neurons=v['repeating_neurons'], maximum_raw_p2p=v['maximum_raw_p2p'])
                                                  for k, v in row['voltage_named_groups'].items()})), flush=True)


if __name__ == '__main__':
    main()
