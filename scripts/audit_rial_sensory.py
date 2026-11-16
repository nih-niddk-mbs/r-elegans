"""Read-only E085 post-run audit: all candidates, gates, bounds and saved signals."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import expit

from scripts.experiment_records import capture_run_record, file_sha256


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def gait_failures(paired, parent):
    """Expand the exact E085 preservation predicates into named failed checks."""
    failures = []
    a, b = paired['no_cue'], paired['cue']

    def check(row, ref, label, lower, floor):
        for i, (amp, base, f, f0) in enumerate(zip(
                row['regional_cycles_rad'], ref['regional_cycles_rad'],
                row['frequency_hz'], ref['frequency_hz'])):
            lo, hi = max(floor, lower * base), 1.1 * base
            if not lo <= amp <= hi:
                failures.append(dict(window=label, region=i, check='cycle_amplitude',
                                     value=amp, lower=lo, upper=hi))
            if f is None or not 1.62 <= f <= 1.98:
                failures.append(dict(window=label, region=i, check='absolute_frequency', value=f))
            if f is None or f0 is None or min(f, f0) <= 0 or abs(np.log(f / f0)) > np.log(1.05):
                failures.append(dict(window=label, region=i, check='relative_frequency', value=f, reference=f0))

    for category in ('active', 'response', 'late'):
        for i, row in enumerate(a[category]):
            check(row, parent['active'][0], f'no_cue/{category}/{i}', .9, .003)
    for i, (row, ref) in enumerate(zip(b['active'], a['active'])):
        check(row, ref, f'cue/active/{i}', .9, .003)
    for i, (row, ref) in enumerate(zip(b['response'], a['response'])):
        check(row, ref, f'cue/response/{i}', .5, 0.)
    return failures


def signal_windows(no_cue, cue, ids, threshold, slope, windows):
    if not np.array_equal(no_cue['dense_time_s'], cue['dense_time_s']):
        raise ValueError('Paired trace clocks differ')
    t = no_cue['dense_time_s']
    rows = []
    for start, end in windows:
        use = (t >= start) & (t < end)
        if not use.any():
            raise ValueError('Missing signal window')
        row = dict(window=[start, end], stages={})
        for key in ('dense_voltage_mV', 'dense_calcium_uM', 'dense_muscles',
                    'dense_command', 'dense_filtered_drive', 'dense_joints'):
            if key not in no_cue or key not in cue:
                row['stages'][key] = dict(missing=True)
                continue
            a, b = no_cue[key][use], cue[key][use]
            row['stages'][key] = dict(no_cue_mean=a.mean(0).tolist(), cue_mean=b.mean(0).tolist(),
                no_cue_std=a.std(0).tolist(), cue_std=b.std(0).tolist(),
                cue_minus_sham_rms=np.sqrt(np.mean((b-a)**2, axis=0)).tolist())
            if key == 'dense_voltage_mV':
                ra, rb = expit((a-threshold)/slope), expit((b-threshold)/slope)
                row['release_by_neuron'] = {
                    n: dict(no_cue_mean=float(ra[:, i].mean()), cue_mean=float(rb[:, i].mean()),
                            no_cue_std=float(ra[:, i].std()), cue_std=float(rb[:, i].std()),
                            difference_rms=float(np.sqrt(np.mean((rb[:, i]-ra[:, i])**2))))
                    for i, n in enumerate(ids)}
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output-dir', type=Path)
    a = p.parse_args()
    run = a.run.resolve()
    out = a.output_dir or run.parent / ('rial-sensory-audit-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out = out.resolve()
    if out == run or run in out.parents or out in run.parents:
        raise ValueError('Audit output must be separate from source run')
    out.mkdir(parents=True, exist_ok=False)
    capture_run_record(out, Path(__file__).resolve().parents[1], a)
    summary = json.loads((run/'summary.json').read_text())
    parent = json.loads((run/'parent_features.json').read_text())
    groups = json.loads((run/'parameter_groups.json').read_text())
    integrity, candidates, problems = [], [], []

    def verify(path, expected=None):
        if not path.is_file():
            problems.append('Missing: ' + str(path)); return
        actual = file_sha256(path)
        matched = expected is None or actual == expected
        integrity.append(dict(path=str(path), sha256=actual, expected=expected, matched=matched))
        if not matched:
            problems.append('SHA mismatch: ' + str(path))

    manifest = json.loads((run/'inputs/manifest.json').read_text())
    for name, sha in manifest['sha256'].items():
        verify(run/'inputs'/name, sha)
    ref = json.loads((run/'inputs/sensory_reference.json').read_text())
    verify(Path(ref['path']), ref['sha256'])
    for path in sorted((run/'evaluations').glob('*/summary.json')):
        r = json.loads(path.read_text()); folder = path.parent
        if r['status'] == 'completed':
            verify(folder/'network.npz', r['checkpoint_sha256'])
            verify(folder/'coordinates.npz')
        candidates.append(dict(output=str(folder), status=r['status'], error=r.get('error'),
            metrics=r.get('metrics'), paired_features=r.get('paired_features'),
            checkpoint_sha256=r.get('checkpoint_sha256')))
    write(out/'candidates.json', candidates)
    selected_path = Path(summary['selected']['output'])
    with np.load(run/'parameter_origin.npz', allow_pickle=False) as z:
        origin, lower, upper = z['origin'], z['lower'], z['upper']
    with np.load(selected_path/'coordinates.npz', allow_pickle=False) as z:
        offsets, latent = z['offsets'], z['z']
    if latent.shape != (len(groups),):
        raise ValueError('Parameter schema/coordinate dimension mismatch')
    used = np.zeros(len(origin), dtype=bool); group_rows = []
    for value, group in zip(latent, groups):
        ix = np.asarray(group['indices']); used[ix] = True
        group_rows.append(dict(name=group['name'], coordinates=len(ix), latent=float(value),
            near_group_bound=bool(abs(value) >= .99),
            log_delta_range=[float((offsets-origin)[ix].min()), float((offsets-origin)[ix].max())],
            near_original_bound=int(np.sum((offsets[ix]-lower[ix] <= .01*(upper[ix]-lower[ix])) |
                                          (upper[ix]-offsets[ix] <= .01*(upper[ix]-lower[ix]))))))
    frozen = bool(np.array_equal(offsets[~used], origin[~used]))
    if not frozen: problems.append('Coordinates outside trainable groups changed')
    write(out/'parameters.json', dict(groups=group_rows, frozen_coordinates_unchanged=frozen,
        original_bounds_respected=bool(np.all(offsets >= lower-1e-10) and np.all(offsets <= upper+1e-10))))
    with np.load(selected_path/'network.npz', allow_pickle=False) as z:
        ids = z['neuron_ids'].tolist()
        threshold, slope = z['fixed_release_threshold_mV'], z['fixed_release_slope_mV']
    gate_rows = []
    for index, row in enumerate(summary['validation']):
        print('Auditing final case:', row['variant'], row['grid'], row['mode'], flush=True)
        folder = Path(row['output']); paired = row.get('paired_features')
        if row.get('status') != 'completed':
            problems.append('Incomplete final evaluation: ' + str(folder))
        required = {'no_cue', 'cue'}
        if row['mode'] == 'rial-replay': required.update(('no_cue_replay', 'cue_replay'))
        if not required.issubset(row.get('trajectory_sha256', {})):
            problems.append('Missing declared final trajectory hashes: ' + str(folder))
        if row.get('checkpoint_sha256') != summary['selected']['checkpoint_sha256']:
            problems.append('Final checkpoint differs from selected model: ' + str(folder))
        if not paired:
            problems.append('Missing final features: ' + str(folder)); continue
        gate_rows.append(dict(output=str(folder), metrics=row['metrics'], mediation=row.get('mediation'),
                              failures=gait_failures(paired, parent)))
        for condition, sha in row.get('trajectory_sha256', {}).items():
            path = folder/(condition+'.npz'); verify(path, sha)
            if path.is_file():
                with np.load(path, allow_pickle=False) as trace:
                    for key in trace.files:
                        if not np.isfinite(trace[key]).all(): problems.append(f'Nonfinite {path}:{key}')
        if (folder/'no_cue.npz').is_file() and (folder/'cue.npz').is_file():
            windows = sorted({tuple(w['window']) for category in ('active','response','late') for w in paired['no_cue'][category]})
            with np.load(folder/'no_cue.npz', allow_pickle=False) as nc, np.load(folder/'cue.npz', allow_pickle=False) as cu:
                # Load each archive once; never write new full trajectories.
                nc = {k: nc[k] for k in nc.files}; cu = {k: cu[k] for k in cu.files}
                signals = signal_windows(nc, cu, ids, threshold, slope, windows)
            write(out/f'signals-{index:02d}.json', dict(case=row['output'], neuron_ids=ids, windows=signals))
    write(out/'gait_gates.json', gate_rows)
    if len(gate_rows) != 10: problems.append('Expected ten final evaluations')
    write(out/'integrity.json', integrity)
    result = dict(run=str(run), audit_complete=True, integrity_passed=not problems, problems=problems,
        candidate_count=len(candidates), final_cases=len(gate_rows),
        failed_gait_checks={Path(r['output']).name:len(r['failures']) for r in gate_rows},
        groups_near_bound=[r['name'] for r in group_rows if r['near_group_bound']],
        frozen_coordinates_unchanged=frozen,
        scope='Saved-data audit only: no forward simulation, fitting, causal intervention or qualification. Release statistics use recorded voltage and fixed release law, not channel-current attribution. Optimizer/source records remain in source run; hashes here cover models, coordinates, declared inputs and final traces. Missing stage fields are explicit.')
    write(out/'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)
    print('Audit:', out, flush=True)


if __name__ == '__main__':
    main()
