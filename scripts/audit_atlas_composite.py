"""E112 saved-data audit of an explicit target-dependent atlas refinement protocol."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
from scripts.experiment_records import capture_run_record, file_sha256
from scripts.selected_atlas_audit_checks import saved_arrays, write, compare_responses
from scripts.atlas_refinement_checks import choose_targets, layer_values
from scripts.atlas_shared_checks import observed_pairs, compare_downstream
from scripts.run_atlas_shared_state import SELECTED_SHA, PARENT_SHA, ANCHOR_SHA

COARSE_STEPS = (.125, .0625, .03125)
FINE_STEPS = (.0078125, .00390625, .001953125)
CRITICAL = ('r_elegans/brain/common_state_replay.py', 'r_elegans/brain/conductance_network.py',
            'r_elegans/brain/single_compartment.py')


def assemble_level(coarse, fine, targets, hard, coarse_step, fine_step):
    if len(set(targets)) != len(targets) or len(set(hard)) != len(hard) or not set(hard).issubset(targets):
        raise ValueError('Unique full target and subset identities required')
    if coarse.ndim != 3 or coarse.shape[:2] != (4, len(targets)) or fine.shape != (4, len(hard), coarse.shape[2]):
        raise ValueError('Composite matrix coverage differs')
    ix = np.asarray([targets.index(t) for t in hard])
    response = coarse.copy();response[:, ix] = fine
    steps = np.full(len(targets), float(coarse_step));steps[ix] = fine_step
    return response, steps


def compare_bridge(a, b, hard, hard_pairs, ids):
    result = {}
    bi = np.asarray([hard.index(t) for t, _ in hard_pairs]);bj = np.asarray([ids.index(r) for _, r in hard_pairs])
    for name, x, y in (('all_modeled_receivers', a, b), ('observed_downstream', a[:, bi, bj], b[:, bi, bj])):
        result[name] = [compare_responses(u, v, rtol=1e-6, atol=1e-12, minimum_correlation=.999999)
                        for u, v in list(zip(x, y)) + [(x.mean(axis=0), y.mean(axis=0))]]
    return result


def worker_identity(source, label, read):
    records = []
    for record in sorted((source / label / 'run_records').glob('*/source_record.json')):
        command = json.loads(read(record.parent / 'command.json').read_text())
        args = command.get('parsed_arguments', {})
        if args.get('label') == label and Path(args.get('root', '/missing')).resolve() == source.resolve():
            records.append(record)
    if len(records) != 1:raise ValueError('One unambiguous worker source record required: ' + str(source / label))
    path = records[0];record = json.loads(read(path).read_text())
    sha = {}
    for name in CRITICAL:
        expected = record['source_sha256'][name]
        read(path.parent / 'source' / name, expected);sha[name] = expected
    env = json.loads(read(path.parent / 'environment.json').read_text())
    if str(env['numerical_environment'].get('JAX_ENABLE_X64', '')).lower() not in ('true', '1'):
        raise ValueError('Recorded float64 setting missing')
    packages = {str(k).lower().replace('_', '-'):v for k, v in env['packages'].items()}
    versions = {name:packages[name] for name in ('jax', 'jaxlib', 'numpy')}
    if any(not value for value in versions.values()):raise ValueError('Forward package version missing')
    return dict(forward_source_sha256=sha, packages=versions, python_version=env['python_version'],
                x64=True, matmul_setting=env['numerical_environment'].get('JAX_DEFAULT_MATMUL_PRECISION'),
                xla_flags=env['numerical_environment'].get('XLA_FLAGS'))


def analyze(coarse, fine, output):
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    from r_elegans.atlas_evaluation import pair_metrics, model_pairs
    evidence, layers = {}, {}
    def read(path, expected=None):
        path = Path(path);key = str(path.resolve())
        digest = evidence.get(key)
        if digest is None:digest = file_sha256(path);evidence[key] = digest
        if expected is not None and digest != expected:raise ValueError('Source hash mismatch: ' + str(path))
        return path
    def document(path, expected=None):return json.loads(read(path, expected).read_text())
    def source_records(root, experiment):
        manifest = document(root / 'manifest.json');audit = document(root / 'audit.json');budget = document(root / 'budget_receipt.json')
        if manifest.get('experiment') != experiment or not audit.get('audit_complete') or not audit.get('execution_complete') or not audit.get('integrity_passed') or not budget.get('all_steps_finished'):
            raise ValueError('Completed passing-integrity ' + experiment + ' records required')
        for d in (manifest, audit):
            for p, sha in d['evidence_sha256'].items():read(p, sha)
        return manifest, audit, budget
    cm, ca, cb = source_records(coarse, 'E109')
    fm, fa, fb = source_records(fine, 'E111')
    mid = Path(fm['source_run']).resolve();mm, ma, mb = source_records(mid, 'E110')
    if Path(mm['source_run']).resolve() != coarse.resolve():raise ValueError('E109/E110/E111 lineage differs')
    if file_sha256(read(fm['source_audit'])) != file_sha256(read(coarse / 'audit.json')):
        raise ValueError('Original selection audit is not supplied E109 audit')
    statistics = document(cm['statistics'])
    for manifest in (mm, fm):
        if file_sha256(read(manifest['statistics'])) != file_sha256(read(cm['statistics'])):
            raise ValueError('Cached statistics differ across sources')
    targets = statistics['targets']
    if cm['targets'] != targets or mm['original_targets'] != targets or fm['original_targets'] != targets:
        raise ValueError('Original complete target order changed')
    if cm['stimulus_nS'] != mm['stimulus_nS'] or cm['stimulus_nS'] != fm['stimulus_nS']:
        raise ValueError('Source pulse amplitudes differ')
    identities, masks = {}, {}
    for label, expected in (('selected', SELECTED_SHA), ('parent', PARENT_SHA)):
        for root, manifest in ((coarse, cm), (mid, mm), (fine, fm)):
            model = manifest['models'][label]
            if model['sha256'] != expected or model['anchors_sha256'] != ANCHOR_SHA[label]:raise ValueError('Model/anchor identity mismatch')
            read(model['path'], expected);read(model['anchors'], ANCHOR_SHA[label])
        for name, root in (('coarse', coarse), ('fine', fine)):
            identities[label + '-' + name] = worker_identity(root, label, read)
        mask = read(coarse / label / 'pair-errors.npz', ca['models'][label]['pair_errors_sha256'])
        masks[label] = saved_arrays(mask)
    identity = next(iter(identities.values()))
    if any(x != identity for x in identities.values()):raise ValueError('Forward source, package or numerical setting identity differs')
    ids = saved_arrays(cm['models']['selected']['path'])['neuron_ids'].tolist()
    if ids != saved_arrays(cm['models']['parent']['path'])['neuron_ids'].tolist():raise ValueError('Neuron order differs')
    hard, failures = choose_targets(statistics, ids, masks, ca)
    if hard != fm['targets'] or hard != mm['targets'] or failures != fm['failures'] or failures != mm['failures']:
        raise ValueError('Fixed target-group selection changed')
    hard_ix = np.asarray([targets.index(t) for t in hard]);other = [t for t in targets if t not in hard]
    if not other:raise ValueError('Composite requires both target groups')
    pairs, ti, ri = observed_pairs(statistics, targets, ids)
    if len(pairs) != cm['observed_downstream_pairs']:raise ValueError('Full observed support incomplete')
    hard_pairs = [p for p in pairs if p[0] in hard]
    def layer(root, manifest, audit, label, index, scope_targets, step):
        path = root / label / ('level-' + str(index) + '.npz');key = str(path.resolve())
        if key not in layers:
            expected_rows = [r for r in audit['models'][label]['levels'] if r['neural_step_ms'] == step and not r.get('origin', '').startswith('copied')]
            if len(expected_rows) != 1:raise ValueError('Ambiguous source grid: ' + str(path))
            digest = expected_rows[0]['artifact_sha256'];read(path, digest)
            data = saved_arrays(path)
            response, control = layer_values(data, scope_targets, ids, step, ANCHOR_SHA[label])
            base = data['reference_sham_rmd_mV']
            rhythms = [{n:rhythm_metrics(base[p, :, j], .001) for j, n in enumerate(('RMDL', 'RMDR'))} for p in range(4)]
            control.update(path=key, sha256=digest, sham_rhythm_by_phase=rhythms,
                           sham_rhythm_persistent=all(m['passed'] for r in rhythms for m in r.values()),
                           neural_step_ms=step, targets=len(scope_targets))
            layers[key] = (response, control)
        return layers[key]
    def bridge(a, b):return compare_bridge(a, b, hard, hard_pairs, ids)
    models, output_hashes = {}, {}
    for label, expected in (('selected', SELECTED_SHA), ('parent', PARENT_SHA)):
        # Independent same-step links, not just historical success flags.
        old, _ = layer(coarse, cm, ca, label, 3, targets, .03125)
        replayed, _ = layer(mid, mm, ma, label, 0, hard, .03125)
        second_old, _ = layer(mid, mm, ma, label, 2, hard, .0078125)
        second_new, _ = layer(fine, fm, fa, label, 0, hard, .0078125)
        bridges = dict(E109_to_E110=bridge(old[:, hard_ix], replayed), E110_to_E111=bridge(second_old, second_new))
        bridges_passed = all(c['passed'] for b in bridges.values() for checks in b.values() for c in checks)
        rows, comparisons, ratios_all, flips_all, previous, previous_steps = [], [], [], [], None, None
        used_controls = []
        for level, (cs, fs) in enumerate(zip(COARSE_STEPS, FINE_STEPS)):
            a, controls_a = layer(coarse, cm, ca, label, level + 1, targets, cs)
            b, controls_b = layer(fine, fm, fa, label, level, hard, fs)
            response, steps = assemble_level(a, b, targets, hard, cs, fs)
            if previous_steps is not None and not np.array_equal(steps * 2, previous_steps):raise ValueError('Every target must refine by2')
            path = output / (label + '-level-' + str(level) + '.npz')
            np.savez_compressed(path, response=response, targets=np.asarray(targets), neuron_ids=np.asarray(ids),
                                neural_step_ms_by_target=steps, fine_group_targets=np.asarray(hard))
            output_hashes[str(path)] = file_sha256(path)
            values = response[:, ti, ri]
            self_values = response[:, np.arange(len(targets)), [ids.index(t) for t in targets]]
            rows.append(dict(level=level, coarse_group_step_ms=cs, fine_group_step_ms=fs,
                coarse_control=controls_a, fine_control=controls_b, artifact=str(path), sha256=output_hashes[str(path)],
                downstream_max_abs=float(abs(values).max()), self_response_max_abs=float(abs(self_values).max())))
            used_controls.extend((controls_a, controls_b))
            if previous is not None:
                checks, ratios, flips = [], [], []
                for x, y in list(zip(previous, values)) + [(previous.mean(axis=0), values.mean(axis=0))]:
                    check, ratio, flip = compare_downstream(x, y, pairs);checks.append(check);ratios.append(ratio);flips.append(flip)
                comparisons.append(dict(levels=[level - 1, level], per_phase=checks[:4], phase_mean=checks[4]))
                ratios_all.append(ratios);flips_all.append(flips)
            previous, previous_steps = values, steps
        # Include bridge-layer controls in the final control/persistence assessment.
        for key, (_, control) in layers.items():
            if Path(key).parent.name == label:used_controls.append(control)
        controls_passed = all(c['controls_passed'] and c['sham_rhythm_persistent'] for c in used_controls)
        def passed(mean):
            checks = [c['phase_mean'] for c in comparisons] if mean else [p for c in comparisons for p in c['per_phase']]
            return bool(bridges_passed and controls_passed and len(comparisons) == 2 and all(c['vector']['passed'] and c['mixed_tolerance_passed'] for c in checks))
        accuracy = {}
        for split, s in statistics['stats'].items():
            keep = [k for k, pair in enumerate(s['pairs']) if pair[0] != pair[1]]
            st = dict(pairs=[tuple(s['pairs'][k]) for k in keep], **{k:np.asarray(s[k])[keep] for k in ('n', 'mean_z', 'stouffer')})
            metrics = pair_metrics(model_pairs(response.mean(axis=0), targets, ids, st), st, permutations=0, seed=statistics['seed'])
            accuracy[split] = {k:v for k,v in metrics.items() if not k.startswith('permutation')}
        diagnostic = output / (label + '-pair-errors.npz')
        np.savez_compressed(diagnostic, pairs=np.asarray(pairs), tolerance_ratios=np.asarray(ratios_all), active_sign_flips=np.asarray(flips_all))
        output_hashes[str(diagnostic)] = file_sha256(diagnostic)
        models[label] = dict(checkpoint_sha256=expected, anchors_sha256=ANCHOR_SHA[label], bridges=bridges,
            bridges_passed=bridges_passed, controls_and_sham_rhythms_passed=controls_passed, levels=rows, comparisons=comparisons,
            full_atlas_per_phase_checks_passed=passed(False), full_atlas_phase_mean_checks_passed=passed(True),
            descriptive_accuracy=accuracy, accuracy_numerically_qualified=passed(True),
            original_E109_flags={k:ca['models'][label].get(k) for k in ('per_phase_checks_passed', 'phase_mean_checks_passed')},
            original_E111_flags={k:fa['models'][label].get(k) for k in ('targeted_per_phase_checks_passed', 'targeted_phase_mean_checks_passed')})
    for path, digest in evidence.items():
        if file_sha256(path) != digest:raise ValueError('Source changed during composite audit: ' + path)
    result = dict(experiment='E112', audit_complete=True, integrity_passed=True, simulation_performed=False,
        targets=len(targets), observed_downstream_pairs=len(pairs),
        targets_without_observed_pairs=[t for t in targets if t not in {p[0] for p in pairs}],
        protocol=dict(fine_group_targets=hard, fine_group_steps_ms=list(FINE_STEPS), coarse_group_targets=other,
                      coarse_group_steps_ms=list(COARSE_STEPS), same_states=True, assignment='Original failure-mask target union; no reassignment'),
        source_budgets=dict(E109=cb, E110=mb, E111=fb), forward_identities=identities,
        source_failures=failures, models=models, input_sha256=evidence, output_sha256=output_hashes,
        model_promoted=False, uniform_finest_grid_claim=False, trainable_parameters=[],
        limits=['A separately declared target-dependent numerical protocol; original homogeneous-grid failures remain unchanged.',
                'No new forward simulation, fitted parameters, random draws or model/behavior promotion.',
                'Cached E106 statistics reused; no raw NWB reprocessing. Repeated test inspection is not a fresh holdout.',
                'Saved means and control differences checked; target-wise long trajectories not available for recomputing means.',
                'No independent solver/continuum error bound, new-anchor robustness, physiological amplitude calibration, body or sensory-control qualification.',
                'Any subsequently changed weights require their own numerical and rhythm revalidation.'])
    write(output / 'summary.json', result)
    return result


def report(result, output):
    print({k:result[k] for k in ('audit_complete', 'integrity_passed', 'targets', 'observed_downstream_pairs', 'targets_without_observed_pairs')})
    print('Target-dependent protocol:', result['protocol'])
    for label, m in result['models'].items():
        print(label, {k:m[k] for k in ('bridges_passed', 'controls_and_sham_rhythms_passed', 'full_atlas_per_phase_checks_passed', 'full_atlas_phase_mean_checks_passed', 'accuracy_numerically_qualified')})
        for c in m['comparisons']:
            print('levels', c['levels'], 'phase checks:', [{'vector':p['vector'], **{k:p[k] for k in ('mixed_failures', 'maximum_tolerance_ratio', 'active_sign_flips')}} for p in c['per_phase']])
            print('phase-mean:', c['phase_mean'])
        print('descriptive accuracy:', m['descriptive_accuracy'])
    print('Report:', output / 'summary.json')
    print('Saved-data audit only; target-dependent grid, original failures unchanged; no model/behavior promotion.')


def main():
    started = time.monotonic()
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coarse-run', type=Path, required=True);p.add_argument('--fine-run', type=Path, required=True);p.add_argument('--output-dir', type=Path)
    args = p.parse_args();coarse, fine = args.coarse_run.resolve(), args.fine_run.resolve()
    output = (args.output_dir or fine.with_name('atlas-composite-audit-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    repo = Path(__file__).resolve().parents[1]
    if any(output == q or output in q.parents or q in output.parents for q in (coarse, fine, repo)):
        p.error('Separate external output required')
    output.mkdir(parents=True, exist_ok=False)
    print('Starting saved-data composite audit; source models/results remain unchanged.', flush=True)
    try:
        capture_run_record(output, repo, args, phase='atlas-composite-saved-data-audit')
        result = analyze(coarse, fine, output)
        result['elapsed_seconds'] = time.monotonic() - started
        write(output / 'summary.json', result);report(result, output)
        print('Elapsed seconds:', result['elapsed_seconds'])
    except Exception as exc:
        write(output / 'failure.json', dict(audit_complete=False, integrity_passed=False, error=repr(exc),
                                           elapsed_seconds=time.monotonic() - started, model_promoted=False))
        raise


if __name__ == '__main__':main()
