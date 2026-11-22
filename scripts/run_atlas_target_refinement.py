"""E110/E111 fixed-target refinement with pinned states and automatic audits."""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scripts.experiment_records import capture_run_record, file_sha256
from scripts.selected_atlas_audit_checks import saved_arrays, write, compare_responses
from scripts.run_atlas_shared_state import worker as atlas_worker, ANCHOR_SHA, SELECTED_SHA, PARENT_SHA
from scripts.run_common_state import coordinate
from scripts.atlas_shared_checks import observed_pairs, compare_downstream
from scripts.atlas_refinement_checks import choose_targets, layer_values

REPO = Path(__file__).resolve().parents[1]
REPLAY_STEPS = (.03125, .015625, .0078125)
PROTOCOLS = {'E110': REPLAY_STEPS, 'E111': (.0078125, .00390625, .001953125)}


def prepare(source, root):
    old = json.loads((source / 'manifest.json').read_text())
    old_audit = json.loads((source / 'audit.json').read_text())
    receipt = json.loads((source / 'budget_receipt.json').read_text())
    if not old_audit.get('audit_complete') or not old_audit.get('execution_complete') or not old_audit.get('integrity_passed') or not receipt.get('all_steps_finished'):
        raise ValueError('Completed E109 integrity audit required')
    evidence = {};dest = root / 'inputs';dest.mkdir()
    def pin(path, expected=None, copy_name=None):
        path = Path(path);digest = file_sha256(path)
        if expected is not None and digest != expected:raise ValueError('Input hash mismatch: ' + str(path))
        evidence[str(path)] = digest
        if copy_name:
            target = dest / copy_name;shutil.copy2(path, target)
            if file_sha256(target) != digest:raise ValueError('Copy hash mismatch')
            evidence[str(target)] = digest
            return target
        return path
    for name in ('manifest.json', 'audit.json', 'budget_receipt.json'):pin(source / name, copy_name='source-' + name)
    for path, sha in old['evidence_sha256'].items():pin(path, sha)
    for path, sha in old_audit['evidence_sha256'].items():pin(path, sha)
    statistics_path = pin(old['statistics'], old['evidence_sha256'][old['statistics']], 'statistics.json')
    statistics = json.loads(statistics_path.read_text())
    if statistics['targets'] != old['targets']:raise ValueError('Original full-target support changed')
    models, masks, mask_paths = {}, {}, {}
    for label, expected in (('selected', SELECTED_SHA), ('parent', PARENT_SHA)):
        source_model = old_audit['models'][label]
        if not source_model.get('complete'):raise ValueError('Source model incomplete')
        model = pin(old['models'][label]['path'], expected, label + '.npz')
        anchor = pin(old['models'][label]['anchors'], ANCHOR_SHA[label], label + '-anchors.npz')
        mask_path = pin(source / label / 'pair-errors.npz', source_model['pair_errors_sha256'], label + '-source-errors.npz')
        mask_paths[label] = str(mask_path);masks[label] = saved_arrays(mask_path)
        if len(source_model['levels']) != 4 or source_model['levels'][3]['neural_step_ms'] != .03125:
            raise ValueError('Source finest baseline unavailable')
        baseline = pin(source / label / 'level-3.npz', source_model['levels'][3]['artifact_sha256'], label + '-baseline.npz')
        models[label] = dict(path=str(model), sha256=expected, anchors=str(anchor), anchors_sha256=ANCHOR_SHA[label],
                             baseline=str(baseline), baseline_sha256=file_sha256(baseline))
    ids = saved_arrays(models['selected']['path'])['neuron_ids'].tolist()
    targets, failures = choose_targets(statistics, ids, masks, old_audit)
    subset_pairs = [p for p in observed_pairs(statistics, statistics['targets'], ids)[0] if p[0] in targets]
    result = dict(experiment='E110', source_run=str(source), source_audit=str(dest / 'source-audit.json'),
        models=models, source_masks=mask_paths, evidence_sha256=evidence, statistics=str(statistics_path),
        original_targets=statistics['targets'], targets=targets, stimulus_nS=old['stimulus_nS'], failures=failures,
        observed_downstream_pairs=len(subset_pairs), targets_without_observed_pairs=[t for t in targets if t not in {p[0] for p in subset_pairs}],
        neural_step_ms=list(REPLAY_STEPS), baseline_step_ms=.03125, trainable_parameters=[],
        selection='Union of ALL last-two-comparison mixed failures or active sign flips, all phases/means and both models; plus fixed ADAL/AIYL/AIYR controls')
    write(root / 'manifest.json', result)
    print('Numerical-failure target union plus fixed controls:', targets, flush=True)
    for label, rows in failures.items():print('Source failed pairs', label, [(r['target'], r['receiver']) for r in rows], flush=True)
    return result


def prepare_continuation(source, root):
    """Keep the complete prior target union and replace only the baseline/grid."""
    old = json.loads((source / 'manifest.json').read_text())
    previous = json.loads((source / 'audit.json').read_text())
    receipt = json.loads((source / 'budget_receipt.json').read_text())
    if old.get('experiment') != 'E110' or not previous.get('audit_complete') or not previous.get('execution_complete') or not previous.get('integrity_passed') or not receipt.get('all_steps_finished'):
        raise ValueError('Completed E110 integrity audit required for E111')
    dest = root / 'inputs';dest.mkdir();evidence = {}
    def pin(path, expected=None, name=None):
        path = Path(path);digest = file_sha256(path)
        if expected is not None and digest != expected:raise ValueError('Continuation input hash mismatch: ' + str(path))
        evidence[str(path)] = digest
        if name:
            copy = dest / name;shutil.copy2(path, copy)
            if file_sha256(copy) != digest:raise ValueError('Continuation copy mismatch')
            evidence[str(copy)] = digest
            return copy
        return path
    for name in ('manifest.json', 'audit.json', 'budget_receipt.json'):pin(source / name, name='preceding-' + name)
    for path, digest in old['evidence_sha256'].items():pin(path, digest)
    for path, digest in previous['evidence_sha256'].items():pin(path, digest)
    stats_path = pin(old['statistics'], old['evidence_sha256'][old['statistics']], 'statistics.json')
    original_audit = pin(old['source_audit'], old['evidence_sha256'][old['source_audit']], 'original-selection-audit.json')
    mask_paths, masks, models = {}, {}, {}
    for label, expected in (('selected', SELECTED_SHA), ('parent', PARENT_SHA)):
        m = previous['models'][label]
        if not m.get('complete') or not all(x['passed'] for x in m.get('baseline_reproduction', [])) or len(m.get('baseline_reproduction', [])) != 5:
            raise ValueError('Complete E110 baseline reproduction required')
        if len(m['levels']) != 4 or m['levels'][-1]['neural_step_ms'] != PROTOCOLS['E111'][0]:raise ValueError('E110 finest baseline missing')
        model = pin(old['models'][label]['path'], expected, label + '.npz')
        anchor = pin(old['models'][label]['anchors'], ANCHOR_SHA[label], label + '-anchors.npz')
        mask = pin(old['source_masks'][label], old['evidence_sha256'][old['source_masks'][label]], label + '-source-errors.npz')
        mask_paths[label] = str(mask);masks[label] = saved_arrays(mask)
        baseline = pin(source / label / 'level-2.npz', m['levels'][-1]['artifact_sha256'], label + '-baseline.npz')
        models[label] = dict(path=str(model), sha256=expected, anchors=str(anchor), anchors_sha256=ANCHOR_SHA[label],
                             baseline=str(baseline), baseline_sha256=file_sha256(baseline))
    statistics = json.loads(stats_path.read_text());ids = saved_arrays(models['selected']['path'])['neuron_ids'].tolist()
    targets, failures = choose_targets(statistics, ids, masks, json.loads(original_audit.read_text()))
    if targets != old['targets'] or failures != old['failures'] or statistics['targets'] != old['original_targets']:
        raise ValueError('Continuation must retain exact original target selection')
    result = dict(experiment='E111', source_run=str(source), source_audit=str(original_audit), models=models,
        source_masks=mask_paths, evidence_sha256=evidence, statistics=str(stats_path), original_targets=old['original_targets'],
        targets=targets, stimulus_nS=old['stimulus_nS'], failures=failures, observed_downstream_pairs=old['observed_downstream_pairs'],
        targets_without_observed_pairs=old['targets_without_observed_pairs'], neural_step_ms=list(PROTOCOLS['E111']),
        baseline_step_ms=PROTOCOLS['E111'][0], baseline_targets=targets, trainable_parameters=[],
        selection='Exact E110 target union and original E109 mask lineage retained; no further target selection')
    write(root / 'manifest.json', result)
    print('E111 unchanged target union:', targets, 'steps_ms:', result['neural_step_ms'], flush=True)
    return result


def protocol_steps(manifest):
    if manifest.get('experiment') not in PROTOCOLS:raise ValueError('Unknown refinement experiment')
    steps = PROTOCOLS[manifest['experiment']]
    if manifest['neural_step_ms'] != list(steps) or manifest['baseline_step_ms'] != steps[0]:
        raise ValueError('Refinement protocol/grid changed')
    return steps


def audit(root, execution_complete):
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    manifest_path = root / 'manifest.json';manifest = json.loads(manifest_path.read_text())
    steps = protocol_steps(manifest)
    hashes = dict(manifest['evidence_sha256']);hashes[str(manifest_path)] = file_sha256(manifest_path)
    statistics = json.loads(Path(manifest['statistics']).read_text())
    source_audit = json.loads(Path(manifest['source_audit']).read_text())
    masks = {label:saved_arrays(path) for label, path in manifest['source_masks'].items()}
    ids = saved_arrays(manifest['models']['selected']['path'])['neuron_ids'].tolist()
    targets, failures = choose_targets(statistics, ids, masks, source_audit)
    if targets != manifest['targets'] or failures != manifest['failures'] or statistics['targets'] != manifest['original_targets']:
        raise ValueError('Failure-driven target selection changed')
    all_pairs = observed_pairs(statistics, statistics['targets'], ids)[0]
    pairs = [p for p in all_pairs if p[0] in targets]
    if len(pairs) != manifest['observed_downstream_pairs']:raise ValueError('Targeted observed support changed')
    problems, models = [], {}
    for label, expected in (('selected', SELECTED_SHA), ('parent', PARENT_SHA)):
        try:
            model = manifest['models'][label]
            if model['sha256'] != expected or file_sha256(model['path']) != expected or file_sha256(model['anchors']) != ANCHOR_SHA[label] or model['anchors_sha256'] != ANCHOR_SHA[label]:
                raise ValueError('Pinned model/full-state anchors changed')
            model_ids = saved_arrays(model['path'])['neuron_ids'].tolist()
            if model_ids != ids:raise ValueError('Neuron ordering differs across fixed models')
            summary_path = root / label / 'summary.json';summary = json.loads(summary_path.read_text());hashes[str(summary_path)] = file_sha256(summary_path)
            if summary['checkpoint_sha256'] != expected or summary['anchors_sha256'] != ANCHOR_SHA[label] or summary['manifest_sha256'] != hashes[str(manifest_path)]:
                raise ValueError('Worker identity mismatch')
            if len(summary['levels']) > 3:raise ValueError('Unexpected fine grids')
            baseline_path = Path(model['baseline'])
            if file_sha256(baseline_path) != model['baseline_sha256']:raise ValueError('Original baseline changed')
            baseline_targets = statistics['targets'] if manifest['experiment'] == 'E110' else targets
            if manifest.get('baseline_targets', baseline_targets) != baseline_targets:raise ValueError('Baseline target scope changed')
            entries = [(baseline_path, baseline_targets, steps[0], model['baseline_sha256'], 'copied prior baseline')]
            for i, row in enumerate(summary['levels']):
                path = root / label / ('level-' + str(i) + '.npz')
                if row['level'] != i or Path(row['artifact']).resolve() != path.resolve():raise ValueError('Fine-grid ordering changed')
                entries.append((path, targets, steps[i], row['sha256'], 'new targeted replay'))
            levels, comparisons, ratios_all, flips_all, previous = [], [], [], [], None
            reproduction = None
            for path, layer_targets, step, expected_sha, origin in entries:
                if file_sha256(path) != expected_sha:raise ValueError('Forward artifact hash mismatch')
                hashes[str(path)] = expected_sha
                data = saved_arrays(path)
                response, controls = layer_values(data, layer_targets, ids, step, ANCHOR_SHA[label])
                ti = np.asarray([layer_targets.index(t) for t, _ in pairs]);ri = np.asarray([ids.index(r) for _, r in pairs])
                values = response[:, ti, ri]
                base = data['reference_sham_rmd_mV']
                rhythm = [{name:rhythm_metrics(base[phase, :, j], .001) for j, name in enumerate(('RMDL', 'RMDR'))} for phase in range(4)]
                levels.append(dict(neural_step_ms=step, origin=origin, simulated_targets=len(layer_targets),
                    evaluated_targets=len(targets), observed_downstream_pairs=len(pairs), **controls,
                    sham_rhythm_by_phase=rhythm, sham_rhythm_persistent=all(m['passed'] for r in rhythm for m in r.values()),
                    downstream_max_abs=float(abs(values).max()), artifact_sha256=expected_sha))
                if len(levels) == 2:
                    reproduction = [compare_responses(a, b, rtol=1e-6, atol=1e-12, minimum_correlation=.999999)
                                    for a, b in list(zip(previous, values)) + [(previous.mean(axis=0), values.mean(axis=0))]]
                elif previous is not None:
                    checks, ratios, flips = [], [], []
                    for a, b in list(zip(previous, values)) + [(previous.mean(axis=0), values.mean(axis=0))]:
                        check, ratio, flip = compare_downstream(a, b, pairs);checks.append(check);ratios.append(ratio);flips.append(flip)
                    comparisons.append(dict(steps_ms=[levels[-2]['neural_step_ms'], step], per_phase=checks[:4], phase_mean=checks[4]))
                    ratios_all.append(ratios);flips_all.append(flips)
                previous = values
            complete = len(levels) == 4
            controls_passed = bool(complete and all(l['controls_passed'] for l in levels))
            rhythms_passed = bool(complete and all(l['sham_rhythm_persistent'] for l in levels))
            diagnostics = root / label / 'targeted-pair-errors.npz'
            if diagnostics.exists():raise ValueError('Refusing to overwrite prior targeted audit artifact')
            np.savez_compressed(diagnostics, pairs=np.asarray(pairs), tolerance_ratios=np.asarray(ratios_all), active_sign_flips=np.asarray(flips_all))
            def passed(mean):
                checks = [c['phase_mean'] for c in comparisons] if mean else [p for c in comparisons for p in c['per_phase']]
                return bool(execution_complete and complete and controls_passed and rhythms_passed and reproduction is not None and all(r['passed'] for r in reproduction) and len(comparisons) == 2
                            and all(p['vector']['passed'] and p['mixed_tolerance_passed'] for p in checks))
            models[label] = dict(complete=complete, checkpoint_sha256=expected, anchors_sha256=ANCHOR_SHA[label],
                levels=levels, comparisons=comparisons, baseline_reproduction=reproduction, controls_passed=controls_passed, sham_rhythms_preserved=rhythms_passed,
                targeted_per_phase_checks_passed=passed(False), targeted_phase_mean_checks_passed=passed(True),
                pair_errors_sha256=file_sha256(diagnostics))
        except Exception as exc:
            problems.append(label + ': ' + repr(exc));models[label] = dict(complete=False, error=repr(exc))
    for path, sha in hashes.items():
        if not Path(path).is_file() or file_sha256(path) != sha:problems.append('Consumed input changed: ' + path)
    if problems:
        for m in models.values():m.update(targeted_per_phase_checks_passed=False, targeted_phase_mean_checks_passed=False)
    result = dict(experiment=manifest['experiment'], audit_complete=True, execution_complete=execution_complete, integrity_passed=not problems,
        problems=problems, targets=targets, observed_downstream_pairs=len(pairs), source_failures=failures,
        targets_without_observed_pairs=manifest['targets_without_observed_pairs'], models=models, evidence_sha256=hashes,
        trainable_parameters=[], model_promoted=False, full_atlas_qualified=False,
        limits=['Numerical-failure-selected subset only; unsimulated finer-grid targets and full atlas are not qualified.',
                'Exact prior anchors/models, pulse clocks, vector criterion and mixed tolerance unchanged. Subset vector scale differs from original full vector.',
                'No original failure overwritten, no test-score selection/re-evaluation, fitting, independent solver or biological promotion.',
                'Cached observed support reused; no raw NWB reprocessing, new anchor robustness, body or sensory-control validation.',
                'Saved means/control differences audited; target-wise long trajectories intentionally omitted.'])
    write(root / 'audit.json', result)
    return result


def report(root):
    if not (root / 'audit.json').exists():print('No final audit; log:', str(root) + '.log');return
    s = json.loads((root / 'audit.json').read_text());print('Run:', root)
    print({k:s.get(k) for k in ('audit_complete', 'execution_complete', 'integrity_passed', 'problems', 'targets', 'observed_downstream_pairs', 'targets_without_observed_pairs')})
    for label, rows in s.get('source_failures', {}).items():print('Original failed pairs', label, [(r['target'], r['receiver']) for r in rows])
    for label, model in s.get('models', {}).items():
        print(label, {k:model.get(k) for k in ('complete', 'checkpoint_sha256', 'anchors_sha256', 'controls_passed', 'sham_rhythms_preserved', 'targeted_per_phase_checks_passed', 'targeted_phase_mean_checks_passed', 'error')})
        print('Same-step baseline reproduction:', model.get('baseline_reproduction'))
        for comparison in model.get('comparisons', []):
            print('steps_ms', comparison['steps_ms'], 'per-phase:', [{k:p[k] for k in ('mixed_failures', 'maximum_tolerance_ratio', 'active_sign_flips')} | {'vector':p['vector'], 'worst_pairs':p['worst_pairs'][:3]} for p in comparison['per_phase']])
            print('phase-mean:', comparison['phase_mean'])
    if (root / 'budget_receipt.json').exists():print('Budget:', json.loads((root / 'budget_receipt.json').read_text()))
    print('Audit:', root / 'audit.json');print('Targeted diagnostic only; no full-atlas, fitting or behavior promotion.')


def main():
    started = time.monotonic();p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path);p.add_argument('--output-dir', type=Path, required=True);p.add_argument('--minutes', type=float, default=120.)
    p.add_argument('--worker', choices=('selected', 'parent'));p.add_argument('--coordinate', action='store_true');p.add_argument('--report', action='store_true')
    p.add_argument('--continue-refinement', action='store_true', help='E111: keep E110 target union and refine from its finest baseline')
    args = p.parse_args();root = args.output_dir.resolve()
    if args.report:report(root);return 0
    if args.worker:
        atlas_worker(root, args.worker, steps=protocol_steps(json.loads((root / 'manifest.json').read_text())));return 0
    if args.coordinate:return coordinate(root, 'scripts.run_atlas_target_refinement')
    if args.source_run is None:p.error('--source-run required')
    if not math.isfinite(args.minutes) or not 5 <= args.minutes <= 1440:p.error('Budget must be5..1440minutes')
    source = args.source_run.resolve()
    if any(root == q or root in q.parents or q in root.parents for q in (source, REPO)):p.error('Separate external output required')
    root.mkdir(parents=True, exist_ok=False);code, timeout = 1, False
    try:
        capture_run_record(root, REPO, args, phase='atlas-target-refinement')
        if args.continue_refinement:prepare_continuation(source, root)
        else:prepare(source, root)
        from scripts.total_job_budget import run_group
        seconds = started + args.minutes * 60 - time.monotonic() - 60
        if seconds <= 0:raise TimeoutError('Preparation exhausted budget')
        code, timeout = run_group([sys.executable, '-u', '-m', 'scripts.run_atlas_target_refinement', '--coordinate', '--output-dir', str(root)], seconds)
    except Exception as exc:write(root / 'failure.json', dict(error=repr(exc)));print('Failure:', repr(exc), flush=True)
    finally:
        try:outcome = audit(root, code == 0 and not timeout)
        except Exception as exc:
            outcome = dict(audit_complete=False, integrity_passed=False, problems=[repr(exc)], models={}, model_promoted=False);write(root / 'audit.json', outcome)
        write(root / 'budget_receipt.json', dict(exit_code=code, timed_out=timeout, total_minutes=args.minutes,
            total_elapsed_seconds=time.monotonic() - started, audit_complete=outcome['audit_complete'],
            all_steps_finished=bool(code == 0 and not timeout and outcome['integrity_passed'] and len(outcome['models']) == 2 and all(m.get('complete') for m in outcome['models'].values())), model_promoted=False))
        report(root)
    return code


if __name__ == '__main__':raise SystemExit(main())
