"""E109 all-target atlas numerical validation from pinned E108 complete states."""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scripts.experiment_records import capture_run_record, file_sha256
from scripts.selected_atlas_audit_checks import saved_arrays, write
from scripts.run_common_state import FIELDS, STEPS, SELECTED_SHA, PARENT_SHA, coordinate
from scripts.atlas_shared_checks import observed_pairs, compare_downstream

REPO = Path(__file__).resolve().parents[1]
BATCH = 8
ANCHOR_SHA = {'selected': '882a7f41805e952048dc8801e42bf5f534f301a858718db290d5a3f40c76c879',
              'parent': 'bb941a77275d39be693a1f2d9a4548365599e1624fe1e1c09b7728028241840b'}


def prepare(source, root):
    old = json.loads((source / 'manifest.json').read_text())
    old_audit = json.loads((source / 'audit.json').read_text())
    receipt = json.loads((source / 'budget_receipt.json').read_text())
    if not old_audit.get('audit_complete') or not old_audit.get('integrity_passed') or not old_audit.get('execution_complete') or not receipt.get('all_steps_finished'):
        raise ValueError('Completed E108 audit required')
    evidence = {}
    def pin(path, expected=None, copy_name=None):
        path = Path(path);digest = file_sha256(path)
        if expected is not None and digest != expected:raise ValueError('Source hash mismatch: ' + str(path))
        evidence[str(path)] = digest
        if copy_name:
            dest = root / 'inputs' / copy_name;shutil.copy2(path, dest)
            if file_sha256(dest) != digest:raise ValueError('Copy mismatch')
            evidence[str(dest)] = digest
            return dest
        return path
    (root / 'inputs').mkdir()
    for name in ('manifest.json', 'audit.json', 'budget_receipt.json'):pin(source / name)
    for path, sha in old_audit['evidence_sha256'].items():pin(path, sha)
    for path, sha in old['evidence_sha256'].items():pin(path, sha)
    e107_path = Path(old['source_run']) / 'manifest.json'
    pin(e107_path, old['evidence_sha256'][str(e107_path)])
    e107 = json.loads(e107_path.read_text())
    job_path = Path(e107['source_audit']) / 'E104-network_control-job.json'
    pin(job_path, old['evidence_sha256'][str(job_path)])
    job = json.loads(job_path.read_text())
    manifest_path = pin(job['manifest'], job['manifest_sha256'])
    atlas_manifest = json.loads(manifest_path.read_text())
    declared = {e['path']: e['sha256'] for e in atlas_manifest['evidence']}
    statistics_path = pin(job['statistics'], declared[job['statistics']], 'statistics.json')
    data_path = Path(job['statistics']).parent / 'data_evidence.json'
    pin(data_path, declared[str(data_path)], 'original_data_evidence.json')
    statistics = json.loads(statistics_path.read_text());targets = statistics['targets']
    if not targets or targets != sorted(set(targets)):raise ValueError('Invalid complete atlas target order')
    models = {}
    for label, expected in (('selected', SELECTED_SHA), ('parent', PARENT_SHA)):
        previous = old_audit['models'][label]
        if not previous.get('per_phase_response_checks_passed') or not previous.get('phase_mean_response_checks_passed'):
            raise ValueError('E108 matched-state protocol did not pass: ' + label)
        model = pin(old['models'][label]['path'], expected, label + '.npz')
        anchor = pin(source / label / 'anchors.npz', ANCHOR_SHA[label], label + '-anchors.npz')
        if previous['anchors_sha256'] != ANCHOR_SHA[label] or previous['checkpoint_sha256'] != expected:
            raise ValueError('E108 audited model/anchor identity mismatch')
        ids = saved_arrays(model)['neuron_ids'].tolist()
        if any(t not in ids for t in targets):raise ValueError('Unmodeled atlas target; coverage incomplete')
        pairs, _, _ = observed_pairs(statistics, targets, ids)
        models[label] = dict(path=str(model), sha256=expected, anchors=str(anchor), anchors_sha256=ANCHOR_SHA[label])
    result = dict(experiment='E109', source_run=str(source), models=models, evidence_sha256=evidence,
        targets=targets, stimulus_nS=old['stimulus_nS'], statistics=str(statistics_path),
        observed_downstream_pairs=len(pairs), batch_size=BATCH, neural_step_ms=list(STEPS), trainable_parameters=[],
        targets_without_observed_pairs=[t for t in targets if t not in {p[0] for p in pairs}],
        source_note='Cached E106 split statistics/data-evidence hashes checked; raw NWB files not re-read or rehashed',
        selection='All cached targets and union of measured downstream pairs; both fixed models; no test-based reselection')
    write(root / 'manifest.json', result)
    return result


def worker(root, label, steps=STEPS):
    import jax
    import jax.numpy as jnp
    from r_elegans.brain.conductance_network import load_network, ConductanceNetworkState
    from r_elegans.brain.common_state_replay import make_replay
    if not jax.config.x64_enabled or jax.default_backend() != 'gpu':raise ValueError('Float64 GPU required')
    manifest_path = root / 'manifest.json';manifest_sha = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text());model = manifest['models'][label]
    for key, digest in (('path', model['sha256']), ('anchors', model['anchors_sha256'])):
        if file_sha256(model[key]) != digest:raise ValueError('Pinned input changed')
    out = root / label;out.mkdir()
    capture_run_record(out, REPO, dict(root=str(root), label=label), phase='atlas-shared-state-worker')
    p, f, c = load_network(model['path'])
    p, f = jax.tree.map(lambda a: a.astype(jnp.float64) if jnp.issubdtype(a.dtype, jnp.floating) else a, (p, f))
    ids = saved_arrays(model['path'])['neuron_ids'].tolist();rmd = [ids.index(n) for n in ('RMDL', 'RMDR')]
    anchors = saved_arrays(model['anchors']);starts = ConductanceNetworkState(*(jnp.asarray(anchors['start_' + k]) for k in FIELDS))
    targets = manifest['targets'];levels = []
    for level, step in enumerate(steps):
        began = time.monotonic();forward = make_replay(c, rmd, step)
        means_all = np.empty((4, len(targets), len(ids)), dtype=np.float64)
        controls, deltas, labels, base = [], [], [], None
        for begin in range(0, len(targets), BATCH):
            chunk = targets[begin:begin + BATCH];n = len(chunk)
            gs = np.zeros((BATCH + 2, len(ids)))
            gs[np.arange(n), [ids.index(t) for t in chunk]] = manifest['stimulus_nS']
            means, _, voltage = forward(p, f, starts, jnp.asarray(gs))
            means = np.asarray(means);means_all[:, begin:begin + n] = means[:, :n]
            control_indices = list(range(n, BATCH + 2))
            v = np.asarray(voltage[:, jnp.asarray(control_indices)])
            if base is None:base = v[:, 0].copy()
            controls.append(np.moveaxis(means[:, control_indices], 1, 0))
            deltas.append(np.moveaxis(v - base[:, None], 1, 0))
            labels.extend([f'batch-{begin // BATCH}-condition-{k}' for k in control_indices])
            print(json.dumps(dict(model=label, level=level, step_ms=step, targets_done=begin + n, targets_total=len(targets))), flush=True)
        path = out / ('level-' + str(level) + '.npz')
        np.savez_compressed(path, indicator_means=means_all, control_means=np.concatenate(controls),
            control_voltage_delta=np.concatenate(deltas), control_labels=labels, reference_sham_rmd_mV=base,
            targets=np.asarray(targets), anchors_sha256=model['anchors_sha256'], neural_step_ms=step, sample_dt_s=.001)
        levels.append(dict(level=level, artifact=str(path), sha256=file_sha256(path), elapsed_seconds=time.monotonic() - began))
        write(out / 'summary.json', dict(checkpoint_sha256=model['sha256'], anchors_sha256=model['anchors_sha256'],
              manifest_sha256=manifest_sha, levels=levels))
    if file_sha256(model['path']) != model['sha256'] or file_sha256(model['anchors']) != model['anchors_sha256'] or file_sha256(manifest_path) != manifest_sha:
        raise ValueError('Pinned input changed during worker')


def audit(root, execution_complete):
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    from r_elegans.atlas_evaluation import pair_metrics, model_pairs
    manifest_path = root / 'manifest.json';manifest = json.loads(manifest_path.read_text())
    hashes = dict(manifest['evidence_sha256']);hashes[str(manifest_path)] = file_sha256(manifest_path)
    statistics = json.loads(Path(manifest['statistics']).read_text());targets = manifest['targets']
    if targets != statistics['targets']:raise ValueError('Full atlas targets changed')
    if set(manifest['models']) != {'selected', 'parent'}:raise ValueError('Both fixed models required')
    problems, models = [], {}
    for label, expected in (('selected', SELECTED_SHA), ('parent', PARENT_SHA)):
        try:
            model = manifest['models'][label]
            if file_sha256(model['path']) != expected or model['sha256'] != expected or file_sha256(model['anchors']) != ANCHOR_SHA[label] or model['anchors_sha256'] != ANCHOR_SHA[label]:
                raise ValueError('Model or shared full-state anchors changed')
            ids = saved_arrays(model['path'])['neuron_ids'].tolist()
            pairs, ti, ri = observed_pairs(statistics, targets, ids)
            if len(pairs) != manifest['observed_downstream_pairs']:raise ValueError('Observed pair coverage changed')
            summary_path = root / label / 'summary.json';summary = json.loads(summary_path.read_text());hashes[str(summary_path)] = file_sha256(summary_path)
            if summary['manifest_sha256'] != hashes[str(manifest_path)] or summary['checkpoint_sha256'] != expected or summary['anchors_sha256'] != ANCHOR_SHA[label]:
                raise ValueError('Worker input identity mismatch')
            if len(summary['levels']) > 4:raise ValueError('Unexpected grids')
            rows, comparisons, ratio_records, flip_records, previous = [], [], [], [], None
            for i, item in enumerate(summary['levels']):
                path = root / label / ('level-' + str(i) + '.npz')
                if item['level'] != i or Path(item['artifact']).resolve() != path.resolve() or file_sha256(path) != item['sha256']:
                    raise ValueError('Missing/changed grid artifact')
                hashes[str(path)] = item['sha256'];data = saved_arrays(path)
                if str(data['anchors_sha256']) != ANCHOR_SHA[label] or float(data['neural_step_ms']) != STEPS[i] or float(data['sample_dt_s']) != .001 or data['targets'].tolist() != targets:
                    raise ValueError('Grid/anchor/target metadata mismatch')
                ncontrols = 2 * math.ceil(len(targets) / BATCH) + (-len(targets)) % BATCH
                means, cm, dv, base = (data[k] for k in ('indicator_means', 'control_means', 'control_voltage_delta', 'reference_sham_rmd_mV'))
                if means.shape != (4, len(targets), 302) or cm.shape != (ncontrols, 4, 302) or dv.shape != (ncontrols, 4, 12000, 2) or base.shape != (4, 12000, 2):
                    raise ValueError('Incomplete target/control coverage')
                expected_labels = [f'batch-{begin // BATCH}-condition-{k}' for begin in range(0, len(targets), BATCH)
                                   for k in range(min(BATCH, len(targets) - begin), BATCH + 2)]
                if data['control_labels'].tolist() != expected_labels:raise ValueError('Missing duplicate/padded control labels')
                if any(a.dtype != np.float64 or not np.isfinite(a).all() for a in (means, cm, dv, base)):
                    raise ValueError('Nonfinite/non-float64 trial arrays')
                control_error = float(abs(cm - cm[0]).max());trace_error = float(abs(dv).max())
                controls_pass = control_error <= 1e-12 and trace_error <= 1e-12
                response = means - cm[0][:, None]
                values = response[:, ti, ri]
                self_values = response[:, np.arange(len(targets)), [ids.index(t) for t in targets]]
                rhythm = [{n: rhythm_metrics(base[phase, :, j], .001) for j, n in enumerate(('RMDL', 'RMDR'))} for phase in range(4)]
                rows.append(dict(level=i, neural_step_ms=STEPS[i], targets=len(targets), downstream_pairs=len(pairs),
                    controls_passed=controls_pass, duplicate_mean_error=control_error, duplicate_trace_error=trace_error,
                    sham_rhythm_by_phase=rhythm, sham_rhythm_persistent=all(m['passed'] for r in rhythm for m in r.values()),
                    downstream_max_abs=float(abs(values).max()), self_response_max_abs=float(abs(self_values).max()),
                    phase_mean_downstream_max_abs=float(abs(values.mean(axis=0)).max()), artifact_sha256=item['sha256']))
                if previous is not None:
                    checks, ratios, flips = [], [], []
                    for a, b in list(zip(previous, values)) + [(previous.mean(axis=0), values.mean(axis=0))]:
                        check, ratio, flip = compare_downstream(a, b, pairs);checks.append(check);ratios.append(ratio);flips.append(flip)
                    comparisons.append(dict(levels=[i - 1, i], per_phase=checks[:4], phase_mean=checks[4]))
                    ratio_records.append(ratios);flip_records.append(flips)
                previous = values
            complete = len(rows) == 4
            controls = bool(complete and all(r['controls_passed'] for r in rows))
            if ratio_records:
                diagnostics_path = root / label / 'pair-errors.npz'
                # Newly computed audit output, never overwrite forward or source data.
                if diagnostics_path.exists():raise ValueError('Pair audit output already exists; refusing overwrite')
                np.savez_compressed(diagnostics_path, pairs=np.asarray(pairs), tolerance_ratios=np.asarray(ratio_records), active_sign_flips=np.asarray(flip_records))
                diagnostics_sha = file_sha256(diagnostics_path)
            else:diagnostics_sha = None
            last = comparisons[-2:]
            def pass_checks(which):
                checks = [c['phase_mean'] for c in last] if which == 'mean' else [p for c in last for p in c['per_phase']]
                return bool(execution_complete and complete and controls and len(last) == 2 and all(c['vector']['passed'] and c['mixed_tolerance_passed'] for c in checks))
            accuracy = {}
            if complete:
                average = response.mean(axis=0)
                for split, s in statistics['stats'].items():
                    # Explicitly remove any self pair even if cached data ever includes it.
                    keep = [k for k, pair in enumerate(s['pairs']) if pair[0] != pair[1]]
                    st = dict(pairs=[tuple(s['pairs'][k]) for k in keep], **{k:np.asarray(s[k])[keep] for k in ('mean_z', 'n', 'stouffer')})
                    metrics = pair_metrics(model_pairs(average, targets, ids, st), st, permutations=0, seed=statistics['seed'])
                    accuracy[split] = {k:v for k,v in metrics.items() if not k.startswith('permutation')}
            models[label] = dict(checkpoint_sha256=expected, anchors_sha256=ANCHOR_SHA[label], complete=complete,
                levels=rows, comparisons=comparisons, controls_passed=controls, per_phase_checks_passed=pass_checks('phase'),
                phase_mean_checks_passed=pass_checks('mean'), finest_descriptive_accuracy=accuracy,
                accuracy_numerically_qualified=pass_checks('mean'), pair_errors_sha256=diagnostics_sha)
        except Exception as exc:
            problems.append(label + ': ' + repr(exc));models[label] = dict(complete=False, error=repr(exc))
    for path, digest in hashes.items():
        if not Path(path).is_file() or file_sha256(path) != digest:problems.append('Consumed input changed: ' + path)
    if problems:
        for m in models.values():m.update(per_phase_checks_passed=False, phase_mean_checks_passed=False, accuracy_numerically_qualified=False)
    result = dict(experiment='E109', audit_complete=True, execution_complete=execution_complete, integrity_passed=not problems,
        problems=problems, targets=len(targets), observed_downstream_pairs=manifest['observed_downstream_pairs'], models=models,
        targets_without_observed_pairs=manifest.get('targets_without_observed_pairs', []),
        evidence_sha256=hashes, model_promoted=False, trainable_parameters=[],
        limits=['Fixed-model numerical diagnostic; no new weights, behaviors or biological validation.',
                'Cached E106 labels/support reused and hashed; raw NWB not reprocessed. Repeatedly inspected test set is not an untouched holdout.',
                'Direct self responses excluded from every downstream convergence vector and pair score; reported separately.',
                'Mixed pair tolerance1e-7+10% is an engineering convention, not empirical noise. Original vector gates also required.',
                'All controls retained as indicator means and full-time sham differences relative to saved reference; full target voltage/indicator trajectories not retained.',
                'E108 anchors reused at one reference grid/cycle; independent anchor/solver or full neural-state convergence not established.',
                'Original E106/E107 failures unchanged; no sensory start/stop, body or swimming promotion.'])
    write(root / 'audit.json', result)
    return result


def report(root):
    if not (root / 'audit.json').exists():print('No final audit; log:', str(root) + '.log');return
    s = json.loads((root / 'audit.json').read_text());print('Run:', root)
    print({k:s.get(k) for k in ('audit_complete', 'execution_complete', 'integrity_passed', 'problems', 'targets', 'observed_downstream_pairs', 'targets_without_observed_pairs')})
    for label, model in s.get('models', {}).items():
        print(label, {k:model.get(k) for k in ('complete', 'controls_passed', 'per_phase_checks_passed', 'phase_mean_checks_passed', 'accuracy_numerically_qualified', 'error')})
        for level in model.get('levels', []):print('level', level['level'], {k:level[k] for k in ('downstream_max_abs', 'self_response_max_abs', 'controls_passed', 'sham_rhythm_persistent')})
        for pair in model.get('comparisons', []):
            print('grids', pair['levels'], 'per-phase:', [{k:p[k] for k in ('mixed_failures', 'mixed_pass_fraction', 'maximum_tolerance_ratio', 'active_sign_flips')} | {'vector':p['vector']} for p in pair['per_phase']])
            print('phase-mean:', pair['phase_mean'])
        print('finest descriptive accuracy:', model.get('finest_descriptive_accuracy'))
    if (root / 'budget_receipt.json').exists():print('Budget:', json.loads((root / 'budget_receipt.json').read_text()))
    print('Audit:', root / 'audit.json');print('No fitting, reselection or model/behavior promotion.')


def main():
    started = time.monotonic();p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path);p.add_argument('--output-dir', type=Path, required=True);p.add_argument('--minutes', type=float, default=120.)
    p.add_argument('--worker', choices=('selected', 'parent'));p.add_argument('--coordinate', action='store_true');p.add_argument('--report', action='store_true');args = p.parse_args();root = args.output_dir.resolve()
    if args.report:report(root);return 0
    if args.worker:worker(root, args.worker);return 0
    if args.coordinate:return coordinate(root, 'scripts.run_atlas_shared_state')
    if args.source_run is None:p.error('--source-run required')
    if not math.isfinite(args.minutes) or not 5 <= args.minutes <= 1440:p.error('Budget must be5..1440minutes')
    source = args.source_run.resolve()
    if any(root == q or root in q.parents or q in root.parents for q in (source, REPO)):p.error('Separate external output required')
    root.mkdir(parents=True, exist_ok=False);code, timeout = 1, False
    try:
        capture_run_record(root, REPO, args, phase='atlas-shared-state')
        manifest = prepare(source, root)
        print('Full targets:', len(manifest['targets']), 'Observed downstream pairs:', manifest['observed_downstream_pairs'], flush=True)
        from scripts.total_job_budget import run_group
        seconds = started + args.minutes * 60 - time.monotonic() - 60
        if seconds <= 0:raise TimeoutError('Preparation exhausted budget')
        code, timeout = run_group([sys.executable, '-u', '-m', 'scripts.run_atlas_shared_state', '--coordinate', '--output-dir', str(root)], seconds)
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
