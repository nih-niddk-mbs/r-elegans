"""E108 common-complete-state and controlled-reference-phase pulse diagnostic."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scripts.experiment_records import capture_run_record, file_sha256
from scripts.selected_atlas_audit_checks import compare_responses, saved_arrays, write
from scripts.run_neural_convergence import SELECTED_SHA, PARENT_SHA

REPO = Path(__file__).resolve().parents[1]
STEPS = (.25, .125, .0625, .03125)
FIELDS = ('voltage_mV', 'gates', 'calcium_uM', 'indicator')
TARGETS = ('ADAL', 'AIYL', 'AIYR')


def prepare(source, root):
    manifest_path = source / 'manifest.json'
    old = json.loads(manifest_path.read_text())
    audit_path, receipt_path = source / 'audit.json', source / 'budget_receipt.json'
    old_audit = json.loads(audit_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if not old_audit.get('audit_complete') or not old_audit.get('integrity_passed') or not receipt.get('all_steps_finished'):
        raise ValueError('Completed E107 source integrity audit required; numerical failure is preserved')
    evidence = {str(p): file_sha256(p) for p in (manifest_path, audit_path, receipt_path)}
    for path, sha in old['evidence_sha256'].items():
        if file_sha256(path) != sha:
            raise ValueError('E107 source input hash mismatch: ' + path)
        evidence[path] = sha
    if not set(TARGETS).issubset(old['targets']):
        raise ValueError('Fixed diagnostic targets absent from E107 target set')
    if set(old['models']) != {'selected', 'parent'}:
        raise ValueError('Both predetermined models required')
    dest = root / 'inputs';dest.mkdir()
    models = {}
    for label, expected in (('selected', SELECTED_SHA), ('parent', PARENT_SHA)):
        source_model = Path(old['models'][label]['path'])
        if old['models'][label]['sha256'] != expected or file_sha256(source_model) != expected:
            raise ValueError('Pinned model mismatch: ' + label)
        path = dest / (label + '.npz')
        shutil.copy2(source_model, path)
        if file_sha256(path) != expected:
            raise ValueError('Copy mismatch')
        models[label] = dict(path=str(path), sha256=expected)
        evidence[str(source_model)] = expected;evidence[str(path)] = expected
    result = dict(experiment='E108', source_run=str(source), source_numerical_component_passed=old_audit['numerical_component_passed'],
                  targets=list(TARGETS), stimulus_nS=old['stimulus_nS'], models=models,
                  evidence_sha256=evidence, neural_step_ms=list(STEPS), trainable_parameters=[], random_operations=False)
    write(root / 'manifest.json', result)
    return result


def worker(root, label):
    import jax
    import jax.numpy as jnp
    from r_elegans.brain.conductance_network import load_network, ConductanceNetworkState
    from r_elegans.brain.common_state_replay import collect_anchors, replay
    if not jax.config.x64_enabled or jax.default_backend() != 'gpu':
        raise ValueError('Float64 GPU worker required')
    manifest_path = root / 'manifest.json';manifest_sha = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text());model = manifest['models'][label]
    if file_sha256(model['path']) != model['sha256']:
        raise ValueError('Model changed before worker')
    out = root / label;out.mkdir()
    capture_run_record(out, REPO, dict(root=str(root), label=label), phase='common-state-worker')
    params, fixed, config = load_network(model['path'])
    params, fixed = jax.tree.map(lambda a: a.astype(jnp.float64) if jnp.issubdtype(a.dtype, jnp.floating) else a, (params, fixed))
    ids = saved_arrays(model['path'])['neuron_ids'].tolist();rmd = [ids.index(n) for n in ('RMDL', 'RMDR')]
    print('Building finest-grid complete-state anchors:', label, flush=True)
    anchors, reference = collect_anchors(params, fixed, config, rmd)
    gs = np.zeros((5, len(ids)), dtype=np.float64)
    gs[np.arange(3), [ids.index(n) for n in TARGETS]] = manifest['stimulus_nS']
    anchor_path = out / 'anchors.npz'
    np.savez_compressed(anchor_path, **{'start_' + k: np.asarray(getattr(anchors, k)) for k in FIELDS},
                        conductances_nS=gs, reference_rmd_mV=reference.pop('reference_rmd_mV'),
                        anchor_indices=reference['indices'], reference_peak_indices=reference['peak_indices'])
    anchor_sha = file_sha256(anchor_path)
    levels = []
    for i, step_ms in enumerate(STEPS):
        started = time.monotonic()
        data = saved_arrays(anchor_path)
        if file_sha256(anchor_path) != anchor_sha:
            raise ValueError('Shared anchors changed')
        # Reconstruct ALL fields from the exact same saved bytes at every grid.
        starts = ConductanceNetworkState(*(jnp.asarray(data['start_' + k]) for k in FIELDS))
        print(json.dumps(dict(model=label, level=i, step_ms=step_ms, status='starting')), flush=True)
        means, short, voltage = replay(params, fixed, config, starts, data['conductances_nS'], rmd, step_ms)
        path = out / ('level-' + str(i) + '.npz')
        np.savez_compressed(path, indicator_means=means, rmd_voltage_mV=voltage,
                            **{'short_' + k: np.asarray(getattr(short, k)) for k in FIELDS},
                            anchor_sha256=anchor_sha, neural_step_ms=step_ms, sample_dt_s=.001)
        levels.append(dict(level=i, artifact=str(path), sha256=file_sha256(path), elapsed_seconds=time.monotonic() - started))
        write(out / 'summary.json', dict(checkpoint_sha256=model['sha256'], manifest_sha256=manifest_sha,
              anchor_sha256=anchor_sha, anchor_reference=reference, levels=levels))
        print(json.dumps(dict(model=label, level=i, status='finished', elapsed_seconds=levels[-1]['elapsed_seconds'])), flush=True)
    if file_sha256(model['path']) != model['sha256'] or file_sha256(anchor_path) != anchor_sha or file_sha256(manifest_path) != manifest_sha:
        raise ValueError('Input changed during worker')


def coordinate(root):
    processes = []
    try:
        for gpu, label in enumerate(('selected', 'parent')):
            env = dict(os.environ, JAX_PLATFORMS='cuda', CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.8')
            log = (root / (label + '.log')).open('w')
            process = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.run_common_state', '--worker', label,
                                        '--output-dir', str(root)], env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append((label, process, log))
        while any(p.poll() is None for _, p, _ in processes):
            print(json.dumps(dict(running=[label for label, p, _ in processes if p.poll() is None])), flush=True)
            time.sleep(30)
        codes = {label: p.returncode for label, p, _ in processes}
        write(root / 'worker_exit_codes.json', codes)
        return int(any(codes.values()))
    finally:
        for _, p, log in processes:
            if p.poll() is None:p.terminate()
            log.close()


def audit(root, execution_complete):
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    manifest_path = root / 'manifest.json';manifest = json.loads(manifest_path.read_text())
    hashes = dict(manifest['evidence_sha256']);hashes[str(manifest_path)] = file_sha256(manifest_path)
    problems, models = [], {}
    for label in ('selected', 'parent'):
        try:
            model = manifest['models'][label]
            expected_model = SELECTED_SHA if label == 'selected' else PARENT_SHA
            if model['sha256'] != expected_model or file_sha256(model['path']) != expected_model:
                raise ValueError('Pinned parameters changed')
            summary_path = root / label / 'summary.json';summary = json.loads(summary_path.read_text())
            hashes[str(summary_path)] = file_sha256(summary_path)
            if summary['checkpoint_sha256'] != expected_model or summary['manifest_sha256'] != hashes[str(manifest_path)]:
                raise ValueError('Worker input identity mismatch')
            anchor_path = root / label / 'anchors.npz'
            if file_sha256(anchor_path) != summary['anchor_sha256']:
                raise ValueError('Anchor hash mismatch')
            hashes[str(anchor_path)] = summary['anchor_sha256']
            anchors = saved_arrays(anchor_path)
            ids = saved_arrays(model['path'])['neuron_ids'].tolist()
            expected_g = np.zeros((5, 302));expected_g[np.arange(3), [ids.index(t) for t in TARGETS]] = manifest['stimulus_nS']
            if not np.array_equal(anchors['conductances_nS'], expected_g):
                raise ValueError('Pulse/sham condition mismatch')
            for k in FIELDS:
                a = anchors['start_' + k]
                if a.shape != ((4, 302, 35) if k == 'gates' else (4, 302)) or a.dtype != np.float64 or not np.isfinite(a).all():
                    raise ValueError('Invalid complete anchor field: ' + k)
            ix, peak = anchors['anchor_indices'], anchors['reference_peak_indices']
            if len(peak) != 2 or peak[1] <= peak[0] or not np.array_equal(ix, np.rint(peak[0] + np.arange(4) * (peak[1] - peak[0]) / 4).astype(int)):
                raise ValueError('Invalid reference quarter phases')
            from scipy.signal import find_peaks
            ref = anchors['reference_rmd_mV']
            if ref.shape != (1000, 2) or ref.dtype != np.float64 or not np.isfinite(ref).all():
                raise ValueError('Invalid reference RMD recording')
            peaks = find_peaks(ref[:, 0], prominence=1., distance=20)[0]
            if len(peaks) < 2 or not np.array_equal(peak, peaks[:2]) or np.any(ix < 0) or np.any(ix >= 1000):
                raise ValueError('Reference is not the first complete detected cycle')
            if not np.array_equal(anchors['start_voltage_mV'][:, [ids.index(n) for n in ('RMDL', 'RMDR')]], ref[ix]):
                raise ValueError('Anchor voltages do not match recorded reference phases')
            ref_meta = summary['anchor_reference']
            if ref_meta['sample_dt_s'] != .001 or ref_meta['reference_step_ms'] != .03125 or ref_meta['warmup_seconds'] != 60. or not np.array_equal(ref_meta['indices'], ix) or not np.array_equal(ref_meta['peak_indices'], peak) or not np.allclose(ref_meta['time_s'], (ix + 1) * .001, rtol=0, atol=1e-12):
                raise ValueError('Reference construction metadata mismatch')
            if len(summary['levels']) > 4:raise ValueError('Unexpected grids')
            levels, comparisons, previous = [], [], None
            for i, item in enumerate(summary['levels']):
                path = root / label / ('level-' + str(i) + '.npz')
                if item['level'] != i or Path(item['artifact']).resolve() != path.resolve() or file_sha256(path) != item['sha256']:
                    raise ValueError('Grid ordering/artifact hash mismatch')
                hashes[str(path)] = item['sha256'];data = saved_arrays(path)
                if str(data['anchor_sha256']) != summary['anchor_sha256'] or float(data['neural_step_ms']) != STEPS[i] or float(data['sample_dt_s']) != .001:
                    raise ValueError('Shared-state/grid metadata mismatch')
                means, voltage = data['indicator_means'], data['rmd_voltage_mV']
                if means.shape != (4, 5, 302) or voltage.shape != (4, 5, 12000, 2):
                    raise ValueError('Invalid trial array shapes')
                for k in ('indicator_means', 'rmd_voltage_mV', *('short_' + k for k in FIELDS)):
                    a = data[k]
                    if a.dtype != np.float64 or not np.isfinite(a).all():raise ValueError('Nonfinite/non-float64: ' + k)
                for k in FIELDS:
                    if data['short_' + k].shape != ((4, 5, 302, 35) if k == 'gates' else (4, 5, 302)):
                        raise ValueError('Invalid short full-state shape')
                null = means[:, 4] - means[:, 3]
                null_short = {k: float(abs(data['short_' + k][:, 4] - data['short_' + k][:, 3]).max()) for k in FIELDS}
                controls = bool(abs(null).max() <= 1e-12 and all(v <= 1e-12 for v in null_short.values())
                                and abs(voltage[:, 4] - voltage[:, 3]).max() <= 1e-12)
                response = means[:, :3] - means[:, 3:4]
                phase_mean = response.mean(axis=0)
                rhythms = [{n: rhythm_metrics(voltage[phase, 3, :, j], .001) for j, n in enumerate(('RMDL', 'RMDR'))} for phase in range(4)]
                levels.append(dict(level=i, neural_step_ms=STEPS[i], duplicate_shams_passed=controls,
                    duplicate_indicator_max=float(abs(null).max()), duplicate_short_max=null_short,
                    sham_rhythm_by_phase=rhythms, response_max_abs=float(abs(response).max()),
                    sham_rhythm_persistent=all(m['passed'] for r in rhythms for m in r.values()),
                    phase_mean_response_max_abs=float(abs(phase_mean).max()), phase_response_std_max=float(response.std(axis=0).max()),
                    artifact_sha256=item['sha256']))
                if previous is not None:
                    old_response = previous['indicator_means'][:, :3] - previous['indicator_means'][:, 3:4]
                    per_phase = [compare_responses(old_response[j], response[j]) for j in range(4)]
                    phase_average = compare_responses(old_response.mean(axis=0), phase_mean)
                    short_errors = {}
                    for k in FIELDS:
                        d = data['short_' + k] - previous['short_' + k]
                        short_errors[k] = dict(max_abs=float(abs(d).max()), rms=float(np.sqrt(np.mean(d ** 2))))
                    comparisons.append(dict(levels=[i - 1, i], per_phase=per_phase, phase_mean=phase_average, short20ms_errors=short_errors))
                previous = data
            complete = len(levels) == 4
            controls = bool(complete and all(l['duplicate_shams_passed'] for l in levels))
            last = comparisons[-2:]
            models[label] = dict(checkpoint_sha256=expected_model, anchors_sha256=summary['anchor_sha256'], complete=complete,
                anchor_reference=ref_meta,
                levels=levels, comparisons=comparisons, controls_passed=controls,
                per_phase_response_checks_passed=bool(execution_complete and complete and controls and len(last) == 2 and all(p['passed'] for c in last for p in c['per_phase'])),
                phase_mean_response_checks_passed=bool(execution_complete and complete and controls and len(last) == 2 and all(c['phase_mean']['passed'] for c in last)))
        except Exception as exc:
            problems.append(label + ': ' + repr(exc));models[label] = dict(complete=False, error=repr(exc))
    for path, digest in hashes.items():
        if not Path(path).is_file() or file_sha256(path) != digest:
            problems.append('Consumed input changed: ' + path)
    result = dict(experiment='E108', audit_complete=True, execution_complete=execution_complete, integrity_passed=not problems,
        problems=problems, models=models, evidence_sha256=hashes, model_promoted=False, trainable_parameters=[],
        source_numerical_component_passed=manifest['source_numerical_component_passed'],
        candidate_selection='Both pinned models and all three targets chosen a priori; no optimizer or reselection',
        limits=['Shared-state/reference-phase diagnostic, not full atlas accuracy or behavior validation.',
                'Reference anchors are built at one finest step; not an independently converged attractor or control of all oscillator phases.',
                'Reference-phase anchors also differ in slow variables and other neuron states; response variability does not isolate RMDL phase causally.',
                'Saved initial states and source code document reuse; initial full-state construction is not independently re-simulated by audit.',
                'Mean indicator differences cannot be independently reconstructed from RMD-only traces; full-brain long traces intentionally omitted.',
                'Short-state discrepancies have no new physiological pass threshold; phase-mean diagnostic does not replace original failed criteria.',
                'No independent solver, sensory start/stop, body swimming or biological qualification.'])
    if problems:
        for m in models.values():
            m['per_phase_response_checks_passed'] = False;m['phase_mean_response_checks_passed'] = False
    write(root / 'audit.json', result)
    return result


def report(root):
    if not (root / 'audit.json').exists():
        print('No final audit yet. Log:', str(root) + '.log');return
    s = json.loads((root / 'audit.json').read_text())
    print('Run:', root)
    print({k: s.get(k) for k in ('audit_complete', 'execution_complete', 'integrity_passed', 'problems')})
    for label, model in s.get('models', {}).items():
        print(label, {k: model.get(k) for k in ('complete', 'checkpoint_sha256', 'anchors_sha256', 'anchor_reference', 'controls_passed', 'per_phase_response_checks_passed', 'phase_mean_response_checks_passed', 'error')})
        for level in model.get('levels', []):
            print('level', level['level'], {k: level[k] for k in ('response_max_abs', 'phase_mean_response_max_abs', 'phase_response_std_max', 'duplicate_shams_passed', 'sham_rhythm_persistent')})
        for pair in model.get('comparisons', []):
            print('grids', pair['levels'], 'per-phase:', pair['per_phase'], 'phase-mean:', pair['phase_mean'], 'short20ms:', pair['short20ms_errors'])
    if (root / 'budget_receipt.json').exists():print('Budget:', json.loads((root / 'budget_receipt.json').read_text()))
    print('Audit:', root / 'audit.json')
    print('No fitting or model promotion; original atlas failures unchanged.')


def main():
    started = time.monotonic();p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path);p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--minutes', type=float, default=120.);p.add_argument('--worker', choices=('selected', 'parent'))
    p.add_argument('--coordinate', action='store_true');p.add_argument('--report', action='store_true');args = p.parse_args()
    root = args.output_dir.resolve()
    if args.report:report(root);return 0
    if args.worker:worker(root, args.worker);return 0
    if args.coordinate:return coordinate(root)
    if args.source_run is None:p.error('--source-run required')
    if not math.isfinite(args.minutes) or not 5 <= args.minutes <= 1440:p.error('Budget must be5..1440minutes')
    source = args.source_run.resolve()
    if any(root == q or root in q.parents or q in root.parents for q in (source, REPO)):
        p.error('Separate external output required')
    root.mkdir(parents=True, exist_ok=False);code, timeout = 1, False
    try:
        capture_run_record(root, REPO, args, phase='common-state-diagnostic')
        prepare(source, root)
        from scripts.total_job_budget import run_group
        seconds = started + args.minutes * 60 - time.monotonic() - 60
        if seconds <= 0:raise TimeoutError('Preparation exhausted budget')
        code, timeout = run_group([sys.executable, '-u', '-m', 'scripts.run_common_state', '--coordinate', '--output-dir', str(root)], seconds)
    except Exception as exc:
        write(root / 'failure.json', dict(error=repr(exc)));print('Failure:', repr(exc), flush=True)
    finally:
        try:outcome = audit(root, execution_complete=code == 0 and not timeout)
        except Exception as exc:
            outcome = dict(audit_complete=False, integrity_passed=False, problems=[repr(exc)], models={}, model_promoted=False)
            write(root / 'audit.json', outcome)
        write(root / 'budget_receipt.json', dict(exit_code=code, timed_out=timeout, total_minutes=args.minutes,
            total_elapsed_seconds=time.monotonic() - started, audit_complete=outcome['audit_complete'],
            all_steps_finished=bool(code == 0 and not timeout and outcome['integrity_passed'] and len(outcome['models']) == 2 and all(m.get('complete') for m in outcome['models'].values())), model_promoted=False))
        report(root)
    return code


if __name__ == '__main__':raise SystemExit(main())
