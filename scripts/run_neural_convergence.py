"""E107 fixed-model neural convergence study with automatic saved-data audit."""
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
from scripts.selected_atlas_audit_checks import compare_responses, parameter_audit, saved_arrays, write

REPO = Path(__file__).resolve().parents[1]
SELECTED_SHA = '4d72ef33f194806155bfb893f1744848f79414a8e333042fbe6970b97835111f'
PARENT_SHA = '47a369a17f8e8c1387cd20f66c00f48db34b935591e043eb216731880c22482d'
ATLAS_STEPS = (40, 80, 160, 320, 640)
RHYTHM_STEPS_MS = (1., .5, .25, .125, .0625)


def prepare(source, out):
    """Pin just the consumed E106 inputs, not mutable live training outputs."""
    job_path = source / 'E104-network_control-job.json'
    job = json.loads(job_path.read_text())
    manifest_path = Path(job['manifest'])
    if file_sha256(manifest_path) != job['manifest_sha256']:
        raise ValueError('E106 manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    inputs = Path(job['inputs'])
    declared = {str(Path(e['path']).resolve()): e['sha256'] for e in manifest['evidence']}
    dest = out / 'inputs'
    dest.mkdir()
    evidence = {str(job_path): file_sha256(job_path), str(manifest_path): file_sha256(manifest_path)}
    for name in ('network_best_val.npz', 'parent_network.npz', 'responses.npz', 'summary.json'):
        path = inputs / name
        digest = file_sha256(path)
        if declared.get(str(path.resolve())) != digest:
            raise ValueError('Unverified E106 input: ' + str(path))
        shutil.copy2(path, dest / name)
        evidence[str(path)] = digest
        evidence[str(dest / name)] = digest
    for name, digest in (('network_best_val.npz', SELECTED_SHA), ('parent_network.npz', PARENT_SHA)):
        if file_sha256(dest / name) != digest:
            raise ValueError('Pinned checkpoint mismatch: ' + name)
    summary = json.loads((dest / 'summary.json').read_text())
    if summary['best_val_sha256'] != SELECTED_SHA or summary['parent_sha256'] != PARENT_SHA:
        raise ValueError('E106 model lineage mismatch')
    with np.load(dest / 'responses.npz', allow_pickle=False) as z:
        targets = z['targets'].tolist()[:13]
        stimulus = float(z['stimulus_nS'])
    if len(targets) != 13 or len(set(targets)) != 13 or stimulus != manifest['arguments']['stimulus_nS']:
        raise ValueError('Target/stimulus mismatch')
    if targets != manifest['statistics']['targets'][:13]:
        raise ValueError('Targets do not match E106 rebuilt atlas order')
    result = dict(experiment='E107', source_audit=str(source), targets=targets, stimulus_nS=stimulus,
                  evidence_sha256=evidence, models={
                      'selected': dict(path=str(dest / 'network_best_val.npz'), sha256=SELECTED_SHA),
                      'parent': dict(path=str(dest / 'parent_network.npz'), sha256=PARENT_SHA)},
                  atlas_substeps=list(ATLAS_STEPS), rhythm_step_ms=list(RHYTHM_STEPS_MS),
                  random_operations=False, trainable_parameters=[], model_promoted=False)
    write(out / 'manifest.json', result)
    return result


def worker(root, label):
    # Controller is CPU-only; this process sees exactly one assigned GPU.
    import jax
    import jax.numpy as jnp
    from r_elegans.brain.conductance_network import load_network
    from r_elegans.brain.selected_atlas_replay import atlas_forward, rhythm_forward
    if not jax.config.x64_enabled or jax.default_backend() != 'gpu':
        raise ValueError('Worker requires GPU and float64')
    manifest_path = root / 'manifest.json'
    manifest_sha = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    model = manifest['models'][label]
    if file_sha256(model['path']) != model['sha256']:
        raise ValueError('Model changed before worker')
    out = root / label
    out.mkdir(exist_ok=False)
    capture_run_record(out, REPO, dict(root=str(root), label=label), phase='convergence-worker')
    params, fixed, config = load_network(model['path'])
    params, fixed = jax.tree.map(lambda a: a.astype(jnp.float64) if jnp.issubdtype(a.dtype, jnp.floating) else a, (params, fixed))
    ids = saved_arrays(model['path'])['neuron_ids'].tolist()
    rmd_indices = [ids.index(n) for n in ('RMDL', 'RMDR')]
    gs = np.zeros((14, len(ids)), dtype=np.float64)
    gs[np.arange(13), [ids.index(n) for n in manifest['targets']]] = manifest['stimulus_nS']
    levels = []
    for level, (substeps, step_ms) in enumerate(zip(ATLAS_STEPS, RHYTHM_STEPS_MS)):
        started = time.monotonic()
        print(json.dumps(dict(model=label, level=level, atlas_step_ms=100 / substeps,
                              rhythm_step_ms=step_ms, status='starting')), flush=True)
        response = atlas_forward(params, fixed, config, jnp.asarray(gs), substeps)
        voltage = rhythm_forward(params, fixed, config, max_step_ms=step_ms, warmup_seconds=60.)
        if not np.isfinite(response).all() or not np.isfinite(voltage).all():
            raise ValueError('Nonfinite forward output at level ' + str(level))
        path = out / ('level-' + str(level) + '.npz')
        np.savez_compressed(path, atlas_response=response, rmd_voltage_mV=voltage[:, rmd_indices],
                            all_cell_late_std_mV=np.std(voltage[len(voltage)//2:], axis=0),
                            sample_dt_s=.002, atlas_step_ms=100 / substeps, rhythm_step_ms=step_ms)
        levels.append(dict(level=level, artifact=str(path), sha256=file_sha256(path),
                           atlas_step_ms=100 / substeps, rhythm_step_ms=step_ms,
                           elapsed_seconds=time.monotonic() - started))
        write(out / 'summary.json', dict(model=label, levels=levels, complete=len(levels) == len(ATLAS_STEPS),
                                        checkpoint_sha256=model['sha256'], manifest_sha256=manifest_sha))
        print(json.dumps(dict(model=label, level=level, status='finished',
                              elapsed_seconds=levels[-1]['elapsed_seconds'])), flush=True)
    if file_sha256(model['path']) != model['sha256'] or file_sha256(manifest_path) != manifest_sha:
        raise ValueError('Input changed during worker')


def coordinate(root):
    processes = []
    try:
        for gpu, label in enumerate(('selected', 'parent')):
            env = dict(os.environ, JAX_PLATFORMS='cuda', CUDA_VISIBLE_DEVICES=str(gpu),
                       XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.8')
            log = (root / (label + '.log')).open('w')
            p = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.run_neural_convergence',
                                  '--worker', label, '--output-dir', str(root)], env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append((label, p, log))
        while any(p.poll() is None for _, p, _ in processes):
            print(json.dumps(dict(running=[label for label, p, _ in processes if p.poll() is None])), flush=True)
            time.sleep(30)
        codes = {label: p.returncode for label, p, _ in processes}
        write(root / 'worker_exit_codes.json', codes)
        return int(any(codes.values()))
    finally:
        for _, p, log in processes:
            if p.poll() is None:
                p.terminate()
            log.close()


def audit(root, *, complete_execution):
    """Re-read compact saved arrays; never use the workers' aggregate pass flags."""
    from r_elegans.brain.rhythm_qualification import rhythm_metrics, compare_cycles
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = file_sha256(manifest_path)
    problems = []
    for path, digest in manifest['evidence_sha256'].items():
        if not Path(path).is_file() or file_sha256(path) != digest:
            problems.append('Input missing/changed: ' + path)
    models = {}
    for label, model in manifest['models'].items():
        rows, pairs = [], []
        try:
            if file_sha256(model['path']) != model['sha256']:
                raise ValueError('Checkpoint changed')
            # Exact unchanged reference to itself; no refit or widened bounds.
            params = parameter_audit(model['path'], model['path'], model['path'], 'E105', {'train_network': 'no'})
            params.pop('masks', None)
            saved = json.loads((root / label / 'summary.json').read_text())
            if saved['checkpoint_sha256'] != model['sha256'] or saved['manifest_sha256'] != manifest_sha:
                raise ValueError('Worker input identity mismatch')
            previous = None
            for expected, item in enumerate(saved['levels']):
                if expected >= len(ATLAS_STEPS) or item['level'] != expected:
                    raise ValueError('Missing, repeated or unexpected grid')
                path = Path(item['artifact'])
                if path.resolve() != (root / label / ('level-' + str(expected) + '.npz')).resolve():
                    raise ValueError('Artifact path mismatch')
                if file_sha256(path) != item['sha256']:
                    raise ValueError('Artifact hash mismatch: ' + str(path))
                data = saved_arrays(path)
                a, v = data['atlas_response'], data['rmd_voltage_mV']
                if a.shape != (14, 302) or v.shape != (20000, 2) or data['all_cell_late_std_mV'].shape != (302,):
                    raise ValueError('Wrong saved array shape')
                if not all(np.isfinite(data[k]).all() for k in data):
                    raise ValueError('Nonfinite artifact')
                if float(data['atlas_step_ms']) != 100 / ATLAS_STEPS[expected] or float(data['rhythm_step_ms']) != RHYTHM_STEPS_MS[expected] or float(data['sample_dt_s']) != .002:
                    raise ValueError('Grid metadata mismatch')
                rhythm = {n: rhythm_metrics(v[:, i], .002) for i, n in enumerate(('RMDL', 'RMDR'))}
                rows.append(dict(level=expected, atlas_step_ms=float(data['atlas_step_ms']),
                                 rhythm_step_ms=float(data['rhythm_step_ms']), rmd=rhythm,
                                 sham_max_abs_response=float(abs(a[-1]).max()),
                                 sham_passed=bool(abs(a[-1]).max() <= 1e-12),
                                 maximum_atlas_response=float(abs(a[:13]).max()),
                                 all_cell_late_std_mV=data['all_cell_late_std_mV'].tolist(),
                                 artifact_sha256=item['sha256']))
                if previous is not None:
                    atlas = compare_responses(previous['atlas_response'][:13], a[:13])
                    cycles = {n: compare_cycles(previous['rmd_voltage_mV'][:, i], v[:, i], .002) for i, n in enumerate(('RMDL', 'RMDR'))}
                    passed = bool(atlas['passed'] and all(c['passed'] for c in cycles.values()))
                    pairs.append(dict(levels=[expected - 1, expected], atlas=atlas, cycles=cycles, passed=passed))
                previous = data
                if file_sha256(path) != item['sha256']:
                    raise ValueError('Artifact changed during audit')
            complete = len(rows) == len(ATLAS_STEPS)
            controls = bool(rows and all(r['sham_passed'] for r in rows))
            persistence = bool(complete and all(m['passed'] for r in rows[-3:] for m in r['rmd'].values()))
            qualified = bool(complete_execution and complete and params['passed'] and controls and persistence
                             and len(pairs) >= 2 and all(p['passed'] for p in pairs[-2:]))
            models[label] = dict(grids_complete=complete, parameters=params, levels=rows, comparisons=pairs,
                                 controls_passed=controls, finest_three_rhythm_persistent=persistence,
                                 numerical_component_passed=qualified)
        except Exception as exc:
            problems.append(label + ': ' + repr(exc))
            models[label] = dict(numerical_component_passed=False, levels=rows, comparisons=pairs, error=repr(exc))
    if file_sha256(manifest_path) != manifest_sha:
        problems.append('Manifest changed during audit')
    for path, digest in manifest['evidence_sha256'].items():
        if not Path(path).is_file() or file_sha256(path) != digest:
            problems.append('Input missing/changed after audit: ' + path)
    result = dict(experiment='E107', audit_complete=True, execution_complete=complete_execution,
                  integrity_passed=not problems, problems=problems, models=models,
                  numerical_component_passed=bool(complete_execution and not problems and len(models) == 2
                                                and all(m['numerical_component_passed'] for m in models.values())),
                  candidate_selection='Both models fixed a priori; no optimizer, reselection or parameter updates',
                  model_promoted=False, sensory_start_stop_qualified=False, swimming_qualified=False,
                  unavailable_evidence=['No body/muscle dynamics, behavior, sensory control or biological fidelity tested.',
                                        'No independent ground truth or held-out atlas accuracy tested; fixed first13 numerical targets only.',
                                        'Refinement over finite tested grids does not prove an exact continuum solution.'],
                  scope='Saved-data post-run audit using existing numerical criteria; constant response identity alone is not an informative signal.')
    write(root / 'audit.json', result)
    return result


def report(root):
    path = root / 'audit.json'
    if not path.exists():
        print('No final audit yet. Log:', str(root) + '.log')
        return
    s = json.loads(path.read_text())
    print('Run:', root)
    print({k: s.get(k) for k in ('audit_complete', 'execution_complete', 'integrity_passed', 'problems', 'numerical_component_passed')})
    for label, m in s.get('models', {}).items():
        print(label, 'numerical component:', m['numerical_component_passed'], 'error:', m.get('error'))
        for p in m.get('comparisons', []):
            print(' grids', p['levels'], 'atlas:', p['atlas'], 'RMD:', p['cycles'])
        for level in m.get('levels', []):
            print(' grid', level['level'], 'rhythm:', {n: dict(passed=r['passed'], period_s=r.get('median_period_s'), reason=r.get('reason'), gates=r.get('gates')) for n, r in level['rmd'].items()}, 'sham:', level['sham_passed'])
    if (root / 'budget_receipt.json').exists():
        print('Budget:', json.loads((root / 'budget_receipt.json').read_text()))
    print('Report:', path)
    print('No model/behavior promoted; no fitting performed.')


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-audit', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--minutes', type=float, default=120.)
    parser.add_argument('--worker', choices=('selected', 'parent'))
    parser.add_argument('--coordinate', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if args.report:
        report(root)
        return 0
    if args.worker:
        worker(root, args.worker)
        return 0
    if args.coordinate:
        return coordinate(root)
    if not math.isfinite(args.minutes) or not 5 <= args.minutes <= 1440:
        parser.error('Total budget must be 5..1440 minutes')
    if args.source_audit is None:
        parser.error('--source-audit required')
    source = args.source_audit.resolve()
    if root == REPO or REPO in root.parents or root in REPO.parents or root == source or source in root.parents or root in source.parents:
        parser.error('Separate external output required')
    root.mkdir(parents=True, exist_ok=False)
    code, timeout, outcome = 1, False, {}
    try:
        capture_run_record(root, REPO, args, phase='neural-convergence-study')
        manifest = prepare(source, root)
        print('Pinned two unchanged models. First13 targets:', manifest['targets'], flush=True)
        from scripts.total_job_budget import run_group
        seconds = started + args.minutes * 60 - time.monotonic() - 60
        if seconds <= 0:
            raise TimeoutError('Preparation exhausted budget')
        code, timeout = run_group([sys.executable, '-u', '-m', 'scripts.run_neural_convergence',
                                   '--coordinate', '--output-dir', str(root)], seconds)
    except Exception as exc:
        write(root / 'failure.json', dict(error=repr(exc)))
        print('Study failure:', repr(exc), flush=True)
    finally:
        try:
            outcome = audit(root, complete_execution=code == 0 and not timeout)
        except Exception as exc:
            outcome = dict(audit_complete=False, numerical_component_passed=False, problems=[repr(exc)], model_promoted=False)
            write(root / 'audit.json', outcome)
        receipt = dict(exit_code=code, timed_out=timeout, total_minutes=args.minutes,
                       total_elapsed_seconds=time.monotonic() - started, audit_complete=outcome['audit_complete'],
                       all_steps_finished=bool(code == 0 and not timeout and outcome.get('integrity_passed')
                                               and all(m.get('grids_complete') for m in outcome.get('models', {}).values())),
                       numerical_component_passed=outcome['numerical_component_passed'], model_promoted=False)
        write(root / 'budget_receipt.json', receipt)
        report(root)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
