"""E128: full-target peptide validation; eight-hour plan, validation finishes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from scripts import run_cycle_ensemble as cycle
from scripts.experiment_records import capture_run_record, file_sha256

resume = cycle.resume
REPO = cycle.REPO
BASE = cycle.BASE
MODULE = 'scripts.run_full_cycle_validation'
SOURCE = BASE / 'cycle-ensemble-20261125T001721Z-1137019'


def jobs():
    from scripts.cv_validation_job import ancestor_pids
    found = cycle.jobs()
    excluded = ancestor_pids(os.getpid())
    for line in subprocess.check_output(['ps', '-ww', '-u', str(os.getuid()), '-o', 'pid=,etime=,args='], text=True).splitlines():
        row = line.split(None, 2)
        if len(row) != 3 or int(row[0]) in excluded:
            continue
        try:
            args = shlex.split(row[2])
        except ValueError:
            continue
        if any(args[i:i+2] == ['-m', MODULE] for i in range(len(args)-1)) and not any(a in args for a in ('--status', '--check', '--select-output', '--report')):
            found.append(dict(pid=int(row[0]), elapsed=row[1], command=row[2]))
    return found


def select_output(source):
    found = []
    for p in BASE.glob('full-cycle-validation-*/job.json'):
        if json.loads(p.read_text()).get('source_run') == str(source.resolve()):
            found.append(p.parent)
    return max(found) if found else BASE / ('full-cycle-validation-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + str(os.getpid()))


def tasks_for(m):
    # Preserve the original three-target padded batch verbatim. Remaining targets
    # occupy separate four-target batches, so no completed stimulus is replayed.
    tasks = []
    for block, b in enumerate(m['target_batches']):
        for group in range(7):
            # Alternate clocks across GPU slots, completing target blocks early.
            cases = m['cases'] if (block+group) % 2 == 0 else m['cases'][::-1]
            tasks += [dict(arm='peptide', level=c['level'], group=group, begin=b['begin'],
                           targets=b['targets'], model_sha256=c['model_sha256']) for c in cases]
    return tasks


def verify_inputs(m):
    for name, expected in m['input_sha256'].items():
        if file_sha256(name) != expected:
            raise ValueError('Input changed: ' + name)
    for name, expected in m['source_sha256'].items():
        if file_sha256(REPO / name) != expected:
            raise ValueError('Pinned source changed: ' + name)


def copy_verified(src, dest, inputs, expected=None):
    actual = file_sha256(src)
    if expected is not None and actual != expected:
        raise ValueError('Source artifact hash: ' + str(src))
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        temp = dest.with_name('.' + dest.name + '.copy')
        shutil.copy2(src, temp)
        os.replace(temp, dest)
        resume.sync_directory(dest.parent)
    if file_sha256(dest) != actual:
        raise ValueError('Copied artifact mismatch: ' + str(dest))
    inputs[str(src)] = actual
    inputs[str(dest)] = actual


def prepare(root, source):
    source = source.resolve()
    audit = json.loads((source / 'audit.json').read_text())
    old = json.loads((source / 'manifest.json').read_text())
    if not all(audit.get(k) for k in ('audit_complete', 'execution_complete', 'integrity_passed')):
        raise ValueError('Complete integrity-passed source required')
    stage = next(r for r in audit['stages'] if r['cycles'] == 32)
    if not all(c['sampling']['passed'] and c['controls_and_dynamics_passed'] for c in stage['cases'] if c['arm'] == 'peptide') or not next(n['passed'] for n in stage['numerical'] if n['arm'] == 'peptide'):
        raise ValueError('Peptide 32-cycle panel checks required')
    for p, h in old['source_sha256'].items():
        if file_sha256(REPO / p) != h:
            raise ValueError('Source numerical code changed: ' + p)
    ancestor = Path(old['source_run'])
    prior = json.loads((ancestor / 'manifest.json').read_text())
    prior_audit = json.loads((ancestor / 'audit.json').read_text())
    inputs = {str(p): file_sha256(p) for p in (source / 'audit.json', source / 'manifest.json', source / 'budget_receipt.json', ancestor / 'audit.json', ancestor / 'manifest.json')}
    for name, expected in old['input_sha256'].items():
        if file_sha256(name) != expected:
            raise ValueError('Recursive source input changed: ' + name)
        inputs[name] = expected
    targets = old['panel'] + [t for t in prior['full_targets'] if t not in old['panel']]
    if len(targets) != 135 or len(set(targets)) != 135 or len(old['panel']) > 4:
        raise ValueError('Expected 135 unique targets and reusable panel <=4')
    batches = [dict(begin=0, targets=old['panel'])]
    batches += [dict(begin=i, targets=targets[i:i+4]) for i in range(len(old['panel']), len(targets), 4)]
    copy_verified(source / 'peptide/model.npz', root / 'peptide/model.npz', inputs, audit['artifact_sha256'][str(source / 'peptide/model.npz')])
    cases = [c for c in old['cases'] if c['arm'] == 'peptide']
    if sorted(c['level'] for c in cases) != [0, 1]:
        raise ValueError('Two source clocks required')
    if any(c['model_sha256'] != file_sha256(root / 'peptide/model.npz') for c in cases):
        raise ValueError('Frozen model identity mismatch')
    reused = []
    for case in cases:
        src = cycle.folder_for(source, case)
        dest = cycle.folder_for(root, case)
        dest.mkdir(parents=True, exist_ok=True)
        for name in ('base-starts.npz', 'base-phases.json', 'base-metrics.json', 'extension.npz', 'extension-states.npz', 'extension-end.npz', 'extension.json'):
            path = src / name
            copy_verified(path, dest / name, inputs, audit['artifact_sha256'].get(str(path)))
        original = ancestor / 'peptide' / f"full-{case['level']}" / 'response.npz'
        if file_sha256(original) != prior_audit['artifact_sha256'][str(original)]:
            raise ValueError('Full source response changed')
        inputs[str(original)] = file_sha256(original)
        y = resume.old.prior.arrays(original)['response'][:, [prior['full_targets'].index(t) for t in targets]]
        path = dest / 'base-response.npz'
        if path.exists():
            if not np.array_equal(resume.old.prior.arrays(path)['response'], y):
                raise ValueError('Prepared base changed')
        else:
            resume.atomic_npz(path, response=y)
        inputs[str(path)] = file_sha256(path)
        for group in range(7):
            task = dict(arm='peptide', level=case['level'], group=group, begin=0, targets=old['panel'], model_sha256=case['model_sha256'])
            cycle.read_task(source, task)
            p = cycle.task_path(source, task)
            copy_verified(p, cycle.task_path(root, task), inputs, audit['artifact_sha256'][str(p)])
            reused.append(task)
    critical = dict(old['source_sha256'])
    critical['scripts/run_full_cycle_validation.py'] = file_sha256(REPO / 'scripts/run_full_cycle_validation.py')
    m = dict(experiment='E128', source_run=str(source), ancestor_run=str(ancestor), panel=targets, source_panel=old['panel'], cases=cases,
             target_batches=batches, input_sha256=inputs, source_sha256=critical, worker_cpus=old['worker_cpus'], reused_tasks=reused,
             numeric_tolerance=old['numeric_tolerance']['peptide'], policy='Fixed weights and source tolerances; 32 cycles x4 phases; peptide only; all135 targets. No fitting or model promotion.')
    verify_inputs(m)
    resume.atomic_json(root / 'manifest.json', m)


def task_iterator(root, m, slot):
    # The complete validation stage is admitted durably before workers start.
    # No clock cutoff applies inside validation, including after crash recovery.
    for task in tasks_for(m)[slot::2]:
        if cycle.task_path(root, task).exists():
            cycle.read_task(root, task)
            continue
        resume.atomic_json(root / f'admitted-{slot}.json', task)
        yield task


def worker(root, slot):
    import jax
    import jax.numpy as jnp
    from r_elegans.brain.peptide_checkpoint import load_model, load_state
    from r_elegans.brain.midpoint_replay import make_replay
    m = json.loads((root / 'manifest.json').read_text())
    job = json.loads((root / 'job.json').read_text())
    if not job.get('full_validation_admitted'):
        raise ValueError('Full validation stage not admitted')
    verify_inputs(m)
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, m['worker_cpus'][slot])
    if not jax.config.x64_enabled or jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
        raise ValueError('One visible float64 GPU required')
    capture_run_record(root / f'worker-{slot}', REPO, dict(slot=slot), phase='E128-full-target-validation')
    cache = {}
    for task in task_iterator(root, m, slot):
        level = task['level']
        folder = root / 'peptide' / f'level-{level}'
        if level not in cache:
            p, f, config, pp, exc, inh, _, arrays = load_model(root / 'peptide/model.npz')
            ids = arrays['neuron_ids'].tolist()
            states = load_state(folder / 'extension-states.npz')
            dt, step = resume.old.clock(level)
            kernel = make_replay(config, [ids.index('RMDL'), ids.index('RMDR')], dt, step, trace_mode='shams')
            cache[level] = p, f, pp, exc, inh, ids, states, kernel
        p, f, pp, exc, inh, ids, states, kernel = cache[level]
        start = jax.tree.map(lambda a: a[task['group']*16:(task['group']+1)*16], states)
        stimuli = np.zeros((6, 302))
        stimuli[np.arange(len(task['targets'])), [ids.index(t) for t in task['targets']]] = .03
        resume.atomic_json(root / f'progress-{slot}.json', dict(stage='computing_batch', **task))
        tick = time.monotonic()
        means, end, traces = jax.tree.map(np.asarray, kernel(p, f, pp, exc, inh, start, jnp.asarray(stimuli)))
        data = dict(means=means, voltage_min=traces[4].min(axis=-1), voltage_max=traces[5].max(axis=-1),
                    sham_voltage=traces[0] if task['begin'] == 0 else np.empty(0),
                    **{'final_'+k: v for k, v in dict(end.neural._asdict(), peptide=end.peptide).items()})
        path = cycle.task_path(root, task)
        resume.atomic_npz(path, provenance_json=np.array(json.dumps(task, sort_keys=True)), **data)
        cycle.read_task(root, task)
        resume.atomic_json(path.with_suffix('.json'), dict(sha256=file_sha256(path), seconds=time.monotonic()-tick))
        print(json.dumps(dict(event='batch_saved_verified', seconds=time.monotonic()-tick, **task)), flush=True)
    resume.atomic_json(root / f'progress-{slot}.json', dict(stage='assigned_batches_complete'))


def evaluate(root, m):
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    available = [t for t in tasks_for(m) if cycle.task_path(root, t).exists()]
    completed = {b['begin'] for b in m['target_batches'] if all(t in available for t in tasks_for(m) if t['begin'] == b['begin'])}
    indices = [i for b in m['target_batches'] if b['begin'] in completed for i in range(b['begin'], b['begin']+len(b['targets']))]
    cases = []
    responses = {}
    for case in m['cases']:
        folder = cycle.folder_for(root, case)
        base = resume.old.prior.arrays(folder / 'base-response.npz')['response'][:, indices]
        parts = [base]
        safe = True
        control_error = voltage_error = 0.
        failed_rhythms = []
        lo, hi = [], []
        for group in range(7):
            values, shams = [], []
            for task in [t for t in available if t['level'] == case['level'] and t['group'] == group]:
                a = cycle.read_task(root, task)
                # Safety includes partial target blocks as well as scored ones.
                lo.append(float(a['voltage_min'].min())); hi.append(float(a['voltage_max'].max()))
                if task['begin'] in completed:
                    values.append(a['means'][:, :len(task['targets'])]-a['means'][:, -1:])
                shams.append(a['means'][:, -2:])
                if task['begin'] == 0:
                    v = a['sham_voltage']
                    voltage_error = max(voltage_error, float(abs(v[:, 0]-v[:, 1]).max()))
                    failed_rhythms += [dict(group=group, phase=i, neuron=j) for i in range(16) for j in range(2)
                                       if not rhythm_metrics(v[i, 0, :, j], .001)['passed']]
            if shams:
                controls = np.stack(shams)
                control_error = max(control_error, float(abs(controls-controls[0, :, 0][None, :, None, :]).max()))
            if values:
                parts.append(np.concatenate(values, axis=1))
        safe = bool(lo and min(lo) >= -120 and max(hi) <= 80 and voltage_error <= 1e-10 and control_error <= 1e-12 and not failed_rhythms)
        y = np.concatenate(parts)
        metadata = json.loads((folder / 'extension.json').read_text())
        times = case['base_cycle_times'] + [case['extension_start_s']+t for t in metadata['cycle_mean_time_s']]
        stats = cycle.ensemble_metrics(y, times, y[:64], case['tolerance'])
        responses[case['level']] = y
        cases.append(dict(level=case['level'], sampling=stats, controls_and_dynamics_passed=safe, duplicate_mean_error=control_error,
                          duplicate_voltage_error=voltage_error, failed_rhythms=failed_rhythms,
                          voltage_min_mV=min(lo), voltage_max_mV=max(hi), final_states_finite=True))
    error = float(abs(responses[0].mean(0)-responses[1].mean(0)).max())
    full = len(available) == len(tasks_for(m))
    passed = all(c['sampling']['passed'] and c['controls_and_dynamics_passed'] for c in cases) and error <= m['numeric_tolerance']
    return dict(cases=cases, numerical=dict(maximum_difference=error, tolerance=m['numeric_tolerance'], passed=error <= m['numeric_tolerance']),
                completed_targets=[m['panel'][i] for i in indices], targets_total=len(m['panel']), saved_batches=len(available), batches_total=len(tasks_for(m)),
                full_target_complete=full, covered_target_checks_passed=bool(passed), full_target_qualified=bool(full and passed))


def audit(root):
    m = json.loads((root / 'manifest.json').read_text())
    verify_inputs(m)
    result = evaluate(root, m)
    # Independently check original/full response selection and extension phase provenance.
    prior = json.loads((Path(m['ancestor_run']) / 'manifest.json').read_text())
    for case in m['cases']:
        folder = cycle.folder_for(root, case)
        original = resume.old.prior.arrays(Path(m['ancestor_run']) / 'peptide' / f"full-{case['level']}" / 'response.npz')['response']
        expected = original[:, [prior['full_targets'].index(t) for t in m['panel']]]
        if not np.array_equal(expected, resume.old.prior.arrays(folder / 'base-response.npz')['response']):
            raise ValueError('Base response arithmetic')
        metadata = json.loads((folder / 'extension.json').read_text())
        ix, recomputed = cycle.extension_phases(metadata['reference_voltage_mV'], resume.old.clock(case['level'])[0])
        if any(metadata[k] != v for k, v in recomputed.items()):
            raise ValueError('Extension phase arithmetic')
        states = resume.old.prior.arrays(folder / 'extension-states.npz')
        if states['voltage_mV'].shape != (112, 302) or not all(np.isfinite(a).all() for a in states.values()):
            raise ValueError('Extension full states invalid')
        if not metadata['finite'] or metadata['voltage_min_mV'] < -120 or metadata['voltage_max_mV'] > 80:
            raise ValueError('Extension dynamics failed')
        ids = resume.old.prior.arrays(root / 'peptide/model.npz')['neuron_ids'].tolist()
        if not np.array_equal(states['voltage_mV'][:, ids.index('RMDL')], np.asarray(metadata['reference_voltage_mV'])[ix]):
            raise ValueError('Extension state correspondence')
    if file_sha256(root / 'peptide/model.npz') != m['cases'][0]['model_sha256']:
        raise ValueError('Frozen model changed')
    for path in root.glob('**/run_records/*/source_record.json'):
        for name, h in json.loads(path.read_text())['source_sha256'].items():
            if file_sha256(path.parent / 'source' / name) != h:
                raise ValueError('Archived source hash')
    result.update(audit_complete=True, integrity_passed=True, problems=[], execution_complete=result['full_target_complete'],
                  parameter_changes=0, all_parameters_frozen=True, model_sha256=file_sha256(root / 'peptide/model.npz'),
                  model_promoted=False, training_launched=False, legacy_gates_changed=False,
                  unavailable=['No fitting or optimizer/reboot recovery test; interrupted in-kernel computation restarts',
                               'Deterministic nested cycles are not confidence intervals; four original cycles retained',
                               'No body, swimming, sensory start/stop, biological or heldout-data qualification',
                               'Signal evidence: stimulus-minus-sham calcium indicator, RMD sham voltages, all-cell voltage bounds and final states; no full channel-current or body propagation attribution'],
                  artifact_sha256={str(p): file_sha256(p) for p in root.rglob('*.npz')})
    if not result['full_target_complete']:
        result['unavailable'].append('Full-target validation incomplete; continuation reuses every verified saved batch')
    resume.atomic_json(root / 'audit.json', result)
    return result


def run_workers(root):
    # Reuse the supervised two-GPU process lifecycle, without editing pinned code.
    previous = cycle.MODULE
    cycle.MODULE = MODULE
    try:
        return cycle.run_workers(root, 32)
    finally:
        cycle.MODULE = previous


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--source-run', type=Path, default=SOURCE)
    p.add_argument('--minutes', type=float, default=480)
    p.add_argument('--worker', type=int, choices=(0, 1))
    p.add_argument('--stage', type=int, default=32)  # shared process lifecycle argument
    for flag in ('status', 'check', 'select-output', 'report'):
        p.add_argument('--'+flag, action='store_true')
    a = p.parse_args()
    if a.select_output:
        print(select_output(a.source_run)); return 0
    if a.status or a.check:
        found = jobs(); print('READ-ONLY; no job started. Active jobs:', json.dumps(found))
        return 3 if a.check and found else 0
    if a.output_dir is None:
        p.error('--output-dir required')
    root = a.output_dir.resolve()
    if a.report:
        print('READ-ONLY REPORT:', root)
        for name in ('audit.json', 'budget_receipt.json'):
            row = json.loads((root / name).read_text()) if (root / name).exists() else {}
            print(name, json.dumps({k: v for k, v in row.items() if k not in ('artifact_sha256', 'completed_targets')}))
        return 0
    if a.worker is not None:
        return worker(root, a.worker)
    if not np.isfinite(a.minutes) or a.minutes <= 0:
        p.error('Positive finite allowance required')
    if jobs():
        p.error('Active GPU work detected; nothing started')
    import fcntl
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / '.cv-peptide-validation.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error('Shared GPU lock held; nothing started')
        if jobs():
            p.error('Active GPU work detected after lock')
        if root == REPO or REPO in root.parents or root in REPO.parents:
            p.error('External output required')
        root.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        path = root / 'job.json'
        job = json.loads(path.read_text()) if path.exists() else dict(source_run=str(a.source_run.resolve()), planning_minutes=a.minutes, full_validation_admitted=False)
        if job['source_run'] != str(a.source_run.resolve()):
            p.error('Source mismatch')
        receipt = root / 'budget_receipt.json'
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if previous.get('full_target_complete') and previous.get('audit_complete'):
                print('Already finished; nothing started. Use --report.'); return 0
            # Preserve every previous receipt/audit before continuation.
            for name in ('budget_receipt.json', 'audit.json'):
                if (root / name).exists():
                    shutil.copy2(root / name, root / (name + '.previous-' + str(time.time_ns())))
        resume.atomic_json(path, job)
        resume.atomic_json(receipt, dict(status='running', all_steps_finished=False, audit_complete=False, training_launched=False))
        code = 1
        try:
            if not (root / 'manifest.json').exists():
                if shutil.disk_usage(root).free < 20*1024**3:
                    raise ValueError('Requires 20 GiB free for compact checkpoints')
                print('Verifying source; reusing 32-cycle states, three-target batches and full four-cycle responses', flush=True)
                prepare(root, a.source_run)
            verify_inputs(json.loads((root / 'manifest.json').read_text()))
            job['full_validation_admitted'] = True
            resume.atomic_json(path, job)
            capture_run_record(root, REPO, vars(a) | dict(output_dir=str(root), source_run=str(a.source_run)), phase='E128-eight-hour-validation')
            code = run_workers(root)
        except Exception as exc:
            traceback.print_exc(); resume.atomic_json(root / 'failure.json', dict(error=repr(exc)))
        try:
            result = audit(root)
        except Exception as exc:
            traceback.print_exc(); result = dict(audit_complete=False, integrity_passed=False, problems=[repr(exc)], full_target_complete=False)
            resume.atomic_json(root / 'audit.json', result)
        outcome = dict(status='finished', exit_code=code, timed_out=False, elapsed_seconds=time.monotonic()-started,
                       planning_minutes=job['planning_minutes'], validation_allowed_to_finish=True, validation_wallclock_cutoff=False,
                       all_steps_finished=bool(code == 0 and result['audit_complete'] and result['full_target_complete']),
                       audit_complete=result['audit_complete'], full_target_complete=result['full_target_complete'],
                       full_target_qualified=bool(code == 0 and result.get('full_target_qualified')),
                       saved_batches=result.get('saved_batches'), batches_total=result.get('batches_total'),
                       training_launched=False, model_promoted=False)
        resume.atomic_json(receipt, outcome); print(json.dumps(outcome))
        return 0 if code == 0 and result['audit_complete'] and result['integrity_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
