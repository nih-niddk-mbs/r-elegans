"""E129 three-hour train-only fivefold peptide pilot; validation/audit finish.

Four-target development pilot, not full-target fitting or behavioral qualification.
Two persistent GPU workers; compact resumable batches; parameters/logs retained.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from scripts import cv_cycle_pilot_core as core
from scripts import run_full_cycle_validation as source_runner
from scripts.experiment_records import capture_run_record, file_sha256

io = core.io
BASE = source_runner.BASE
REPO = source_runner.REPO
MODULE = 'scripts.run_cv_cycle_pilot'
SOURCE = BASE/'full-cycle-validation-20261125T051038Z-1150797'
FOLDS = BASE/'cv-peptide-fit-20261124T042745Z-989175'


def jobs():
    from scripts.cv_validation_job import ancestor_pids
    found = source_runner.jobs(); excluded = ancestor_pids(os.getpid())
    import shlex
    for line in subprocess.check_output(['ps', '-ww', '-u', str(os.getuid()), '-o', 'pid=,etime=,args='], text=True).splitlines():
        row = line.split(None, 2)
        if len(row) != 3 or int(row[0]) in excluded:
            continue
        try:
            args = shlex.split(row[2])
        except ValueError:
            continue
        if any(args[i:i+2] == ['-m', MODULE] for i in range(len(args)-1)) and not any(a in args for a in ('--status', '--report', '--check', '--select-output')):
            found.append(dict(pid=int(row[0]), elapsed=row[1], command=row[2]))
    return found


def output_for(source):
    paths = [p.parent for p in BASE.glob('cv-cycle-pilot-*/job.json') if json.loads(p.read_text())['source_run'] == str(source.resolve())]
    return max(paths) if paths else BASE/('cv-cycle-pilot-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(os.getpid()))


def prepare(root, source, folds_source):
    audit = json.loads((source/'audit.json').read_text())
    if not all(audit.get(k) for k in ('audit_complete', 'integrity_passed', 'full_target_complete', 'full_target_qualified')):
        raise ValueError('Complete qualified E128 audit required')
    model = source/'peptide/model.npz'
    if file_sha256(model) != core.PARENT_SHA or audit['model_sha256'] != core.PARENT_SHA:
        raise ValueError('Exact clean peptide parent required')
    prior = json.loads((folds_source/'manifest.json').read_text())
    folds_path = folds_source/'folds.json'
    if file_sha256(folds_path) != prior['input_sha256'][str(folds_path)]:
        raise ValueError('Animal fold identity changed')
    folds = json.loads(folds_path.read_text())
    if len(folds) != 5:
        raise ValueError('Five independent animal folds required')
    held = []
    for fold in folds:
        train = fold['subjects']['train_subjects']; test = fold['subjects']['heldout_subjects']
        if set(train) & set(test) or len(set(train+test)) != 87:
            raise ValueError('Animal leakage or unexpected development cohort')
        held += test
    if len(held) != 87 or len(set(held)) != 87:
        raise ValueError('Heldout animal coverage must be exactly once')
    previous = json.loads((source/'manifest.json').read_text())
    panel, rng_state = core.panel_for(folds, previous['panel'], return_state=True)
    io.atomic_json(root/'rng_state.json', rng_state)
    inputs = {}
    for src, dst in ((model, root/'parent-model.npz'), (folds_path, root/'folds.json')):
        source_runner.copy_verified(src, dst, inputs)
    for path in (source/'audit.json', source/'manifest.json', folds_source/'manifest.json'):
        inputs[str(path)] = file_sha256(path)
    from scripts.run_cv_peptide_pilot import physical_cpus
    cpus = physical_cpus()
    if len(cpus) < 8:
        raise ValueError('Eight physical CPU cores required')
    critical = dict(previous['source_sha256'])
    for path, expected in critical.items():
        if file_sha256(REPO/path) != expected:
            raise ValueError('Validated numerical source changed: '+path)
    for name in ('scripts/cv_cycle_pilot_core.py', 'scripts/run_cv_cycle_pilot.py', 'scripts/cv_fit_core.py', 'scripts/stable_atlas_policy.py'):
        critical[name] = file_sha256(REPO/name)
    m = dict(experiment='E129', source_run=str(source), fold_source=str(folds_source), panel=panel, source_targets=previous['panel'], seed=1290,
             parent_sha256=core.PARENT_SHA, source_sha256=critical, input_sha256=inputs, cpus=[cpus[:4], cpus[4:8]],
             sampling_tolerances={str(c['level']): c['sampling']['tolerance'] for c in audit['cases']},
             numerical_tolerance=audit['numerical']['tolerance'], peptide_only=True, folds=5,
             scope='Four-target development fitting/direction/throughput pilot, no full135-target fitting or behavior qualification',
             labels='All87 historically inspected development animals; five independent fits, heldout labels never select updates')
    io.atomic_json(root/'manifest.json', m)
    # Reuse only verified source panel means. If previously pruned, explicitly
    # recompute this four-target baseline (not the whole135-target validation).
    for case in previous['cases']:
        level = case['level']; folder = source_runner.cycle.folder_for(source, case)
        out = root/'evaluations'/f'parent-{level}'; out.mkdir(parents=True, exist_ok=True)
        try:
            ix = [previous['panel'].index(t) for t in panel]
            path = folder/'base-response.npz'
            if file_sha256(path) != audit['artifact_sha256'][str(path)]:
                raise ValueError('Source baseline hash mismatch')
            parts = [core.arrays(path)['response'][:, ix]]; reused_hashes = {str(path): file_sha256(path)}
            tasks = source_runner.tasks_for(previous)
            for group in range(7):
                values = []
                for target in panel:
                    task = next(t for t in tasks if t['level'] == level and t['group'] == group and target in t['targets'])
                    path = source_runner.cycle.task_path(source, task)
                    if file_sha256(path) != audit['artifact_sha256'][str(path)]:
                        raise ValueError('Source baseline batch hash mismatch')
                    with np.load(path, allow_pickle=False) as z:
                        means = z['means']
                    values.append(means[:, task['targets'].index(target)]-means[:, -1])
                    reused_hashes[str(path)] = file_sha256(path)
                parts.append(np.stack(values, axis=1))
            response = np.concatenate(parts)
            meta = json.loads((folder/'extension.json').read_text())
            times = case['base_cycle_times']+[case['extension_start_s']+t for t in meta['cycle_mean_time_s']]
            sampling = core.check_response(response, times, m['sampling_tolerances'][str(level)])
            if not sampling['passed']:
                raise ValueError('Reused subset no longer passes source sampling')
            io.atomic_npz(out/'response.npz', response=response)
            task = dict(id=f'parent-{level}', model='parent-model.npz', model_sha256=core.PARENT_SHA, level=level)
            io.atomic_json(out/'result.json', dict(id=task['id'], task=task, status='completed', safe=True, sampling=sampling,
                reused_source=True, source_sha256=reused_hashes, response_sha256=file_sha256(out/'response.npz'), elapsed_seconds=0.,
                batch_checks=[], controls_from_audited_source=True, cycle_times=times))
        except FileNotFoundError:
            io.atomic_json(out/'reuse-unavailable.json', dict(reason='Source arrays previously pruned; four-target parent baseline will be recomputed', level=level))


def verify(root):
    m = json.loads((root/'manifest.json').read_text())
    for name, h in m['input_sha256'].items():
        if file_sha256(name) != h:
            raise ValueError('Input changed: '+name)
    for name, h in m['source_sha256'].items():
        if file_sha256(REPO/name) != h:
            raise ValueError('Pinned numerical source changed: '+name)
    return m


def result(root, name):
    folder = root/'evaluations'/name
    row = json.loads((folder/'result.json').read_text())
    if row['status'] != 'completed':
        return row, None
    if file_sha256(folder/'response.npz') != row['response_sha256']:
        raise ValueError('Completed response changed: '+name)
    return row, core.arrays(folder/'response.npz')['response']


def worker(root, slot):
    import jax
    m = verify(root)
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, m['cpus'][slot])
    if not jax.config.x64_enabled or jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
        raise ValueError('One float64 GPU required')
    capture_run_record(root/f'worker-{slot}', REPO, dict(slot=slot), phase='E129-persistent-worker')
    engine = core.Engine(root); engine.manifest['worker_slot'] = slot
    while not (root/'stop-workers.json').exists():
        queue = json.loads((root/'tasks.json').read_text())
        tasks = [t for i, t in enumerate(queue) if i % 2 == slot and not (root/'evaluations'/t['id']/'result.json').exists()]
        if not tasks:
            time.sleep(1); continue
        task = tasks[0]
        try:
            engine.evaluate(task)
        except Exception as exc:
            traceback.print_exc()
            io.atomic_json(root/'evaluations'/task['id']/'result.json', dict(status='failed', task=task, error=repr(exc)))


def submit(root, tasks, processes):
    path = root/'tasks.json'; queue = json.loads(path.read_text())
    for task in tasks:
        existing = next((t for t in queue if t['id'] == task['id']), None)
        if existing is not None and existing != task:
            raise ValueError('Task identity changed')
        if existing is None:
            queue.append(task)
    io.atomic_json(path, queue)
    while not all((root/'evaluations'/t['id']/'result.json').exists() for t in tasks):
        if any(p.poll() is not None for p in processes):
            raise RuntimeError('GPU worker stopped; saved batches retained for continuation')
        print(json.dumps(dict(stage='waiting', tasks=[t['id'] for t in tasks], saved_results=sum((root/'evaluations'/t['id']/'result.json').exists() for t in tasks))), flush=True)
        time.sleep(20)
    return [result(root, t['id']) for t in tasks]


def model_task(root, origin, coeff, name, level):
    path = root/'models'/(name+'.npz'); path.parent.mkdir(exist_ok=True)
    expected, _ = core.decode(origin, coeff)
    if path.exists():
        if not core.parameter_audit(origin, core.arrays(path), coeff)['passed']:
            raise ValueError('Model changed')
    else:
        io.atomic_npz(path, **expected)
    return dict(id=name+f'-level-{level}', model=str(path.relative_to(root)), model_sha256=file_sha256(path), level=level)


def coordinate(root):
    m = verify(root); folds = json.loads((root/'folds.json').read_text()); origin = core.arrays(root/'parent-model.npz')
    ids = origin['neuron_ids'].tolist()
    for fold in folds:
        fold['_origin'] = origin
    queue = root/'tasks.json'
    if not queue.exists():
        io.atomic_json(queue, [])
    (root/'stop-workers.json').unlink(missing_ok=True)
    processes = []; logs = []
    try:
        for slot in range(2):
            env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(slot), JAX_PLATFORMS='cuda', JAX_ENABLE_X64='true')
            log = (root/f'worker-{slot}.log').open('a'); logs.append(log)
            processes.append(subprocess.Popen([sys.executable, '-u', '-m', MODULE, '--output-dir', str(root), '--worker', str(slot)], env=env, stdout=log, stderr=subprocess.STDOUT))
        parent_tasks = [dict(id=f'parent-{level}', model='parent-model.npz', model_sha256=core.PARENT_SHA, level=level) for level in (0, 1)]
        parents = submit(root, parent_tasks, processes)
        if not all(r['status'] == 'completed' and r['safe'] for r, _ in parents):
            raise ValueError('Parent baseline failed')
        parent_numeric = core.numerical(parents[0][1], parents[1][1], m['numerical_tolerance'])
        if not parent_numeric['passed']:
            raise ValueError('Parent baseline numerical failure')
        state_path = root/'optimizer.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
        else:
            state = dict(folds=[dict(coefficients=[0.]*5, selected='parent-0', probes=0, score=core.score(parents[0][1], m['panel'], ids, fold, [0.]*5)) for fold in folds],
                         history=[], next_fold=0, deadline_unix=time.time()+json.loads((root/'job.json').read_text())['optimization_minutes']*60)
            if not all(np.isfinite(f['score']) for f in state['folds']):
                raise ValueError('Finite baseline training scores required')
            io.atomic_json(state_path, state)
        # A pending pair survives interruption and is finished before a planned
        # optimization stop; new pairs are admitted using measured wall times.
        while not state.get('stop_reason'):
            pending = state.get('pending')
            if pending is None:
                elapsed = [r['elapsed_seconds'] for p in (root/'evaluations').glob('*/result.json')
                           if (r := json.loads(p.read_text())).get('status') == 'completed' and r.get('elapsed_seconds', 0) > 0 and r['task']['level'] == 0]
                estimate = max(elapsed[-6:], default=1200.)
                if time.time()+1.15*estimate >= state['deadline_unix']:
                    state['stop_reason'] = 'optimization_allowance'; io.atomic_json(state_path, state); break
                index = state['next_fold']; current = state['folds'][index]
                group = (index+current['probes']) % 5
                pair = []
                for sign in (-1, 1):
                    coeff = np.asarray(current['coefficients']).copy(); coeff[group] = np.clip(coeff[group]+sign*.01, -.1, .1)
                    name = f"fold-{index}-probe-{current['probes']:03d}-{sign:+d}"
                    pair.append(dict(task=model_task(root, origin, coeff, name, 0), coefficients=coeff.tolist()))
                pending = dict(fold=index, group=core.GROUPS[group], proposals=pair)
                state['pending'] = pending; io.atomic_json(state_path, state)
            results = submit(root, [p['task'] for p in pending['proposals']], processes)
            index = pending['fold']; current = state['folds'][index]; proposals = []
            for p, (row, y) in zip(pending['proposals'], results):
                proposals.append(dict(name=p['task']['id'], coefficients=p['coefficients'], safe=row.get('safe', False),
                    score=core.score(y, m['panel'], ids, folds[index], p['coefficients']) if y is not None else None))
            point = core.choose(dict(name=current['selected'], coefficients=current['coefficients'], score=current['score'], safe=True), proposals)
            history = dict(fold=index, group=pending['group'], before=dict(current), proposals=proposals, selected=point['name'])
            current.update(coefficients=point['coefficients'], score=point['score'], selected=point['name'], probes=current['probes']+1)
            state['history'].append(history); state.pop('pending'); state['next_fold'] = (index+1) % 5
            io.atomic_json(state_path, state)
            io.atomic_npz(root/'optimizer.npz', coefficients=np.asarray([f['coefficients'] for f in state['folds']]), next_fold=np.array(state['next_fold']))
        io.atomic_npz(root/'optimizer.npz', coefficients=np.asarray([f['coefficients'] for f in state['folds']]), next_fold=np.array(state['next_fold']))
        # Freeze all choices before the first heldout loss call. Validation has
        # no inherited deadline and receives only fixed endpoints.
        endpoints = []
        for index, current in enumerate(state['folds']):
            coeff = current['coefficients']
            if not np.any(coeff):
                task = parent_tasks[1]
            else:
                coarse = json.loads((root/'evaluations'/current['selected']/'result.json').read_text())['task']
                task = dict(coarse, id=current['selected']+'-fine', level=1)
            endpoints.append(task)
        io.atomic_json(root/'selection-frozen.json', dict(endpoints=endpoints, folds=state['folds'], heldout_used=False))
        endpoint_results = submit(root, endpoints, processes)
        summaries = []
        for index, ((row, fine), current) in enumerate(zip(endpoint_results, state['folds'])):
            coarse_row, coarse = result(root, current['selected'])
            coeff = current['coefficients']; valid = row['status'] == 'completed' and row.get('safe', False)
            check = core.numerical(coarse, fine, m['numerical_tolerance']) if valid else dict(passed=False)
            parent_train = core.score(parents[1][1], m['panel'], ids, folds[index], [0.]*5)
            candidate_train = core.score(fine, m['panel'], ids, folds[index], coeff) if valid else None
            accepted = bool(np.any(coeff) and valid and check['passed'] and candidate_train < parent_train-1e-8)
            selected = fine if accepted else parents[1][1]
            chosen_coeff = coeff if accepted else [0.]*5
            out = root/f'fold-{index}'; out.mkdir(exist_ok=True)
            selected_values, _ = core.decode(origin, chosen_coeff)
            starts = None
            if accepted:
                starts = core.arrays(root/'evaluations'/current['selected']/'starts.npz')
                starts = {k: starts[k] for k in ('voltage_mV', 'gates', 'calcium_uM', 'indicator', 'peptide')}
                for key, values in starts.items():
                    selected_values['start_'+key] = values[:4]
            from r_elegans.brain.peptide_checkpoint import validate
            validate(selected_values)
            pa = core.parameter_audit(origin, selected_values, chosen_coeff, starts)
            if accepted:
                io.atomic_npz(out/'selected-model.npz', **selected_values)
            else:
                shutil.copy2(root/'parent-model.npz', out/'selected-model.npz')
            io.atomic_npz(out/'selected-coordinates.npz', coefficients=np.asarray(chosen_coeff))
            try:
                heldout = dict(loss=core.loss(selected, m['panel'], ids, folds[index], 'heldout'), parent_loss=core.loss(parents[1][1], m['panel'], ids, folds[index], 'heldout'), unavailable=None)
            except ValueError as exc:
                heldout = dict(loss=None, parent_loss=None, unavailable=str(exc))
            summary = dict(fold=index, probes=current['probes'], selected_candidate=accepted, candidate_coefficients=coeff, selected_coefficients=chosen_coeff,
                parent_train_score=parent_train, candidate_train_score=candidate_train, selected_train_score=candidate_train if accepted else parent_train,
                numerical=check, candidate_safe=bool(valid), parameters=pa, heldout=heldout,
                selected_model_sha256=file_sha256(out/'selected-model.npz'),
                scope='Four-target pilot endpoint; no full135-target or behavior qualification')
            io.atomic_json(out/'summary.json', summary); summaries.append(summary)
        io.atomic_json(root/'summary.json', dict(experiment='E129', folds=summaries, optimization_probes=len(state['history']),
            parent_numerical=parent_numeric, panel=m['panel'], selection_uses_heldout=False, validation_complete=True,
            pilot_descent_demonstrated=any(s['selected_candidate'] for s in summaries), full_target_qualified=False, behavior_qualified=False))
    finally:
        io.atomic_json(root/'stop-workers.json', dict(stop=True))
        for p in processes:
            # Let an already admitted evaluation finish its durable batches even
            # if the coordinator encountered an error; no planned kernel kill.
            p.wait()
        for log in logs:
            log.close()


def audit(root):
    m = verify(root); summary = json.loads((root/'summary.json').read_text())
    state = json.loads((root/'optimizer.json').read_text()); origin = core.arrays(root/'parent-model.npz')
    ids = origin['neuron_ids'].tolist(); folds = json.loads((root/'folds.json').read_text())
    problems = []; unavailable = []
    if summary['selection_uses_heldout'] or json.loads((root/'selection-frozen.json').read_text())['heldout_used']:
        problems.append('Heldout selection flag')
    if 'source_targets' in m:
        expected_panel, rng_state = core.panel_for(folds, m['source_targets'], return_state=True)
        if expected_panel != m['panel'] or rng_state != json.loads((root/'rng_state.json').read_text()):
            problems.append('Training-only panel/RNG arithmetic')
    optimizer = core.arrays(root/'optimizer.npz')
    if not np.array_equal(optimizer['coefficients'], np.asarray([f['coefficients'] for f in state['folds']])) or int(optimizer['next_fold']) != state['next_fold']:
        problems.append('Optimizer JSON/NPZ mismatch')
    for fold in folds:
        fold['_origin'] = origin
    evaluated = {}
    for path in sorted((root/'evaluations').glob('*/result.json')):
        row, y = result(root, path.parent.name)
        if row['status'] != 'completed':
            unavailable.append(dict(task=path.parent.name, error=row.get('error'))); continue
        task = row['task']
        if file_sha256(root/task['model']) != task['model_sha256']:
            problems.append('Evaluation model hash '+task['id'])
        if row.get('reused_source'):
            for name, h in row['source_sha256'].items():
                if file_sha256(name) != h:
                    problems.append('Reused parent response '+name)
        else:
            folder = path.parent; meta = json.loads((folder/'phases.json').read_text())
            cp = json.loads((folder/'states-verified.json').read_text())
            if cp['model_sha256'] != task['model_sha256'] or any(file_sha256(folder/n) != h for n, h in cp['sha256'].items()):
                problems.append('Starting-state checkpoint '+task['id'])
            from r_elegans.brain.midpoint_replay import cycle_phases
            dt = io.old.clock(task['level'])[0]
            base_ix, first = cycle_phases(meta['base']['reference_voltage_mV'], dt)
            ext_ix, second = source_runner.cycle.extension_phases(meta['extension']['reference_voltage_mV'], dt)
            if any(meta['base'][k] != v for k, v in first.items()) or any(meta['extension'][k] != v for k, v in second.items()):
                problems.append('Phase arithmetic '+task['id'])
            starts = core.arrays(folder/'starts.npz')
            reference = np.r_[np.asarray(meta['base']['reference_voltage_mV'])[base_ix], np.asarray(meta['extension']['reference_voltage_mV'])[ext_ix]]
            if starts['voltage_mV'].shape != (128, 302) or not np.array_equal(starts['voltage_mV'][:, ids.index('RMDL')], reference):
                problems.append('Full state/phase correspondence '+task['id'])
            pieces = []
            for group in range(8):
                batch = folder/f'batch-{group:02d}.npz'; data = core.arrays(batch)
                details = json.loads(str(data['record_json'])); checks = details['checks']
                if file_sha256(batch) != row['batch_sha256'][batch.name]:
                    problems.append('Batch checksum '+str(batch))
                if details['provenance'] != dict(task=task, group=group, starts_sha256=file_sha256(folder/'starts.npz')):
                    problems.append('Batch provenance '+str(batch))
                passed = (checks['finite'] and checks['voltage_min_mV'] >= -120 and checks['voltage_max_mV'] <= 80
                          and checks['duplicate_mean_error'] <= 1e-12 and checks['duplicate_voltage_error_mV'] <= 1e-10 and not checks['failed_rhythms'])
                if checks['passed'] != passed or checks != row['batch_checks'][group]:
                    problems.append('Control/dynamics arithmetic '+str(batch))
                pieces.append(data['means'][:, :4]-data['means'][:, -1:])
            if not np.array_equal(np.concatenate(pieces), y):
                problems.append('Response arithmetic '+task['id'])
            if core.check_response(y, meta['cycle_times'], m['sampling_tolerances'][str(task['level'])]) != row['sampling']:
                problems.append('Sampling arithmetic '+task['id'])
        computed_sampling = core.check_response(y, row['cycle_times'], m['sampling_tolerances'][str(task['level'])])
        safe = bool(computed_sampling['passed'] and (row.get('controls_from_audited_source') or all(c['passed'] for c in row['batch_checks'])))
        if row['sampling'] != computed_sampling or row['safe'] != safe:
            problems.append('Evaluation safety/sampling arithmetic '+task['id'])
        evaluated[task['id']] = row, y
    # Recompute every training-only proposal and greedy choice, never using heldout.
    parent_coarse = result(root, 'parent-0')[1]
    replayed = [dict(coefficients=[0.]*5, selected='parent-0', probes=0, score=core.score(parent_coarse, m['panel'], ids, fold, [0.]*5)) for fold in folds]
    for entry in state['history']:
        index = entry['fold']; proposals = []
        if entry['before'] != replayed[index]:
            problems.append('Cross-fold optimizer independence/history')
        for p in entry['proposals']:
            row, y = result(root, p['name'])
            expected_score = core.score(y, m['panel'], ids, folds[index], p['coefficients']) if y is not None else None
            if p['score'] != expected_score or p['safe'] != row.get('safe', False):
                problems.append('Training proposal arithmetic '+p['name'])
            if not core.parameter_audit(origin, core.arrays(root/row['task']['model']), p['coefficients'])['passed']:
                problems.append('Proposal parameter bounds/frozen fields '+p['name'])
            proposals.append(p)
        before = entry['before']
        chosen = core.choose(dict(name=before['selected'], coefficients=before['coefficients'], score=before['score'], safe=True), proposals)
        if chosen['name'] != entry['selected']:
            problems.append('Train-only selection arithmetic')
        replayed[index].update(coefficients=chosen['coefficients'], selected=chosen['name'], score=chosen['score'], probes=replayed[index]['probes']+1)
    if replayed != state['folds']:
        problems.append('Final optimizer state/history mismatch')
    parent0, parent1 = result(root, 'parent-0')[1], result(root, 'parent-1')[1]
    if core.numerical(parent0, parent1, m['numerical_tolerance']) != summary['parent_numerical']:
        problems.append('Parent numerical arithmetic')
    for index, s in enumerate(summary['folds']):
        selected_path = root/f'fold-{index}'/'selected-model.npz'
        if file_sha256(selected_path) != s['selected_model_sha256']:
            problems.append('Selected model hash')
        current = state['folds'][index]; coarse = result(root, current['selected'])[1]
        endpoint = json.loads((root/'selection-frozen.json').read_text())['endpoints'][index]
        row, fine = result(root, endpoint['id'])
        safe = row['status'] == 'completed' and row.get('safe', False)
        numerical = core.numerical(coarse, fine, m['numerical_tolerance']) if safe else dict(passed=False)
        candidate_score = core.score(fine, m['panel'], ids, folds[index], current['coefficients']) if safe else None
        parent_score = core.score(parent1, m['panel'], ids, folds[index], [0.]*5)
        accepted = bool(np.any(current['coefficients']) and safe and numerical['passed'] and candidate_score < parent_score-1e-8)
        if s['selected_candidate'] != accepted or s['numerical'] != numerical or s['candidate_train_score'] != candidate_score or s['parent_train_score'] != parent_score:
            problems.append('Endpoint acceptance arithmetic')
        starts = None
        if accepted:
            a = core.arrays(root/'evaluations'/current['selected']/'starts.npz')
            starts = {k: a[k] for k in ('voltage_mV', 'gates', 'calcium_uM', 'indicator', 'peptide')}
        actual = core.parameter_audit(origin, core.arrays(selected_path), s['selected_coefficients'], starts)
        if actual != s['parameters'] or not actual['passed']:
            problems.append('Selected parameter/frozen audit')
        selected = fine if accepted else parent1
        try:
            held = core.loss(selected, m['panel'], ids, folds[index], 'heldout')
            parent_held = core.loss(parent1, m['panel'], ids, folds[index], 'heldout')
            if s['heldout'] != dict(loss=held, parent_loss=parent_held, unavailable=None):
                problems.append('Heldout arithmetic')
        except ValueError:
            if s['heldout']['unavailable'] is None:
                problems.append('Unrecorded heldout coverage gap')
    for path in root.glob('**/run_records/*/source_record.json'):
        for name, h in json.loads(path.read_text())['source_sha256'].items():
            if file_sha256(path.parent/'source'/name) != h:
                problems.append('Archived source '+name)
    outcome = dict(audit_complete=True, execution_complete=len(summary['folds']) == 5, integrity_passed=not problems, problems=problems,
        optimization_probes=summary['optimization_probes'], trained_folds=sum(f['probes'] > 0 for f in summary['folds']),
        selected_improvements=sum(f['selected_candidate'] for f in summary['folds']),
        pilot_descent_demonstrated=summary['pilot_descent_demonstrated'],
        pilot_ready_for_longer_fit=bool(not problems and all(f['probes'] > 0 for f in summary['folds']) and summary['pilot_descent_demonstrated']),
        fold_results=summary['folds'], parent_numerical=summary['parent_numerical'], signal_evidence='Per-phase cue-minus-sham indicator means, sham RMD voltage metrics, all-cell bounds and in-memory final-state finite reductions',
        unavailable=unavailable+['Four training targets only; no full135-target changed-model qualification',
            'No optimizer/reboot recovery drill; durable batch/pair continuation only, in-flight kernel/JIT restarts',
            'Full final neural states/trial trajectories intentionally not retained; in-memory checks are logged, not raw replay evidence',
            'No body propagation, sensory start/stop, swimming, biological or pristine-test qualification'],
        behavior_qualified=False, full_target_qualified=False, heldout_used_for_selection=False,
        artifact_sha256={str(p): file_sha256(p) for p in root.rglob('*.npz')})
    io.atomic_json(root/'audit.json', outcome)
    return outcome


def prune_temporary(root, outcome):
    """After verified audit only: retain all parameters, optimizer and JSON logs."""
    if not outcome['audit_complete'] or not outcome['integrity_passed'] or not outcome['execution_complete']:
        return
    journal = root/'retention.jsonl'
    # A completed retention pass is idempotent. A partial pass is also resumable.
    for path in (root/'evaluations').rglob('*.npz'):
        if path.name != 'starts.npz' and path.name != 'response.npz' and not path.name.startswith('batch-'):
            continue
        expected = outcome['artifact_sha256'][str(path)]
        if file_sha256(path) != expected:
            raise ValueError('Temporary array changed before pruning')
        row = dict(path=str(path), sha256=expected, bytes=path.stat().st_size, evidence_limit='Raw simulation array removed after audit; parameters and detailed metrics retained')
        with journal.open('a') as stream:
            stream.write(json.dumps(dict(event='delete_planned', **row))+'\n'); stream.flush(); os.fsync(stream.fileno())
        path.unlink(); io.sync_directory(path.parent)
        with journal.open('a') as stream:
            stream.write(json.dumps(dict(event='deleted', **row))+'\n'); stream.flush(); os.fsync(stream.fileno())
    io.atomic_json(root/'retention-complete.json', dict(parameters_and_logs_retained=True, raw_arrays_available=False,
        audit_before_prune=True, journal=str(journal)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, default=SOURCE); p.add_argument('--fold-source', type=Path, default=FOLDS)
    p.add_argument('--output-dir', type=Path); p.add_argument('--minutes', type=float, default=180)
    p.add_argument('--worker', type=int, choices=(0, 1)); p.add_argument('--coordinate', action='store_true')
    for flag in ('check', 'status', 'report', 'select-output'):
        p.add_argument('--'+flag, action='store_true')
    a = p.parse_args()
    if a.status or a.check:
        active = jobs(); print('READ-ONLY; nothing started. Active jobs:', json.dumps(active)); return 3 if a.check and active else 0
    if a.select_output:
        print(output_for(a.source_run)); return 0
    if a.output_dir is None:
        p.error('--output-dir required')
    root = a.output_dir.resolve()
    if a.report:
        print('READ-ONLY report:', root)
        for name in ('budget_receipt.json', 'audit.json'):
            if not (root/name).exists():
                print(name, 'not written yet'); continue
            row = json.loads((root/name).read_text())
            keys = ('status', 'elapsed_seconds', 'all_steps_finished', 'audit_complete', 'integrity_passed', 'optimization_probes',
                    'trained_folds', 'selected_improvements', 'pilot_ready_for_longer_fit', 'problems')
            print(name, {k: row[k] for k in keys if k in row})
        if (root/'summary.json').exists():
            for s in json.loads((root/'summary.json').read_text())['folds']:
                print('Fold', s['fold'], 'probes', s['probes'], 'selected', s['selected_candidate'], 'train', s['parent_train_score'], '->', s['selected_train_score'], 'heldout', s['heldout']['loss'])
        return 0
    if a.worker is not None:
        return worker(root, a.worker)
    if a.coordinate:
        return coordinate(root)
    if not np.isfinite(a.minutes) or a.minutes <= 0:
        p.error('Positive optimization allowance required')
    if jobs():
        p.error('Active GPU work detected; nothing started')
    import fcntl
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE/'.cv-peptide-validation.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error('Shared GPU lock held')
        if jobs():
            p.error('GPU work detected after lock')
        if root == REPO or REPO in root.parents or root in REPO.parents:
            p.error('External output required')
        root.mkdir(parents=True, exist_ok=True)
        receipt = root/'budget_receipt.json'
        if receipt.exists() and json.loads(receipt.read_text()).get('all_steps_finished'):
            print('Already completed; nothing started. Use --report.'); return 0
        # If audit finished but post-audit retention was interrupted, finish only
        # that retention, never repeat an expensive completed forward stage.
        if (root/'audit.json').exists():
            old = json.loads((root/'audit.json').read_text())
            if old.get('execution_complete') and old.get('integrity_passed'):
                prune_temporary(root, old)
                io.atomic_json(receipt, dict(status='finished', all_steps_finished=True, audit_complete=True, integrity_passed=True, resumed_retention_only=True))
                print('Completed retained audit/cleanup only; no forward computation repeated.'); return 0
        start = time.monotonic()
        job_path = root/'job.json'
        job = json.loads(job_path.read_text()) if job_path.exists() else dict(source_run=str(a.source_run.resolve()), fold_source=str(a.fold_source.resolve()), optimization_minutes=a.minutes)
        if job['source_run'] != str(a.source_run.resolve()) or job['fold_source'] != str(a.fold_source.resolve()):
            p.error('Source mismatch; never replace an existing source')
        io.atomic_json(job_path, job)
        io.atomic_json(receipt, dict(status='running', all_steps_finished=False, audit_complete=False))
        code = 1
        try:
            if not (root/'manifest.json').exists():
                if shutil.disk_usage(root).free < 5*1024**3:
                    raise ValueError('Requires 5 GiB temporary space')
                prepare(root, a.source_run.resolve(), a.fold_source.resolve())
            verify(root)
            capture_run_record(root, REPO, vars(a), phase='E129-three-hour-CV-pilot')
            from scripts.total_job_budget import run_group
            code, _ = run_group([sys.executable, '-u', '-m', MODULE, '--output-dir', str(root), '--coordinate'], None)
        except Exception as exc:
            traceback.print_exc(); io.atomic_json(root/'failure.json', dict(error=repr(exc)))
        try:
            outcome = audit(root)
        except Exception as exc:
            traceback.print_exc(); outcome = dict(audit_complete=False, execution_complete=False, integrity_passed=False, problems=[repr(exc)])
            io.atomic_json(root/'audit.json', outcome)
        if code == 0 and outcome['integrity_passed'] and outcome['execution_complete']:
            prune_temporary(root, outcome)
        result_row = dict(status='finished', exit_code=code, elapsed_seconds=time.monotonic()-start,
            optimization_minutes=job['optimization_minutes'], validation_wallclock_cutoff=False,
            all_steps_finished=bool(code == 0 and outcome['audit_complete'] and outcome['execution_complete'] and outcome['integrity_passed']),
            audit_complete=outcome['audit_complete'], integrity_passed=outcome['integrity_passed'],
            pilot_ready_for_longer_fit=outcome.get('pilot_ready_for_longer_fit', False), behavior_qualified=False)
        io.atomic_json(receipt, result_row); print(json.dumps(result_row)); return 0 if result_row['all_steps_finished'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
