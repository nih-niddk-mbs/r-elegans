"""E113 bounded atlas physiology fit with a hard total budget and automatic audit."""
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
from scripts.experiment_records import capture_run_record, file_sha256
from scripts.selected_atlas_audit_checks import saved_arrays
from scripts.stable_atlas_worker import atomic_json as write
from scripts.stable_atlas_policy import Coordinates, loss, prior, adam_step, choose_batch, eligible_targets, preserved, admit
from scripts.atlas_shared_checks import observed_pairs, compare_downstream
from scripts.audit_atlas_composite import COARSE_STEPS, FINE_STEPS
from scripts.run_atlas_shared_state import SELECTED_SHA, ANCHOR_SHA

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = '/home/carsonc/r-elegans-results/atlas-composite-audit-20261122T041912596864Z/summary.json'


def prepare(source, root):
    s = json.loads(source.read_text())
    if not s.get('audit_complete') or not s.get('integrity_passed') or not s['models']['selected']['full_atlas_per_phase_checks_passed']:
        raise ValueError('Completed E112 selected full-atlas audit required')
    evidence = {str(source):file_sha256(source)}
    for key in ('input_sha256', 'output_sha256'):
        for path, sha in s[key].items():
            if file_sha256(path) != sha:raise ValueError('E112 evidence changed: '+path)
            evidence[path] = sha
    inputs = root/'inputs';inputs.mkdir()
    levels = s['models']['selected']['levels']
    fine_root = Path(levels[0]['fine_control']['path']).parents[1]
    fm = json.loads((fine_root/'manifest.json').read_text());model = fm['models']['selected']
    def copy(path, name, expected=None):
        sha = file_sha256(path)
        if expected is not None and sha != expected:raise ValueError('Pinned parent identity differs')
        evidence[str(path)] = sha;dest = inputs/name;shutil.copy2(path, dest)
        if file_sha256(dest) != sha:raise ValueError('Copy mismatch')
        evidence[str(dest)] = sha;return str(dest)
    parent = copy(model['path'], 'parent.npz', SELECTED_SHA)
    anchors = copy(model['anchors'], 'anchors.npz', ANCHOR_SHA['selected'])
    statistics = copy(fm['statistics'], 'statistics.json')
    baselines = [copy(r['artifact'], 'baseline-'+str(i)+'.npz', r['sha256']) for i, r in enumerate(levels)]
    references = {}
    for row in levels:
        for group in ('coarse', 'fine'):
            control = row[group+'_control'];step = str(control['neural_step_ms'])
            arrays = saved_arrays(control['path']);v = arrays['reference_sham_rmd_mV']
            from r_elegans.brain.rhythm_qualification import rhythm_metrics
            references[step] = [[rhythm_metrics(v[p, :, j], .001) for j in range(2)] for p in range(4)]
    manifest = dict(experiment='E113', parent=parent, parent_sha256=SELECTED_SHA, anchors=anchors,
        anchors_sha256=ANCHOR_SHA['selected'], statistics=statistics, baselines=baselines,
        targets=json.loads(Path(statistics).read_text())['targets'], hard=s['protocol']['fine_group_targets'],
        stimulus_nS=fm['stimulus_nS'], source=str(source), evidence_sha256=evidence,
        forward_identity=s['forward_identities']['selected-coarse'], rhythm_references=references,
        seed=1130, policy=dict(bound=.1, lr=.002, perturbation=.005, prior=.001, minibatch=2,
        training_cutoff_minutes=330, total_minutes=480, selection='complete validation loss; no test-based selection'),
        scope='Physiology fitting only; no behavioral or biological qualification')
    write(root/'manifest.json', manifest);return manifest


class Pool:
    def __init__(self, root):
        self.root = root;self.processes = [];self.logs = []
        for gpu in range(2):
            folder = root/'workers'/str(gpu);folder.mkdir(parents=True, exist_ok=True)
            for name in ('STOP','ready.json','startup-error.json','task.json'):
                (folder/name).unlink(missing_ok=True)
            log = open(folder/'worker.log', 'a');self.logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false')
            self.processes.append(subprocess.Popen([sys.executable, '-u', '-m', 'scripts.stable_atlas_worker',
                '--root', str(root), '--gpu', str(gpu)], env=env, stdout=log, stderr=subprocess.STDOUT))
        deadline = time.monotonic()+180
        while not all((root/'workers'/str(g)/'ready.json').exists() for g in range(2)):
            self.check()
            if time.monotonic() > deadline:raise TimeoutError('GPU worker startup timeout')
            time.sleep(.5)

    def check(self):
        for p in self.processes:
            if p.poll() is not None:raise RuntimeError('GPU worker exited: '+str(p.returncode))

    def tasks(self, tasks, deadline):
        pending = list(enumerate(tasks));active = {};results = [None]*len(tasks);last = 0.
        while pending or active:
            self.check()
            if time.monotonic() >= deadline:raise TimeoutError('Phase deadline reached')
            for gpu in list(active):
                i, task = active[gpu]
                if Path(task['result']).exists():
                    r = json.loads(Path(task['result']).read_text());results[i] = r;del active[gpu]
                    if r['status'] != 'completed':raise RuntimeError('Evaluation failed: '+str(r))
            for gpu in range(2):
                if gpu not in active and pending:
                    i, task = pending.pop(0);path = Path(task['result']).parent
                    path.mkdir(parents=True, exist_ok=False);write(path/'task.json', task)
                    write(self.root/'workers'/str(gpu)/'task.json', task);active[gpu] = (i, task)
            if time.monotonic()-last > 30:
                print(json.dumps(dict(active_tasks=[t['name'] for _, t in active.values()], pending=len(pending),
                    remaining_phase_minutes=(deadline-time.monotonic())/60)), flush=True);last = time.monotonic()
            time.sleep(.25)
        return results

    def close(self):
        for gpu in range(2):(self.root/'workers'/str(gpu)/'STOP').touch()
        for p in self.processes:
            if p.poll() is None:p.terminate()
        for p in self.processes:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in self.logs:log.close()


def task(root, name, model, anchors=None, targets=None, step=None, stimulus=None):
    out = root/'tasks'/name
    d = dict(name=name, kind='autonomy' if anchors is None else 'replay', model=str(model),
        model_sha256=file_sha256(model), output=str(out/'artifacts'), result=str(out/'result.json'))
    if anchors is not None:d.update(anchors=str(anchors), anchors_sha256=file_sha256(anchors), targets=targets, step_ms=step, stimulus_nS=stimulus)
    return d


def replay_tasks(root, manifest, name, model, anchors, targets, levels):
    return [task(root, name+'-'+str(level)+'-'+group, model, anchors, chunk, steps[level], manifest['stimulus_nS'])
            for level in levels for group, steps in [('coarse', COARSE_STEPS), ('fine', FINE_STEPS)]
            if (chunk := [t for t in targets if (t in manifest['hard']) == (group == 'fine')])]


def analyze_results(results, manifest, statistics, ids, targets, levels, split, reference=True, final=False):
    matrices = {level:np.empty((4, len(targets), len(ids))) for level in levels}
    filled = {level:set() for level in levels};controls = True;rhythms = True
    for r in results:
        spec = r['task'];step = spec['step_ms'];group = 'fine' if spec['targets'][0] in manifest['hard'] else 'coarse'
        level = list(FINE_STEPS if group == 'fine' else COARSE_STEPS).index(step)
        a = saved_arrays(r['artifact'])
        if file_sha256(r['artifact']) != r['sha256'] or a['targets'].tolist() != spec['targets']:raise ValueError('Task artifact mismatch')
        if str(a['anchors_sha256']) != spec['anchors_sha256'] or not np.isfinite(a['response']).all():raise ValueError('Task anchors/finite check')
        for i, t in enumerate(spec['targets']):
            if t in filled[level]:raise ValueError('Duplicate target')
            matrices[level][:, targets.index(t)] = a['response'][:, i];filled[level].add(t)
        controls &= bool(r['controls_passed'])
        base = manifest['rhythm_references'][str(step)]
        rhythms &= all(preserved(base[p][j], r['rhythms'][p][j]) if reference else r['rhythms'][p][j]['passed']
                       for p in range(4) for j in range(2))
    if any(filled[l] != set(targets) for l in levels):raise ValueError('Missing target/grid coverage')
    # Optimization screens consume labels from requested split only, never test labels.
    support_stats = statistics if final else dict(stats={split:statistics['stats'][split]})
    pairs = sorted({tuple(p) for s in support_stats['stats'].values() for p in s['pairs'] if p[0] in targets and p[0] != p[1]})
    ti = np.asarray([targets.index(t) for t, _ in pairs]);ri = np.asarray([ids.index(r) for _, r in pairs])
    checks = []
    for first, second in zip(levels[:-1], levels[1:]):
        a, b = matrices[first][:, ti, ri], matrices[second][:, ti, ri]
        checks.append([compare_downstream(x, y, pairs)[0] for x, y in list(zip(a, b))+[(a.mean(axis=0), b.mean(axis=0))]])
    losses = {str(l):loss(matrices[l], targets, ids, statistics, split) for l in levels}
    numerical = all(c['mixed_tolerance_passed'] and (not final or c['vector']['passed']) for row in checks for c in row)
    loss_stable = max(losses.values())-min(losses.values()) <= .02
    return dict(passed=bool(controls and rhythms and numerical and loss_stable), controls_passed=bool(controls),
        rhythms_passed=bool(rhythms), numerical_passed=bool(numerical), loss_grid_stable=bool(loss_stable),
        comparisons=checks, losses=losses, data_loss=losses[str(levels[-1])],
        targets=targets, observed_pairs=len(pairs)), matrices


def coordinator(root):
    budget = json.loads((root/'deadline.json').read_text());start = budget['started_monotonic']
    train_deadline = start+330*60;final_deadline = budget['deadline_monotonic']-600
    mfest = prepare(Path(budget['source']), root);statistics = json.loads(Path(mfest['statistics']).read_text())
    arrays = saved_arrays(mfest['parent']);ids = arrays['neuron_ids'].tolist();codec = Coordinates(arrays)
    rng = np.random.default_rng(1130);x = np.zeros(codec.size);mom = x.copy();var = x.copy()
    pool = None;ledger = [];iteration = 0;accepted = 0;moment_steps = 0;selected = None
    checkpoint_dir = root/'checkpoints';checkpoint_dir.mkdir()
    def save(name, values):
        folder = checkpoint_dir/name;folder.mkdir();path = folder/'network.npz'
        np.savez_compressed(path, **codec.apply(values));np.savez_compressed(folder/'coordinates.npz', offsets=values)
        write(folder/'identity.json', dict(sha256=file_sha256(path), coordinates_sha256=file_sha256(folder/'coordinates.npz')))
        return path
    current = save('initial', x);selected = current;best_x = x.copy()
    baseline = saved_arrays(mfest['baselines'][-1])['response']
    best_score = loss(baseline, mfest['targets'], ids, statistics, 'val');parent_score = best_score
    train_pool = eligible_targets(statistics, 'train')
    val_targets = sorted({t for t, r in statistics['stats']['val']['pairs'] if t != r})
    last_validation = time.monotonic();recent = 600.;state_number = 0
    def save_state():
        nonlocal state_number
        state_number += 1;folder = root/'optimizer'/f'{state_number:05d}';folder.mkdir(parents=True)
        np.savez_compressed(folder/'adam.npz', current=x, first_moment=mom, second_moment=var, iteration=iteration, moment_steps=moment_steps)
        write(folder/'rng.json', rng.bit_generator.state)
        write(folder/'selection.json', dict(current=str(current), selected=str(selected), validation_loss=best_score,
            accepted=accepted, iteration=iteration, semantics='Fresh trajectory; no exact live-worker-queue resume implemented; Adam moments advance on valid perturbation pairs, even if update rejected'))
        write(root/'history.json', ledger)
    try:
        pool = Pool(root)
        # Source endpoint replay before fitting; both numerical groups when available.
        batch = choose_batch(rng, train_pool, mfest['hard'], 0)
        replay_spec = replay_tasks(root, mfest, 'source-check', current, mfest['anchors'], batch, [1,2])
        source_results = pool.tasks(replay_spec, train_deadline)
        screen, matrices = analyze_results(source_results, mfest, statistics, ids, batch, [1,2], 'train')
        differences = [float(abs(matrices[l]-saved_arrays(mfest['baselines'][l])['response'][:,
            [mfest['targets'].index(t) for t in batch]]).max()) for l in [1,2]]
        write(root/'source-replay.json', dict(screen=screen, maximum_response_errors=differences))
        if not screen['passed'] or max(differences) > 1e-10:raise ValueError('Source endpoint replay/screen failed')
        while admit(time.monotonic(), train_deadline, recent):
            began = time.monotonic();iteration += 1
            batch = choose_batch(rng, train_pool, mfest['hard'], iteration)
            direction = rng.choice([-1., 1.], codec.size)
            record = dict(iteration=iteration, batch=batch, current_before=str(current), rng_before=rng.bit_generator.state)
            round_dir = root/'rounds'/f'{iteration:05d}';round_dir.mkdir(parents=True)
            np.savez_compressed(round_dir/'direction.npz', direction=direction)
            plus_x = np.clip(x+.005*direction, -.1, .1);minus_x = np.clip(x-.005*direction, -.1, .1)
            plus = save(f'{iteration:05d}-plus', plus_x);minus = save(f'{iteration:05d}-minus', minus_x)
            record.update(plus=str(plus), minus=str(minus));write(round_dir/'intent.json', record)
            specs = replay_tasks(root, mfest, f'r{iteration}-plus', plus, mfest['anchors'], batch, [1,2])
            specs += replay_tasks(root, mfest, f'r{iteration}-minus', minus, mfest['anchors'], batch, [1,2])
            results = pool.tasks(specs, train_deadline);half = len(results)//2
            a, _ = analyze_results(results[:half], mfest, statistics, ids, batch, [1,2], 'train')
            b, _ = analyze_results(results[half:], mfest, statistics, ids, batch, [1,2], 'train')
            record.update(plus_metrics=a, minus_metrics=b, accepted=False)
            if a['passed'] and b['passed']:
                gradient = ((a['data_loss']+prior(plus_x))-(b['data_loss']+prior(minus_x)))/.01*direction
                moment_steps += 1
                proposal_x, mom, var = adam_step(x, mom, var, gradient, moment_steps)
                proposal = save(f'{iteration:05d}-proposal', proposal_x);record['proposal'] = str(proposal)
                specs = replay_tasks(root, mfest, f'r{iteration}-current', current, mfest['anchors'], batch, [1,2])
                specs += replay_tasks(root, mfest, f'r{iteration}-proposal', proposal, mfest['anchors'], batch, [1,2])
                results = pool.tasks(specs, train_deadline);half = len(results)//2
                c, _ = analyze_results(results[:half], mfest, statistics, ids, batch, [1,2], 'train')
                d, _ = analyze_results(results[half:], mfest, statistics, ids, batch, [1,2], 'train')
                record.update(current_metrics=c, proposal_metrics=d)
                if c['passed'] and d['passed'] and d['data_loss']+prior(proposal_x) < c['data_loss']+prior(x):
                    x = proposal_x;current = proposal;accepted += 1;record['accepted'] = True
            ledger.append(record);write(round_dir/'decision.json', record);save_state()
            print(json.dumps(dict(iteration=iteration, accepted=record['accepted'], selected_validation_loss=best_score)), flush=True)
            recent = max(60., time.monotonic()-began)
            # Full validation panel is expensive; admit only with substantial remaining search budget.
            if accepted and time.monotonic()-last_validation >= 1800 and train_deadline-time.monotonic() > max(1800, 2*recent):
                val_start = time.monotonic()
                specs = replay_tasks(root, mfest, f'validation-{iteration}', current, mfest['anchors'], val_targets, [1,2])
                results = pool.tasks(specs, train_deadline)
                metrics, _ = analyze_results(results, mfest, statistics, ids, val_targets, [1,2], 'val')
                score = metrics['data_loss']+prior(x)
                chosen = bool(metrics['passed'] and score < best_score)
                write(root/f'validation-{iteration}.json', dict(model=str(current), sha256=file_sha256(current),
                    offsets=x.tolist(), metrics=metrics, score=score, selected=chosen, previous_score=best_score))
                if chosen:best_score = score;selected = current;best_x = x.copy()
                last_validation = time.monotonic();recent = max(recent, time.monotonic()-val_start);save_state()
    except TimeoutError as exc:
        write(root/'training-stop.json', dict(reason='optimization_deadline', detail=str(exc), iteration=iteration))
    finally:
        if pool is not None:pool.close()
    # Freeze selection before own-state/final/test evaluation. No test-based fallback.
    save_state();final_model = save('optimizer-final', x);chosen_model = save('validation-selected', best_x)
    write(root/'selection.json', dict(selected=str(chosen_model), sha256=file_sha256(chosen_model),
        source=str(selected), selected_validation_loss=best_score, parent_validation_loss=parent_score,
        optimizer_final=str(final_model), accepted_updates=accepted, iterations=iteration,
        criterion='Validation only, before own-state/final/test; unchanged initial parent is not a trained success'))
    pool = Pool(root)
    try:
        auto = pool.tasks([task(root, 'autonomy-parent', mfest['parent']), task(root, 'autonomy-selected', chosen_model)], final_deadline)
        write(root/'autonomy.json', dict(parent=auto[0], selected=auto[1]))
        fresh = auto[1]['anchors'];parent_fresh = auto[0]['anchors']
        specs = replay_tasks(root, mfest, 'final', chosen_model, fresh, mfest['targets'], [0,1,2])
        specs += replay_tasks(root, mfest, 'fresh-parent-validation', mfest['parent'], parent_fresh, val_targets, [2])
        results = pool.tasks(specs, final_deadline)
        count = len(replay_tasks(root, mfest, 'unused', chosen_model, fresh, mfest['targets'], [0,1,2]))
        metrics, matrices = analyze_results(results[:count], mfest, statistics, ids, mfest['targets'], [0,1,2], 'val', reference=False, final=True)
        parent_metrics, _ = analyze_results(results[count:], mfest, statistics, ids, val_targets, [2], 'val', reference=False)
        scores = {split:loss(matrices[2], mfest['targets'], ids, statistics, split) for split in ('train','val','test')}
        np.savez_compressed(root/'final-matrices.npz', **{'level_'+str(l):v for l, v in matrices.items()})
        autonomy_pass = all(preserved(auto[0]['rhythms'][l][j], auto[1]['rhythms'][l][j]) for l in range(2) for j in range(2))
        params = codec.audit(saved_arrays(chosen_model), best_x)
        passed = bool(metrics['passed'] and parent_metrics['passed'] and autonomy_pass and params['passed'] and
            params['changed_coordinates'] > 0 and best_score < parent_score and scores['val'] < parent_metrics['data_loss'])
        result = dict(experiment='E113', execution_complete=True, validation_complete=True, iterations=iteration,
            accepted_updates=accepted, fit_qualified_before_audit=passed, final_metrics=metrics,
            fresh_parent_validation=parent_metrics, scores=scores, autonomy_passed=autonomy_pass,
            parameters=params, selected_sha256=file_sha256(chosen_model), model_promoted=False,
            biological_fidelity_qualified=False, sensory_start_stop_qualified=False, swimming_qualified=False,
            limits=['Supervised conditional atlas statistics; not calibrated calcium amplitudes or behavior.',
                    'Historical test set repeatedly inspected; no untouched holdout claim.',
                    'No new sensory start/stop, physical body or learned swimming evidence.'])
        write(root/'summary.json', result)
    finally:pool.close()


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path(DEFAULT_SOURCE));p.add_argument('--coordinator', action='store_true');p.add_argument('--report', action='store_true')
    a = p.parse_args();root = a.output_dir.resolve()
    if a.report:
        for name in ('summary.json','audit.json','budget_receipt.json'):
            path = root/name;print(name+':', path.read_text() if path.exists() else 'not written yet')
        return
    if a.coordinator:
        try:coordinator(root)
        except Exception as exc:
            traceback.print_exc();write(root/'failure.json', dict(error=repr(exc)));raise
        return
    if root == REPO or REPO in root.parents or root in REPO.parents:p.error('External output required')
    start = time.monotonic();deadline = start+480*60;root.mkdir(parents=True, exist_ok=False)
    write(root/'deadline.json', dict(started_monotonic=start, deadline_monotonic=deadline,
        source=str(a.source.resolve()), total_minutes=480, optimization_minutes=330,
        started_utc=datetime.now(timezone.utc).isoformat(), includes='Preparation, optimization, fresh states, validation, audit and cleanup'))
    capture_run_record(root, REPO, vars(a), phase='stable-atlas-supervisor')
    from scripts.total_job_budget import run_group
    code, timeout = run_group([sys.executable, '-u', '-m', 'scripts.run_stable_atlas_fit',
        '--output-dir', str(root), '--coordinator'], max(.01, deadline-time.monotonic()-600))
    # Bounded separate saved-data audit; watchdog kills surviving GPU descendants first.
    audit_code, audit_timeout = run_group([sys.executable, '-u', '-m', 'scripts.audit_stable_atlas_fit',
        '--run', str(root)], max(.01, deadline-time.monotonic()-10))
    audit_path = root/'audit.json';audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    summary_path = root/'summary.json';summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    elapsed = time.monotonic()-start
    finished = bool(code == 0 and not timeout and audit_code == 0 and not audit_timeout and audit.get('audit_complete') and audit.get('execution_complete') and elapsed <= 480*60)
    receipt = dict(exit_code=code, timed_out=timeout, audit_exit_code=audit_code, audit_timed_out=audit_timeout,
        total_minutes=480, elapsed_seconds=elapsed, all_steps_finished=finished,
        validation_complete=bool(summary.get('validation_complete')), audit_complete=bool(audit.get('audit_complete')),
        fit_qualified=bool(finished and audit.get('integrity_passed') and summary.get('fit_qualified_before_audit')),
        report=str(summary_path), audit=str(audit_path))
    write(root/'budget_receipt.json', receipt);print(json.dumps(receipt), flush=True)
    raise SystemExit(0 if finished else 2)


if __name__ == '__main__':main()
