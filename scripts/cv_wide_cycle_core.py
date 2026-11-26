"""E130 wider-panel evaluations and measured GPU batch calibration."""
import json
import shutil
from pathlib import Path
import time
import numpy as np
from scripts import run_full_cycle_validation as validated
from scripts.experiment_records import file_sha256
from scripts.stable_atlas_policy import Coordinates, eligible_targets, loss
from scripts.cv_fit_core import group_directions, raw_peptide_gains, penalty
from scripts.cv_cycle_pilot_core import decode, parameter_audit, score, choose

io = validated.resume
PARENT_SHA = 'f626ded0494902820432046f1031cafe35b65216cb25ecc6ffb1cee2d829e652'
GROUPS = ('inward', 'outward', 'chemical', 'gap', 'peptide_balance')


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def panel_for(folds, source_targets, seed=1300, return_state=False):
    common = set(source_targets)
    for fold in folds:
        common.intersection_update(eligible_targets(fold, 'train'))
    if len(common) < 8:
        raise ValueError('At least eight common training-eligible targets required')
    rng = np.random.default_rng(seed)
    panel = sorted(rng.choice(sorted(common), min(16, len(common)), replace=False).tolist())
    return (panel, rng.bit_generator.state) if return_state else panel


def check_response(y, times, tolerance):
    return validated.cycle.ensemble_metrics(y, times, y[:64], tolerance)


def numerical(coarse, fine, tolerance):
    error = float(abs(coarse.mean(0)-fine.mean(0)).max())
    return dict(maximum_difference=error, tolerance=tolerance, passed=error <= tolerance)


def batch_metrics(means, final, traces):
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    import jax
    means = np.asarray(means)
    voltage = np.asarray(traces[0])
    duplicate_mean = float(abs(means[:, -2]-means[:, -1]).max())
    duplicate_voltage = float(abs(voltage[:, 0]-voltage[:, 1]).max())
    failed = []
    for phase in range(16):
        for cell in range(2):
            if not rhythm_metrics(voltage[phase, 0, :, cell], .001)['passed']:
                failed.append(dict(phase=phase, cell=cell))
    lo, hi = float(np.asarray(traces[4]).min()), float(np.asarray(traces[5]).max())
    finite = bool(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves((means, final, traces))))
    return dict(finite=finite, voltage_min_mV=lo, voltage_max_mV=hi,
                duplicate_mean_error=duplicate_mean, duplicate_voltage_error_mV=duplicate_voltage,
                failed_rhythms=failed, passed=bool(finite and lo >= -120 and hi <= 80 and duplicate_mean <= 1e-12 and duplicate_voltage <= 1e-10 and not failed),
                evidence='In-memory full end-state/trace reductions; full final states and trial traces deliberately not archived')


class Engine:
    """Persistent per-GPU kernels; checkpoint every candidate state and batch."""
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        self.cache = {}

    def evaluate(self, task):
        import jax
        import jax.numpy as jnp
        from r_elegans.brain.peptide_checkpoint import load_model, load_state
        from r_elegans.brain.midpoint_replay import make_anchors, make_replay
        from r_elegans.brain.cycle_ensemble import make_extension
        folder = self.root/'evaluations'/task['id']
        folder.mkdir(parents=True, exist_ok=True)
        model = self.root/task['model']
        if file_sha256(model) != task['model_sha256']:
            raise ValueError('Evaluation model identity changed')
        level = task['level']; dt, step = io.old.clock(level)
        panel = task.get('targets', self.manifest['panel'])
        p, f, config, pp, exc, inh, _, origin = load_model(model)
        ids = origin['neuron_ids'].tolist()
        if level not in self.cache:
            self.cache[level] = (make_anchors(config, ids.index('RMDL'), dt, step),
                                 make_extension(config, ids.index('RMDL'), dt, step),
                                 make_replay(config, [ids.index('RMDL'), ids.index('RMDR')], dt, step, trace_mode='shams'))
        anchor, extend, replay = self.cache[level]
        reference = self.root/'evaluations'/task.get('starts_from', f'benchmark-{level}')
        if task.get('kind') != 'benchmark' and (task.get('starts_from') or task['model_sha256'] == self.manifest.get('parent_sha256')) and (reference/'states-verified.json').exists():
            saved = json.loads((reference/'states-verified.json').read_text())
            if saved['model_sha256'] != task['model_sha256'] or any(file_sha256(reference/name) != h for name, h in saved['sha256'].items()):
                raise ValueError('Parent benchmark states changed')
            meta = json.loads((reference/'phases.json').read_text())
            if meta['base']['record_dt_s'] != dt:
                raise ValueError('Reused state clock mismatch')
            for name in ('starts.npz', 'phases.json', 'states-verified.json'):
                dest = folder/name
                if not dest.exists():
                    shutil.copy2(reference/name, dest)
                if file_sha256(dest) != file_sha256(reference/name):
                    raise ValueError('Shared parent state mismatch')
            io.atomic_json(folder/'state-source.json', dict(reused_verified_parent_states=str(reference)))
        start_path = folder/'starts.npz'
        meta_path = folder/'phases.json'
        checkpoint = folder/'states-verified.json'
        tick = time.monotonic()
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved['model_sha256'] != task['model_sha256'] or any(file_sha256(folder/name) != h for name, h in saved['sha256'].items()):
                raise ValueError('Starting-state checkpoint changed')
            starts = load_state(start_path); meta = json.loads(meta_path.read_text())
        else:
            # A single atomic NPZ stores both full states and metadata, allowing
            # recovery if JSON materialization is interrupted after the write.
            if start_path.exists():
                saved = arrays(start_path)
                if str(saved.pop('model_sha256')) != task['model_sha256']:
                    raise ValueError('Starting-state model mismatch')
                meta = json.loads(str(saved.pop('metadata_json')))
                starts = load_state(start_path)
            else:
                base, first = anchor(p, f, pp, exc, inh)
                extra, _, second = extend(p, f, pp, exc, inh, jax.tree.map(lambda v: v[-1], base))
                starts = jax.tree.map(lambda a, b: jnp.concatenate((a, b)), base, extra)
                finite = all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(starts))
                if not finite or float(np.asarray(starts.neural.voltage_mV).min()) < -120 or float(np.asarray(starts.neural.voltage_mV).max()) > 80 or not second['finite'] or second['voltage_min_mV'] < -120 or second['voltage_max_mV'] > 80:
                    raise ValueError('Fresh own-state initialization failed')
                first_times = (60+(np.asarray(first['indices']).reshape(4, 4)+1)*dt).mean(1)
                offset = 60+(first['indices'][-1]+1)*dt
                times = first_times.tolist()+[offset+t for t in second['cycle_mean_time_s']]
                meta = dict(base=first, extension=second, cycle_times=times, model_sha256=task['model_sha256'])
                state_values = jax.tree.map(np.asarray, starts)
                io.atomic_npz(start_path, **state_values.neural._asdict(), peptide=state_values.peptide,
                              model_sha256=np.array(task['model_sha256']), metadata_json=np.array(json.dumps(meta)))
                starts = load_state(start_path)
            io.atomic_json(meta_path, meta)
            io.atomic_json(checkpoint, dict(model_sha256=task['model_sha256'], sha256={name: file_sha256(folder/name) for name in ('starts.npz', 'phases.json')}))
        state_ready_seconds = time.monotonic()-tick
        if task.get('kind') == 'benchmark':
            return self.benchmark(task, folder, p, f, pp, exc, inh, starts, replay, ids, tick)
        capacity = json.loads((self.root/'batch-policy.json').read_text())[str(level)]['capacity']
        pieces = []; checks = []; hashes = {}; known_forward_seconds = 0.
        for group in range(8):
            group_parts = []
            for begin in range(0, len(panel), capacity):
                names = panel[begin:begin+capacity]
                path = folder/f'batch-{group:02d}-target-{begin:03d}.npz'
                record = path.with_suffix('.json')
                provenance = dict(task=task, group=group, begin=begin, targets=names, capacity=capacity, starts_sha256=file_sha256(start_path))
                benchmark_path = reference/f'batch-benchmark-{capacity:02d}-warm-{begin:03d}.npz'
                if not path.exists() and group == 0 and panel == self.manifest['panel'] and task['model_sha256'] == self.manifest.get('parent_sha256') and benchmark_path.exists():
                    data = arrays(benchmark_path); original = json.loads(str(data['record_json']))
                    if original['provenance']['task']['model_sha256'] != task['model_sha256'] or original['provenance']['starts_sha256'] != file_sha256(start_path):
                        raise ValueError('Reusable benchmark batch identity mismatch')
                    row = dict(provenance=provenance, forward_seconds=0., checks=original['checks'], source_batch=dict(path=str(benchmark_path), sha256=file_sha256(benchmark_path)))
                    io.atomic_npz(path, means=data['means'], record_json=np.array(json.dumps(row)))
                if path.exists():
                    data = arrays(path)
                    row = json.loads(str(data.pop('record_json')))
                    if row['provenance'] != provenance or (record.exists() and json.loads(record.read_text())['sha256'] != file_sha256(path)):
                        raise ValueError('Saved response batch changed')
                    means = data['means']
                else:
                    io.atomic_json(self.root/f"progress-{self.manifest.get('worker_slot', 0)}.json", dict(task=task['id'], group=group, begin=begin, capacity=capacity, stage='response_batch'))
                    stimuli = np.zeros((capacity+2, 302)); stimuli[np.arange(len(names)), [ids.index(t) for t in names]] = .03
                    initial = jax.tree.map(lambda v: v[group*16:(group+1)*16], starts)
                    before = time.monotonic()
                    means, final, traces = jax.tree.map(np.asarray, replay(p, f, pp, exc, inh, initial, jnp.asarray(stimuli)))
                    forward_seconds = time.monotonic()-before
                    row = dict(provenance=provenance, forward_seconds=forward_seconds, checks=batch_metrics(means, final, traces))
                    io.atomic_npz(path, means=means, record_json=np.array(json.dumps(row)))
                    data = arrays(path); means = data['means']
                    if json.loads(str(data['record_json'])) != row:
                        raise ValueError('Batch readback failed')
                if means.shape != (16, capacity+2, 302) or not np.isfinite(means).all():
                    raise ValueError('Invalid compact response shape/values')
                io.atomic_json(record, dict(sha256=file_sha256(path), **row))
                hashes[path.name] = file_sha256(path)
                known_forward_seconds += row.get('forward_seconds', 0.)
                group_parts.append(means[:, :len(names)]-means[:, -1:]); checks.append(row['checks'])
            pieces.append(np.concatenate(group_parts, axis=1))
        response = np.concatenate(pieces)
        sampling = check_response(response, meta['cycle_times'], self.manifest['sampling_tolerances'][str(level)])
        path = folder/'response.npz'
        io.atomic_npz(path, response=response)
        result = dict(id=task['id'], status='completed', task=task, sampling=sampling, batch_checks=checks,
                      safe=bool(sampling['passed'] and all(c['passed'] for c in checks)), elapsed_seconds=time.monotonic()-tick,
                      response_sha256=file_sha256(path), batch_sha256=hashes,
                      state_ready_seconds=state_ready_seconds, known_saved_batch_forward_seconds=known_forward_seconds,
                      model_sha256=task['model_sha256'], fresh_own_states=True, cycle_times=meta['cycle_times'], batch_capacity=capacity)
        io.atomic_json(folder/'result.json', result)
        return result

    def benchmark(self, task, folder, p, f, pp, exc, inh, starts, replay, ids, tick):
        """Full panel on identical first16 states, cold/warm, atomic per tile."""
        import jax
        import jax.numpy as jnp
        cases = []; reference = None
        initial = jax.tree.map(lambda v: v[:16], starts)
        for capacity in (4, 8, 16):
            records = []; predictions = []; complete = True; error = None
            try:
                for repeat in ('first', 'warm'):
                    parts = []; seconds = 0.; checks = []
                    for begin in range(0, len(self.manifest['panel']), capacity):
                        names = self.manifest['panel'][begin:begin+capacity]
                        path = folder/f'batch-benchmark-{capacity:02d}-{repeat}-{begin:03d}.npz'
                        provenance = dict(task=task, capacity=capacity, repeat=repeat, begin=begin, targets=names,
                                          starts_sha256=file_sha256(folder/'starts.npz'))
                        if path.exists():
                            data = arrays(path); row = json.loads(str(data['record_json']))
                            if row['provenance'] != provenance:
                                raise ValueError('Benchmark provenance changed')
                        else:
                            stimuli = np.zeros((capacity+2, 302)); stimuli[np.arange(len(names)), [ids.index(t) for t in names]] = .03
                            io.atomic_json(self.root/f"progress-{self.manifest.get('worker_slot', 0)}.json", dict(stage='benchmark', capacity=capacity, repeat=repeat, begin=begin, task=task['id']))
                            before = time.monotonic()
                            means, final, traces = jax.tree.map(np.asarray, replay(p, f, pp, exc, inh, initial, jnp.asarray(stimuli)))
                            elapsed = time.monotonic()-before
                            row = dict(provenance=provenance, seconds=elapsed, checks=batch_metrics(means, final, traces))
                            io.atomic_npz(path, means=means, record_json=np.array(json.dumps(row)))
                            data = arrays(path)
                        if data['means'].shape != (16, capacity+2, 302) or not np.isfinite(data['means']).all():
                            raise ValueError('Invalid benchmark values')
                        side = path.with_suffix('.json')
                        if side.exists() and json.loads(side.read_text())['sha256'] != file_sha256(path):
                            raise ValueError('Benchmark checksum changed')
                        io.atomic_json(side, dict(sha256=file_sha256(path), **row))
                        parts.append(data['means'][:, :len(names)]-data['means'][:, -1:]); seconds += row['seconds']; checks.append(row['checks'])
                    predictions.append(np.concatenate(parts, axis=1)); records.append(dict(repeat=repeat, seconds=seconds, checks=checks))
                if reference is None:
                    reference = predictions[1]
                difference = float(abs(predictions[1]-reference).max())
                repeat_error = float(abs(predictions[0]-predictions[1]).max())
                passed = all(c['passed'] for r in records for c in r['checks']) and difference <= 1e-10 and repeat_error <= 1e-10
                try:
                    memory = jax.devices()[0].memory_stats()
                    memory = None if memory is None else {k: int(v) for k, v in memory.items() if k in ('bytes_in_use', 'peak_bytes_in_use', 'bytes_limit') and v is not None}
                except Exception as exc:
                    memory = dict(unavailable=repr(exc))
                cases.append(dict(capacity=capacity, passed=bool(passed), first_seconds=records[0]['seconds'], warm_seconds=records[1]['seconds'],
                                  panel_difference=difference, repeat_difference=repeat_error, equivalence_tolerance=1e-10,
                                  target_trials_per_second=16*len(self.manifest['panel'])/records[1]['seconds'], memory=memory))
            except Exception as exc:
                complete = False; error = repr(exc)
                cases.append(dict(capacity=capacity, passed=False, error=error))
            io.atomic_json(folder/f'benchmark-{capacity:02d}.json', cases[-1])
        if reference is None or not cases[0]['passed']:
            raise ValueError('Four-slot benchmark reference did not pass')
        selected = choose_capacity(cases)
        io.atomic_npz(folder/'response.npz', response=reference)
        row = dict(status='completed', id=task['id'], task=task, safe=True, chosen_capacity=selected,
                   cases=cases, elapsed_seconds=time.monotonic()-tick, response_sha256=file_sha256(folder/'response.npz'),
                   scope='Measured identical-state full-panel batch equivalence and warm throughput; not a biological or full32-cycle qualification')
        io.atomic_json(folder/'result.json', row)
        return row


def animal_scores(prediction, parent, panel, ids, fold):
    rows = []
    for subject, statistics in sorted(fold.get('animal_statistics', {}).items()):
        try:
            data = dict(stats=dict(animal=statistics))
            row = dict(subject=subject, loss=loss(prediction, panel, ids, data, 'animal'),
                       parent_loss=loss(parent, panel, ids, data, 'animal'), unavailable=None)
            if not np.isfinite(row['loss']) or not np.isfinite(row['parent_loss']):
                raise ValueError('Nonfinite animal score')
        except ValueError as exc:
            row = dict(subject=subject, loss=None, parent_loss=None, unavailable=str(exc))
        rows.append(row)
    return rows


def choose_capacity(cases):
    valid = [c for c in cases if c.get('passed') and np.isfinite(c.get('warm_seconds', np.nan)) and c['warm_seconds'] > 0]
    if not valid:
        raise ValueError('No passing measured batch capacity')
    return min(valid, key=lambda c: c['warm_seconds'])['capacity']
