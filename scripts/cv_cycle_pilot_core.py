"""E129 train-only structured fitting and compact midpoint/cycle evaluations."""
import json
from pathlib import Path
import time
import numpy as np
from scripts import run_full_cycle_validation as validated
from scripts.experiment_records import file_sha256
from scripts.stable_atlas_policy import Coordinates, eligible_targets, loss
from scripts.cv_fit_core import group_directions, raw_peptide_gains, penalty

io = validated.resume
PARENT_SHA = 'f626ded0494902820432046f1031cafe35b65216cb25ecc6ffb1cee2d829e652'
GROUPS = ('inward', 'outward', 'chemical', 'gap', 'peptide_balance')


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def panel_for(folds, source_targets, seed=1290, return_state=False):
    # Training eligibility only; no heldout scores/means enter target choice.
    common = set(source_targets)
    for fold in folds:
        common.intersection_update(eligible_targets(fold, 'train'))
    if len(common) < 4:
        raise ValueError('Four common training-eligible targets required')
    rng = np.random.default_rng(seed)
    panel = sorted(rng.choice(sorted(common), 4, replace=False).tolist())
    return (panel, rng.bit_generator.state) if return_state else panel


def decode(origin, coefficients):
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (5,) or not np.isfinite(coefficients).all() or abs(coefficients).max() > .1+1e-12:
        raise ValueError('Five bounded finite group coordinates required')
    codec = Coordinates(origin)
    directions = group_directions(codec)
    x = sum(coefficients[i]*directions[name] for i, name in enumerate(GROUPS[:4]))
    result = codec.apply(x)
    exc, inh = raw_peptide_gains(coefficients[4])
    result['peptide_raw_exc'] = np.full_like(origin['peptide_raw_exc'], exc)
    result['peptide_raw_inh'] = np.full_like(origin['peptide_raw_inh'], inh)
    return result, x


def parameter_audit(origin, actual, coefficients, starts=None):
    expected, x = decode(origin, coefficients)
    if starts is not None:
        for key, values in starts.items():
            expected['start_'+key] = np.asarray(values)[:4]
    errors = [key for key in expected if key not in actual or not np.array_equal(actual[key], expected[key])]
    errors += sorted(set(actual)-set(expected))
    return dict(passed=not errors, mismatched_fields=errors,
                changed_neural_coordinates=int(np.count_nonzero(x)),
                peptide_balance=float(coefficients[4]), maximum_neural_offset=float(abs(x).max()),
                fixed_parameter_fields_exact=not errors, near_bound=int(np.sum(abs(np.asarray(coefficients)) >= .099)))


def score(y, names, ids, fold, coefficients):
    _, x = decode(fold['_origin'], coefficients)
    return float(loss(y, names, ids, fold, 'train')+penalty(x, coefficients[4]))


def choose(current, proposals):
    valid = [current]+[r for r in proposals if r.get('safe') and np.isfinite(r.get('score', np.nan))]
    best = min(valid, key=lambda r: r['score'])
    return best if best['score'] < current['score']-1e-8 else current


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
        p, f, config, pp, exc, inh, _, origin = load_model(model)
        ids = origin['neuron_ids'].tolist()
        if level not in self.cache:
            self.cache[level] = (make_anchors(config, ids.index('RMDL'), dt, step),
                                 make_extension(config, ids.index('RMDL'), dt, step),
                                 make_replay(config, [ids.index('RMDL'), ids.index('RMDR')], dt, step, trace_mode='shams'))
        anchor, extend, replay = self.cache[level]
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
        pieces = []; checks = []; hashes = {}
        for group in range(8):
            path = folder/f'batch-{group:02d}.npz'
            record = path.with_suffix('.json')
            provenance = dict(task=task, group=group, starts_sha256=file_sha256(start_path))
            if path.exists():
                data = arrays(path)
                row = json.loads(str(data.pop('record_json')))
                if row['provenance'] != provenance or (record.exists() and json.loads(record.read_text())['sha256'] != file_sha256(path)):
                    raise ValueError('Saved response batch changed')
                means = data['means']
            else:
                io.atomic_json(self.root/f"progress-{self.manifest.get('worker_slot', 0)}.json", dict(task=task['id'], group=group, stage='response_batch'))
                stimuli = np.zeros((6, 302)); stimuli[np.arange(4), [ids.index(t) for t in self.manifest['panel']]] = .03
                initial = jax.tree.map(lambda v: v[group*16:(group+1)*16], starts)
                means, final, traces = jax.tree.map(np.asarray, replay(p, f, pp, exc, inh, initial, jnp.asarray(stimuli)))
                row = dict(provenance=provenance, checks=batch_metrics(means, final, traces))
                io.atomic_npz(path, means=means, record_json=np.array(json.dumps(row)))
                # Durable readback before the next expensive batch.
                data = arrays(path); means = data['means']
                if json.loads(str(data['record_json'])) != row:
                    raise ValueError('Batch readback failed')
            if means.shape != (16, 6, 302) or not np.isfinite(means).all():
                raise ValueError('Invalid compact response shape/values')
            io.atomic_json(record, dict(sha256=file_sha256(path), **row))
            hashes[path.name] = file_sha256(path)
            pieces.append(means[:, :4]-means[:, -1:]); checks.append(row['checks'])
        response = np.concatenate(pieces)
        sampling = check_response(response, meta['cycle_times'], self.manifest['sampling_tolerances'][str(level)])
        path = folder/'response.npz'
        io.atomic_npz(path, response=response)
        result = dict(id=task['id'], status='completed', task=task, sampling=sampling, batch_checks=checks,
                      safe=bool(sampling['passed'] and all(c['passed'] for c in checks)), elapsed_seconds=time.monotonic()-tick,
                      response_sha256=file_sha256(path), batch_sha256=hashes,
                      model_sha256=task['model_sha256'], fresh_own_states=True, cycle_times=meta['cycle_times'])
        io.atomic_json(folder/'result.json', result)
        return result
