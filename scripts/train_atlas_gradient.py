"""E101: gradient fit of all-cell channel densities and synapse strengths to the Randi 2023 stimulation atlas.

Starts from a preserved conductance checkpoint. One optimizer step simulates every atlas target neuron (one stimulus
each, fixed added conductance, same window and z-score definitions as E100), and maximizes the Pearson correlation
between the simulated response and the pooled training-animal mean z of each (target, responder) pair. Held-out
animals are only evaluated, never used to select or stop. This fits a population-level propagation statistic; it is
not a trajectory fit and makes no locomotion, start/stop or biological-fidelity claim.
"""
import argparse
import hashlib
import json
import math
import shutil
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from r_elegans.assets import load_asset_document
from r_elegans.atlas_evaluation import model_pairs, pair_metrics, self_response_fraction
from r_elegans.brain.conductance_network import (
    MembraneConductanceInput, advance_network, initial_network_state, load_network, save_network)
from r_elegans.brain.fit_constraints import conservative_fit_constraints, frozen_parameter_changes, project_unknowns

try:
    from .experiment_records import capture_run_record
    from .run_atlas_baseline import build_statistics
except ImportError:
    from experiment_records import capture_run_record
    from run_atlas_baseline import build_statistics

warnings.filterwarnings('ignore', category=RuntimeWarning)
SCOPE = ('Population-level gradient fit to simulated single-neuron stimulation responses; held-out animals evaluated only. '
         'No behavior, locomotion, start/stop or biological-fidelity claim.')
DT_S, FRAMES, STIM_S, WINDOW_S, REVERSAL_MV = .1, 120, .5, (1.5, 11.5), 0.
CHANNEL_BOUND, EDGE_BOUND = math.log(10.), 2.


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def dense_pairs(stats, targets, neuron_ids, clip):
    row, column = {t: i for i, t in enumerate(targets)}, {n: i for i, n in enumerate(neuron_ids)}
    z, w = np.zeros((len(targets), len(neuron_ids))), np.zeros((len(targets), len(neuron_ids)))
    for (target, responder), value in zip(stats['pairs'], stats['mean_z']):
        if target in row and responder in column:
            z[row[target], column[responder]] = np.clip(value, -clip, clip)
            w[row[target], column[responder]] = 1.
    return z, w


def correlation_loss(response, z, w):
    """1 - (pooled Pearson + mean per-target Pearson)/2 over the weighted pair entries."""
    n = jnp.maximum(w.sum(), 1.)
    a, b = (response - (response * w).sum() / n) * w, (z - (z * w).sum() / n) * w
    pooled = (a * b).sum() / jnp.sqrt((a * a).sum() * (b * b).sum() + 1e-30)
    count = jnp.maximum(w.sum(axis=1), 1.)
    a_t = (response - ((response * w).sum(axis=1) / count)[:, None]) * w
    b_t = (z - ((z * w).sum(axis=1) / count)[:, None]) * w
    per_target = (a_t * b_t).sum(axis=1) / jnp.sqrt((a_t * a_t).sum(axis=1) * (b_t * b_t).sum(axis=1) + 1e-30)
    valid = (w.sum(axis=1) >= 5).astype(response.dtype)
    target_mean = (per_target * valid).sum() / jnp.maximum(valid.sum(), 1.)
    return 1. - .5 * (pooled + target_mean), (pooled, target_mean)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', type=Path, required=True, help='starting checkpoint')
    parser.add_argument('--anchor-network', type=Path, default=None,
                        help='checkpoint that defines clip bounds, the prior and frozen fields (default: --network); use the original bundle when continuing')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--experiment-id', default='E101')
    parser.add_argument('--minutes', type=float, default=60., help='total wall budget including data loading and final evaluation')
    parser.add_argument('--final-reserve-minutes', type=float, default=5.)
    parser.add_argument('--stimulus-nS', type=float, default=.011182781160944818, help='E100 train-selected added conductance')
    parser.add_argument('--learning-rate', type=float, default=.005)
    parser.add_argument('--pair-fraction', type=float, default=.5, help='random fraction of training pairs scored each step')
    parser.add_argument('--resettle-every', type=int, default=10)
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--checkpoint-every', type=int, default=50)
    parser.add_argument('--snapshot-every', type=int, default=0, help='also keep numbered network snapshots (0 = off)')
    parser.add_argument('--max-steps', type=int, default=10**9)
    parser.add_argument('--max-targets', type=int, default=None, help='dry runs only')
    parser.add_argument('--substeps', type=int, default=20)
    parser.add_argument('--clip-z', type=float, default=5.)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not (0 < args.pair_fraction <= 1 and args.learning_rate > 0 and args.minutes > 0 and args.resettle_every >= 1):
        parser.error('Invalid training settings')
    started = time.monotonic()
    deadline = started + (args.minutes - args.final_reserve_minutes) * 60.
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.expanduser().resolve()
    if output == repo or repo in output.parents:
        parser.error('Keep artifacts external')
    output.mkdir(parents=True, exist_ok=False)
    capture_run_record(output, repo, args)
    shutil.copy2(args.network, output / 'parent_network.npz')
    if args.anchor_network:
        shutil.copy2(args.anchor_network, output / 'anchor_network.npz')

    neuron_ids = list(load_asset_document('runtime_model_v1.json')['neuron_ids'])
    float_type = jnp.result_type(float)
    begin, fixed, config = load_network(args.network)
    cast = lambda a: a.astype(float_type) if jnp.issubdtype(a.dtype, jnp.floating) else a
    parent, anchor_fixed, _ = load_network(args.anchor_network or args.network)
    begin, fixed, parent, anchor_fixed = jax.tree_util.tree_map(cast, (begin, fixed, parent, anchor_fixed))
    if not all(np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(fixed), jax.tree_util.tree_leaves(anchor_fixed))):
        parser.error('Start and anchor checkpoints must have identical fixed fields')
    animals, trials, train_stats, test_stats, train_subjects, test_subjects, n_train, n_test = build_statistics(
        args.data_root / 'raw/functional/randi2023_dandi001075_segmentation', neuron_ids, args.seed)
    targets = sorted({t['target'] for t in trials})
    if args.max_targets:
        targets = targets[:args.max_targets]
    target_index = np.array([neuron_ids.index(t) for t in targets])
    z_train, w_train = dense_pairs(train_stats, targets, neuron_ids, args.clip_z)
    data_summary = dict(animals=animals, successful_trials=len(trials), train_trials=n_train, test_trials=n_test,
                        train_animals=len(train_subjects), test_animals=len(test_subjects), targets=len(targets),
                        train_pairs=len(train_stats['pairs']), test_pairs=len(test_stats['pairs']),
                        scored_train_pairs=int(w_train.sum()), z_clip=args.clip_z)
    print(json.dumps(data_summary), flush=True)

    devices = jax.devices()
    padded = int(math.ceil(len(targets) / len(devices)) * len(devices))
    onehot = np.zeros((padded, 302))
    onehot[np.arange(len(targets)), target_index] = args.stimulus_nS
    sharding = NamedSharding(Mesh(np.array(devices), ('b',)), P('b'))
    stimulus = jax.device_put(jnp.asarray(onehot, float_type), sharding)
    lo, hi = (int(round(w / DT_S)) for w in WINDOW_S)
    stim_frames = int(round(STIM_S / DT_S))

    masks, policy = conservative_fit_constraints(fixed, neuron_ids, neuron_ids)
    (output / 'fit_policy.json').write_text(json.dumps(policy, indent=2) + '\n')
    np.savez_compressed(output / 'trainable_masks.npz', channels=np.asarray(masks.channels),
                        chemical=np.asarray(masks.chemical), gap=np.asarray(masks.gap))

    def make_responses(substeps):
        def run(p, state0, conductance):
            def step(state, k):
                on = (k < stim_frames).astype(conductance.dtype)
                membrane = MembraneConductanceInput(conductance * on, conductance * on * REVERSAL_MV)
                state = advance_network(state, p, fixed, DT_S, substeps, config, membrane_input=membrane)
                return state, state.indicator
            return jax.lax.scan(jax.checkpoint(step), state0, jnp.arange(FRAMES))[1]

        def responses(p, state0, conductances):
            control = run(p, state0, jnp.zeros(302, float_type))
            stimulated = jax.vmap(lambda g: run(p, state0, g))(conductances)
            return (stimulated - control[None])[:, lo:hi].mean(axis=1)
        return responses

    responses = make_responses(args.substeps)

    def make_settled(substeps):
        @jax.jit
        def settled(p):
            step = lambda s, _: (advance_network(s, p, fixed, DT_S, substeps, config), None)
            return jax.lax.scan(step, initial_network_state(fixed, config), None, length=int(round(60. / DT_S)))[0]
        return settled

    settled = make_settled(args.substeps)

    def objective(p, state0, conductances, z, w):
        q = project_unknowns(p, parent, masks)
        response = responses(q, state0, conductances)[:len(targets)]
        loss, (pooled, per_target) = correlation_loss(response, z, w)
        prior = jnp.sum(((q.log_channel_scale - parent.log_channel_scale) * masks.channels) ** 2) / jnp.maximum(masks.channels.sum(), 1.)
        for value, start, mask in ((q.raw_chemical, parent.raw_chemical, masks.chemical), (q.raw_gap, parent.raw_gap, masks.gap)):
            prior += jnp.sum(((value - start) * mask) ** 2) / jnp.maximum(mask.sum(), 1.)
        return loss + .001 * prior, (loss, pooled, per_target, response)

    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(args.learning_rate))

    @jax.jit
    def update(p, opt_state, state0, conductances, z, w):
        (_, (loss, pooled, per_target, response)), gradient = jax.value_and_grad(objective, has_aux=True)(p, state0, conductances, z, w)
        gradient = project_unknowns(gradient, jax.tree.map(jnp.zeros_like, gradient), masks)
        norms = [jnp.linalg.norm(g) for g in gradient]
        finite = jnp.all(jnp.asarray([jnp.isfinite(g).all() for g in gradient]))
        updates, opt_state = optimizer.update(gradient, opt_state, p)
        new = optax.apply_updates(p, updates)
        new = new._replace(
            log_channel_scale=jnp.clip(new.log_channel_scale, parent.log_channel_scale - CHANNEL_BOUND, parent.log_channel_scale + CHANNEL_BOUND),
            raw_chemical=jnp.where(fixed.chemical_mask, jnp.clip(new.raw_chemical, parent.raw_chemical - EDGE_BOUND, parent.raw_chemical + EDGE_BOUND), parent.raw_chemical),
            raw_gap=jnp.where(fixed.gap_mask, jnp.clip(new.raw_gap, parent.raw_gap - EDGE_BOUND, parent.raw_gap + EDGE_BOUND), parent.raw_gap))
        return project_unknowns(new, parent, masks), opt_state, loss, pooled, per_target, response, norms, finite

    def evaluate(response, split, permutations):
        stats = train_stats if split == 'train' else test_stats
        result = pair_metrics(model_pairs(np.asarray(response), targets, neuron_ids, stats), stats, seed=args.seed, permutations=permutations)
        if permutations == 0:
            result = {k: v for k, v in result.items() if not k.startswith('permutation')}
        return result

    def snapshot(response, permutations):
        return dict(train=evaluate(response, 'train', permutations), test=evaluate(response, 'test', permutations),
                    self_response_fraction=self_response_fraction(np.asarray(response), targets, neuron_ids))

    rng = np.random.default_rng(args.seed)
    params, opt_state = begin, optimizer.init(begin)
    history = (output / 'history.jsonl').open('w')
    evaluations, best = [], dict(loss=math.inf)
    status, stop_reason, state0, step = 'completed', 'max_steps', None, 0
    initial_response = None
    z_all, w_all = jnp.asarray(z_train, float_type), jnp.asarray(w_train, float_type)
    while step < args.max_steps:
        if step and time.monotonic() > deadline:
            stop_reason = 'deadline'
            break
        resettled = step % args.resettle_every == 0
        if resettled:
            state0 = jax.lax.stop_gradient(settled(project_unknowns(params, parent, masks)))
        w_step = w_all * jnp.asarray(rng.random(w_all.shape) < args.pair_fraction, float_type)
        new, new_opt, loss, pooled, per_target, response, norms, finite = update(params, opt_state, state0, stimulus, z_all, w_step)
        loss = float(loss)
        if not (math.isfinite(loss) and bool(finite)):
            status = 'aborted_nonfinite_gradient_or_loss'
            history.write(json.dumps(dict(step=step, loss=loss, finite=bool(finite))) + '\n')
            break
        if step == 0:
            initial_response = np.asarray(response)
        if step % args.eval_every == 0:
            evaluations.append(dict(step=step, elapsed_s=time.monotonic() - started, **snapshot(response, 0)))
            print(json.dumps(evaluations[-1], default=float), flush=True)
        if loss < best['loss']:
            best = dict(loss=loss, step=step, params=params)
        record = dict(step=step, elapsed_s=round(time.monotonic() - started, 2), loss=loss, pooled_corr=float(pooled),
                      per_target_corr=float(per_target), grad_norms=[float(n) for n in norms], resettled=resettled)
        history.write(json.dumps(record) + '\n')
        history.flush()
        if step % 5 == 0:
            print(json.dumps(record), flush=True)
        params, opt_state, step = new, new_opt, step + 1
        if args.snapshot_every and step % args.snapshot_every == 0:
            save_network(output / f'network_step{step:06d}.npz', params, fixed, config)
        if step % args.checkpoint_every == 0:
            save_network(output / 'network_latest.npz', params, fixed, config)
            (output / 'optimizer_latest.msgpack').write_bytes(serialization.to_bytes(jax.device_get(opt_state)))
    history.close()

    final_state0 = jax.lax.stop_gradient(settled(params))
    final_response = np.asarray(jax.jit(responses)(params, final_state0, stimulus))[:len(targets)]
    convergence_targets = min(13, len(targets))
    fine = np.asarray(jax.jit(make_responses(2 * args.substeps))(
        params, make_settled(2 * args.substeps)(params), stimulus[:convergence_targets])) if convergence_targets else None
    coarse = final_response[:convergence_targets]
    save_network(output / 'network_final.npz', params, fixed, config)
    save_network(output / 'network_best_train_loss.npz', best.get('params', params), fixed, config)
    (output / 'optimizer_final.msgpack').write_bytes(serialization.to_bytes(jax.device_get(opt_state)))
    np.savez_compressed(output / 'responses.npz', targets=np.array(targets), initial=initial_response if initial_response is not None else np.zeros(0),
                        final=final_response, stimulus_nS=args.stimulus_nS)
    delta = jax.tree.map(lambda a, b: np.asarray(a) - np.asarray(b), params, parent)
    changed = {name: int(np.count_nonzero(d)) for name, d in zip(('log_channel_scale', 'raw_chemical', 'raw_gap'), delta)}
    bounds = dict(
        channels_at_bound=int(np.sum((np.abs(delta.log_channel_scale) >= CHANNEL_BOUND - 1e-6) & np.asarray(masks.channels))),
        chemical_at_bound=int(np.sum((np.abs(delta.raw_chemical) >= EDGE_BOUND - 1e-6) & np.asarray(masks.chemical))),
        gap_at_bound=int(np.sum((np.abs(delta.raw_gap) >= EDGE_BOUND - 1e-6) & np.asarray(masks.gap))))
    final_snapshot = snapshot(final_response, 500)
    summary = dict(
        experiment=args.experiment_id, scope=SCOPE, status=status, stop_reason=stop_reason, steps=step, elapsed_minutes=(time.monotonic() - started) / 60.,
        budget_minutes=args.minutes, deadline_reached=stop_reason == 'deadline',
        parent_network=str(args.network.resolve()), parent_sha256=sha256(args.network),
        anchor_network=str((args.anchor_network or args.network).resolve()), anchor_sha256=sha256(args.anchor_network or args.network),
        optimizer_start='fresh (not a mid-run state resume)',
        final_sha256=sha256(output / 'network_final.npz'), best_train_loss_sha256=sha256(output / 'network_best_train_loss.npz'),
        best_train_loss=dict(loss=best['loss'], step=best.get('step')), devices=[str(d) for d in devices],
        data=data_summary, stimulus_nS=args.stimulus_nS, learning_rate=args.learning_rate, pair_fraction=args.pair_fraction,
        resettle_every=args.resettle_every, trainable=dict(channels=int(masks.channels.sum()), chemical=int(masks.chemical.sum()),
                                                           gap=int(masks.gap.sum())),
        changed_entries=changed, frozen_parameter_changes=frozen_parameter_changes(params, parent, masks), at_clip_bound=bounds,
        max_abs_change=dict(zip(('log_channel_scale', 'raw_chemical', 'raw_gap'), [float(np.abs(d).max()) for d in delta])),
        evaluations=evaluations, final=final_snapshot,
        substep_convergence=None if fine is None else dict(
            targets=convergence_targets, correlation=float(np.corrcoef(fine.ravel(), coarse.ravel())[0, 1]),
            max_abs_difference_over_max_abs_response=float(np.abs(fine - coarse).max() / max(np.abs(coarse).max(), 1e-30))),
        selection_rule='final and best-train-loss checkpoints are both kept; held-out metrics are reported, never used for selection',
        unavailable=['body/behavior responses of the trained network', 'sensory start/stop control', 'trajectory (non-pooled) calcium fit',
                     'amplitude calibration of the calcium indicator'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    print(json.dumps(dict(status=status, steps=step, final=final_snapshot), default=float), flush=True)
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
