"""E104: all-parameter gradient fit to the Randi 2023 stimulation atlas with a separate validation split.

Trainable (`--parameter-set all`): every positive channel density of every neuron (including RMD and LEAK), every anatomical
chemical and gap edge strength, and per-cell capacitance, leak reversal, release threshold, release slope, calcium removal
time and free calcium fraction (offsets from the anchor, within the bounds in r_elegans.brain.all_parameters). Fixed: anatomy,
edge sign/reversal, gating kinetics, ion reversals, cell volume, calcium equilibrium, bias, initial voltage, the stimulus.
`--parameter-set network` trains the E101/E102 set under the same protocol as a control. Animals are split into train,
validation (every `--validation-every-nth` training animal) and the E100 test animals; validation selects the kept
checkpoint, test animals are only evaluated. Fits a population-level propagation statistic, not behavior.
"""
import argparse
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

from r_elegans.assets import load_asset_document
from r_elegans.atlas_evaluation import model_pairs, pair_metrics, pair_statistics, self_response_fraction, split_subjects
from r_elegans.brain.all_parameters import (
    AllParams, EXTRA_SCALE, ExtraOffsets, apply_extra, fixed_fields_equal_except_extra, offset_bounds, offsets_between, to_physical)
from r_elegans.brain.conductance_network import (
    MembraneConductanceInput, advance_network, initial_network_state, load_network, save_network)
from r_elegans.brain.fit_constraints import UnknownMasks, conservative_fit_constraints, frozen_parameter_changes, project_unknowns

try:
    from .experiment_records import capture_run_record
    from .run_atlas_baseline import build_statistics
    from .train_atlas_gradient import (
        CHANNEL_BOUND, DT_S, EDGE_BOUND, FRAMES, REVERSAL_MV, STIM_S, WINDOW_S, correlation_loss, dense_pairs, sha256)
except ImportError:
    from experiment_records import capture_run_record
    from run_atlas_baseline import build_statistics
    from train_atlas_gradient import (
        CHANNEL_BOUND, DT_S, EDGE_BOUND, FRAMES, REVERSAL_MV, STIM_S, WINDOW_S, correlation_loss, dense_pairs, sha256)

warnings.filterwarnings('ignore', category=RuntimeWarning)
SCOPE = ('Population-level gradient fit to simulated single-neuron stimulation responses; validation animals select the kept '
         'checkpoint, test animals are only evaluated. No behavior, locomotion, start/stop or biological-fidelity claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', type=Path, required=True, help='starting checkpoint')
    parser.add_argument('--anchor-network', type=Path, default=None, help='defines bounds, prior and frozen fields (default: --network)')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--experiment-id', default='E104')
    parser.add_argument('--parameter-set', choices=('all', 'network'), default='all')
    parser.add_argument('--minutes', type=float, default=180.)
    parser.add_argument('--final-reserve-minutes', type=float, default=10.)
    parser.add_argument('--stimulus-nS', type=float, default=.011182781160944818, help='E100 train-selected added conductance')
    parser.add_argument('--learning-rate', type=float, default=.005)
    parser.add_argument('--pair-fraction', type=float, default=.5)
    parser.add_argument('--resettle-every', type=int, default=10)
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--checkpoint-every', type=int, default=50)
    parser.add_argument('--max-steps', type=int, default=10**9)
    parser.add_argument('--max-targets', type=int, default=None, help='dry runs only')
    parser.add_argument('--substeps', type=int, default=40)
    parser.add_argument('--clip-z', type=float, default=5.)
    parser.add_argument('--validation-every-nth', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not (0 < args.pair_fraction <= 1 and args.learning_rate > 0 and args.minutes > 0 and args.resettle_every >= 1
            and args.validation_every_nth >= 2):
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
    cast = lambda a: a.astype(float_type) if jnp.issubdtype(a.dtype, jnp.floating) else a
    begin_params, begin_fixed, config = load_network(args.network)
    anchor, anchor_fixed, _ = load_network(args.anchor_network or args.network)
    begin_params, begin_fixed, anchor, anchor_fixed = jax.tree_util.tree_map(cast, (begin_params, begin_fixed, anchor, anchor_fixed))
    if not fixed_fields_equal_except_extra(begin_fixed, anchor_fixed):
        parser.error('Start and anchor checkpoints may differ only in the per-cell extra fields')
    fixed = anchor_fixed
    u0 = ExtraOffsets(*(jnp.asarray(o / s, float_type) for o, s in zip(offsets_between(begin_fixed, anchor_fixed), EXTRA_SCALE)))
    lower, upper = offset_bounds(anchor_fixed)
    u_lo = ExtraOffsets(*(jnp.asarray(x / s, float_type) for x, s in zip(lower, EXTRA_SCALE)))
    u_hi = ExtraOffsets(*(jnp.asarray(x / s, float_type) for x, s in zip(upper, EXTRA_SCALE)))
    train_extra = 1. if args.parameter_set == 'all' else 0.

    animals, trials, _, test_stats, train_subjects, test_subjects, _, n_test = build_statistics(
        args.data_root / 'raw/functional/randi2023_dandi001075_segmentation', neuron_ids, args.seed)
    fit_subjects, val_subjects = split_subjects(train_subjects, every=args.validation_every_nth)
    fit_trials = [t for t in trials if t['subject'] in set(fit_subjects)]
    val_trials = [t for t in trials if t['subject'] in set(val_subjects)]
    split_stats = dict(train=pair_statistics(fit_trials), val=pair_statistics(val_trials), test=test_stats)
    targets = sorted({t['target'] for t in trials})
    if args.max_targets:
        targets = targets[:args.max_targets]
    target_index = np.array([neuron_ids.index(t) for t in targets])
    z_train, w_train = dense_pairs(split_stats['train'], targets, neuron_ids, args.clip_z)
    data_summary = dict(animals=animals, successful_trials=len(trials), fit_trials=len(fit_trials), validation_trials=len(val_trials),
                        test_trials=n_test, fit_animals=len(fit_subjects), validation_animals=len(val_subjects),
                        test_animals=len(test_subjects), targets=len(targets),
                        fit_pairs=len(split_stats['train']['pairs']), validation_pairs=len(split_stats['val']['pairs']),
                        test_pairs=len(test_stats['pairs']), scored_fit_pairs=int(w_train.sum()), z_clip=args.clip_z)
    print(json.dumps(data_summary), flush=True)

    devices = jax.devices()
    padded = int(math.ceil(len(targets) / len(devices)) * len(devices))
    onehot = np.zeros((padded, 302))
    onehot[np.arange(len(targets)), target_index] = args.stimulus_nS
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    stimulus = jax.device_put(jnp.asarray(onehot, float_type), NamedSharding(Mesh(np.array(devices), ('b',)), P('b')))
    lo, hi = (int(round(w / DT_S)) for w in WINDOW_S)
    stim_frames = int(round(STIM_S / DT_S))

    if args.parameter_set == 'all':
        masks = UnknownMasks(jnp.asarray(np.asarray(fixed.cells.conductances_nS) > 0), jnp.asarray(np.asarray(fixed.chemical_mask) > 0),
                             jnp.asarray(np.asarray(fixed.gap_mask) > 0))
        policy = dict(version='all-parameters-v1', note='all positive channel densities, all anatomical edges, six per-cell extra fields',
                      fixed=['gating kinetics (code constants)', 'ion reversals', 'cell volume', 'calcium equilibrium', 'bias', 'initial voltage',
                             'anatomy', 'edge sign/reversal', 'stimulus conductance'])
    else:
        masks, policy = conservative_fit_constraints(fixed, neuron_ids, neuron_ids)
    (output / 'fit_policy.json').write_text(json.dumps(policy, indent=2, default=str) + '\n')
    np.savez_compressed(output / 'trainable_masks.npz', channels=np.asarray(masks.channels), chemical=np.asarray(masks.chemical),
                        gap=np.asarray(masks.gap))

    def make_responses(substeps):
        def run(p, fixed_, state0, conductance):
            def step(state, k):
                on = (k < stim_frames).astype(conductance.dtype)
                membrane = MembraneConductanceInput(conductance * on, conductance * on * REVERSAL_MV)
                state = advance_network(state, p, fixed_, DT_S, substeps, config, membrane_input=membrane)
                return state, state.indicator
            return jax.lax.scan(jax.checkpoint(step), state0, jnp.arange(FRAMES))[1]

        def responses(p, fixed_, state0, conductances):
            control = run(p, fixed_, state0, jnp.zeros(302, float_type))
            stimulated = jax.vmap(lambda g: run(p, fixed_, state0, g))(conductances)
            return (stimulated - control[None])[:, lo:hi].mean(axis=1)
        return responses

    def make_settled(substeps):
        @jax.jit
        def settled(p, fixed_):
            step = lambda s, _: (advance_network(s, p, fixed_, DT_S, substeps, config), None)
            return jax.lax.scan(step, initial_network_state(fixed_, config), None, length=int(round(60. / DT_S)))[0]
        return settled

    responses, settled = make_responses(args.substeps), make_settled(args.substeps)
    physical = lambda u: to_physical(jax.tree.map(lambda x: x * train_extra, u))
    fixed_of = lambda theta: apply_extra(physical(theta.extra), fixed)

    def objective(theta, state0, conductances, z, w):
        q = project_unknowns(theta.network, anchor, masks)
        response = responses(q, apply_extra(physical(theta.extra), fixed), state0, conductances)[:len(targets)]
        loss, (pooled, per_target) = correlation_loss(response, z, w)
        prior = jnp.sum(((q.log_channel_scale - anchor.log_channel_scale) * masks.channels) ** 2) / jnp.maximum(masks.channels.sum(), 1.)
        for value, start, mask in ((q.raw_chemical, anchor.raw_chemical, masks.chemical), (q.raw_gap, anchor.raw_gap, masks.gap)):
            prior += jnp.sum(((value - start) * mask) ** 2) / jnp.maximum(mask.sum(), 1.)
        for u, a, b in zip(theta.extra, u_lo, u_hi):
            prior += jnp.mean((u * train_extra / jnp.maximum(b - a, 1e-6)) ** 2)
        return loss + .001 * prior, (loss, pooled, per_target, response)

    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(args.learning_rate))

    @jax.jit
    def update(theta, opt_state, state0, conductances, z, w):
        (_, (loss, pooled, per_target, response)), grad = jax.value_and_grad(objective, has_aux=True)(theta, state0, conductances, z, w)
        grad = AllParams(project_unknowns(grad.network, jax.tree.map(jnp.zeros_like, grad.network), masks),
                         jax.tree.map(lambda g: g * train_extra, grad.extra))
        norms = [jnp.linalg.norm(g) for g in (*grad.network, *grad.extra)]
        finite = jnp.all(jnp.asarray([jnp.isfinite(g).all() for g in (*grad.network, *grad.extra)]))
        updates, opt_state = optimizer.update(grad, opt_state, theta)
        new = optax.apply_updates(theta, updates)
        net = new.network._replace(
            log_channel_scale=jnp.clip(new.network.log_channel_scale, anchor.log_channel_scale - CHANNEL_BOUND, anchor.log_channel_scale + CHANNEL_BOUND),
            raw_chemical=jnp.where(fixed.chemical_mask, jnp.clip(new.network.raw_chemical, anchor.raw_chemical - EDGE_BOUND, anchor.raw_chemical + EDGE_BOUND), anchor.raw_chemical),
            raw_gap=jnp.where(fixed.gap_mask, jnp.clip(new.network.raw_gap, anchor.raw_gap - EDGE_BOUND, anchor.raw_gap + EDGE_BOUND), anchor.raw_gap))
        extra = ExtraOffsets(*(jnp.clip(u, a, b) for u, a, b in zip(new.extra, u_lo, u_hi)))
        return AllParams(project_unknowns(net, anchor, masks), extra), opt_state, loss, pooled, per_target, response, norms, finite

    def save(path, theta):
        save_network(path, project_unknowns(theta.network, anchor, masks), fixed_of(theta), config)

    def evaluate(response, split, permutations):
        stats = split_stats[split]
        result = pair_metrics(model_pairs(np.asarray(response), targets, neuron_ids, stats), stats, seed=args.seed, permutations=permutations)
        return result if permutations else {k: v for k, v in result.items() if not k.startswith('permutation')}

    def snapshot(response, permutations):
        return dict(train=evaluate(response, 'train', permutations), val=evaluate(response, 'val', permutations),
                    test=evaluate(response, 'test', permutations), self_response_fraction=self_response_fraction(np.asarray(response), targets, neuron_ids))

    rng = np.random.default_rng(args.seed)
    theta = AllParams(begin_params, u0)
    opt_state = optimizer.init(theta)
    history = (output / 'history.jsonl').open('w')
    evaluations, best_val = [], dict(score=-math.inf)
    status, stop_reason, state0, step, initial_response = 'completed', 'max_steps', None, 0, None
    z_all, w_all = jnp.asarray(z_train, float_type), jnp.asarray(w_train, float_type)
    while step < args.max_steps:
        if step and time.monotonic() > deadline:
            stop_reason = 'deadline'
            break
        resettled = step % args.resettle_every == 0
        if resettled:
            state0 = jax.lax.stop_gradient(settled(project_unknowns(theta.network, anchor, masks), fixed_of(theta)))
        w_step = w_all * jnp.asarray(rng.random(w_all.shape) < args.pair_fraction, float_type)
        new, new_opt, loss, pooled, per_target, response, norms, finite = update(theta, opt_state, state0, stimulus, z_all, w_step)
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
            score = evaluations[-1]['val'].get('spearman_mean_z')
            if score is not None and math.isfinite(score) and score > best_val['score']:
                best_val = dict(score=score, step=step, theta=theta)
        record = dict(step=step, elapsed_s=round(time.monotonic() - started, 2), loss=loss, pooled_corr=float(pooled),
                      per_target_corr=float(per_target), grad_norms=[float(n) for n in norms], resettled=resettled)
        history.write(json.dumps(record) + '\n')
        history.flush()
        if step % 5 == 0:
            print(json.dumps(record), flush=True)
        theta, opt_state, step = new, new_opt, step + 1
        if step % args.checkpoint_every == 0:
            save(output / 'network_latest.npz', theta)
            (output / 'optimizer_latest.msgpack').write_bytes(serialization.to_bytes(jax.device_get(opt_state)))
            if 'theta' in best_val:
                save(output / 'network_best_val.npz', best_val['theta'])
    history.close()

    jit_responses = jax.jit(responses)
    final_fixed = fixed_of(theta)
    final_response = np.asarray(jit_responses(project_unknowns(theta.network, anchor, masks), final_fixed,
                                              jax.lax.stop_gradient(settled(project_unknowns(theta.network, anchor, masks), final_fixed)), stimulus))[:len(targets)]
    best_theta = best_val.get('theta', theta)
    best_fixed = fixed_of(best_theta)
    best_response = np.asarray(jit_responses(project_unknowns(best_theta.network, anchor, masks), best_fixed,
                                             jax.lax.stop_gradient(settled(project_unknowns(best_theta.network, anchor, masks), best_fixed)), stimulus))[:len(targets)]
    convergence_targets = min(13, len(targets))
    final_net = project_unknowns(theta.network, anchor, masks)
    fine = np.asarray(jax.jit(make_responses(2 * args.substeps))(
        final_net, final_fixed, make_settled(2 * args.substeps)(final_net, final_fixed), stimulus[:convergence_targets])) if convergence_targets else None
    coarse = final_response[:convergence_targets]
    save(output / 'network_final.npz', theta)
    save(output / 'network_best_val.npz', best_theta)
    (output / 'optimizer_final.msgpack').write_bytes(serialization.to_bytes(jax.device_get(opt_state)))
    np.savez_compressed(output / 'responses.npz', targets=np.array(targets), initial=initial_response if initial_response is not None else np.zeros(0),
                        final=final_response, best_val=best_response, stimulus_nS=args.stimulus_nS)
    net_delta = jax.tree.map(lambda a, b: np.asarray(a) - np.asarray(b), project_unknowns(theta.network, anchor, masks), anchor)
    extra_physical = physical(theta.extra)
    at_bound = lambda u, a, b: int(np.sum((np.asarray(u) <= np.asarray(a) + 1e-9) | (np.asarray(u) >= np.asarray(b) - 1e-9)) - np.sum((np.asarray(a) == 0) & (np.asarray(b) == 0)))
    summary = dict(
        experiment=args.experiment_id, scope=SCOPE, parameter_set=args.parameter_set, status=status, stop_reason=stop_reason, steps=step,
        elapsed_minutes=(time.monotonic() - started) / 60., budget_minutes=args.minutes, deadline_reached=stop_reason == 'deadline',
        parent_network=str(args.network.resolve()), parent_sha256=sha256(args.network),
        anchor_network=str((args.anchor_network or args.network).resolve()), anchor_sha256=sha256(args.anchor_network or args.network),
        optimizer_start='fresh (not a mid-run state resume)', final_sha256=sha256(output / 'network_final.npz'),
        best_val_sha256=sha256(output / 'network_best_val.npz'), best_val=dict(spearman=best_val['score'] if 'theta' in best_val else None, step=best_val.get('step')),
        devices=[str(d) for d in devices], data=data_summary, stimulus_nS=args.stimulus_nS, learning_rate=args.learning_rate,
        pair_fraction=args.pair_fraction, resettle_every=args.resettle_every, substeps=args.substeps,
        trainable=dict(channels=int(masks.channels.sum()), chemical=int(masks.chemical.sum()), gap=int(masks.gap.sum()),
                       extra_per_field=0 if not train_extra else 302, extra_fields=list(ExtraOffsets._fields)),
        changed_entries={n: int(np.count_nonzero(d)) for n, d in zip(('log_channel_scale', 'raw_chemical', 'raw_gap'), net_delta)},
        frozen_parameter_changes=frozen_parameter_changes(project_unknowns(theta.network, anchor, masks), anchor, masks),
        at_clip_bound=dict(
            channels=int(np.sum((np.abs(net_delta.log_channel_scale) >= CHANNEL_BOUND - 1e-6) & np.asarray(masks.channels))),
            chemical=int(np.sum((np.abs(net_delta.raw_chemical) >= EDGE_BOUND - 1e-6) & np.asarray(masks.chemical))),
            gap=int(np.sum((np.abs(net_delta.raw_gap) >= EDGE_BOUND - 1e-6) & np.asarray(masks.gap))),
            extra={f: at_bound(u, a, b) for f, u, a, b in zip(ExtraOffsets._fields, theta.extra, u_lo, u_hi)} if train_extra else None),
        max_abs_change=dict(zip(('log_channel_scale', 'raw_chemical', 'raw_gap'), [float(np.abs(d).max()) for d in net_delta])),
        extra_max_abs_change={f: float(np.abs(np.asarray(v)).max()) for f, v in zip(ExtraOffsets._fields, extra_physical)},
        evaluations=evaluations, final=snapshot(final_response, 500), best_val_checkpoint=snapshot(best_response, 500),
        substep_convergence=None if fine is None else dict(
            targets=convergence_targets, correlation=float(np.corrcoef(fine.ravel(), coarse.ravel())[0, 1]),
            max_abs_difference_over_max_abs_response=float(np.abs(fine - coarse).max() / max(np.abs(coarse).max(), 1e-30))),
        selection_rule='best validation-animal Spearman over evaluation steps (network_best_val); final also kept; test animals never used for selection',
        unavailable=['body/behavior responses of the trained network', 'sensory start/stop control', 'trajectory (non-pooled) calcium fit',
                     'amplitude calibration of the calcium indicator', 'gating kinetics were not trainable'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    print(json.dumps(dict(status=status, steps=step, final=summary['final'], best_val_checkpoint=summary['best_val_checkpoint']), default=float), flush=True)
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
