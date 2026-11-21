"""E105: gradient fit of an extrasynaptic (peptide) signaling term, with or without the wired network, to the Randi 2023 atlas.

Adds the peptide-receptor term of r_elegans.brain.extrasynaptic (269 Beets EC50 <= 500 nM pairs x CeNGEN expression; separate
excitatory and inhibitory gains per pair, one shared time constant) to the single-neuron stimulation simulation used in E100-E104.
`--train-network yes` also trains the E101/E102 parameter set (channel densities, edge strengths); `no` freezes the network at the
starting checkpoint and trains only the peptide term. Same objective, data and split as E104 (fit / validation / E100 test
animals); validation selects the kept checkpoint, test animals are only evaluated. The final and best checkpoints are also
evaluated with the peptide term switched off. Wild-type atlas only: the unc-31 atlas is not used yet, so the peptide gains can
absorb any residual, not only peptide effects. The term acts in this stimulation task only; the core network step and the body
are unchanged.
"""
import argparse
import json
import math
import shutil
import time
import warnings
from importlib.resources import files
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from r_elegans.assets import load_asset_document
from r_elegans.atlas_evaluation import model_pairs, pair_metrics, pair_statistics, self_response_fraction, split_subjects
from r_elegans.brain.conductance_network import (
    MembraneConductanceInput, advance_network, initial_network_state, load_network, save_network)
from r_elegans.brain.extrasynaptic import (
    PeptideParams, clip_peptide_params, coupling_matrices, initial_peptide_params, load_extrasynaptic_asset, peptide_level_step,
    peptide_membrane_input)
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
SCOPE = ('Population-level gradient fit with an extrasynaptic peptide term to simulated single-neuron stimulation responses; '
         'validation animals select the kept checkpoint, test animals are only evaluated. No behavior, gait, start/stop or '
         'biological-fidelity claim; the peptide term is not part of the core network or the body.')


class Theta(NamedTuple):
    network: object
    pep: PeptideParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', type=Path, required=True, help='starting checkpoint')
    parser.add_argument('--anchor-network', type=Path, default=None, help='defines bounds and the prior when the network is trained')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--experiment-id', default='E105')
    parser.add_argument('--train-network', choices=('yes', 'no'), default='yes')
    parser.add_argument('--minutes', type=float, default=120.)
    parser.add_argument('--final-reserve-minutes', type=float, default=10.)
    parser.add_argument('--stimulus-nS', type=float, default=.011182781160944818, help='E100 train-selected added conductance')
    parser.add_argument('--learning-rate', type=float, default=.005)
    parser.add_argument('--pair-fraction', type=float, default=.5)
    parser.add_argument('--peptide-tau-s', type=float, default=5.)
    parser.add_argument('--peptide-prior', type=float, default=.01)
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
            and args.validation_every_nth >= 2 and 1. <= args.peptide_tau_s <= 30.):
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
    begin_params, fixed, config = load_network(args.network)
    anchor, anchor_fixed, _ = load_network(args.anchor_network or args.network)
    begin_params, fixed, anchor, anchor_fixed = jax.tree_util.tree_map(cast, (begin_params, fixed, anchor, anchor_fixed))
    if not all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(jax.tree_util.tree_leaves(fixed), jax.tree_util.tree_leaves(anchor_fixed))):
        parser.error('Start and anchor checkpoints must have identical fixed fields')
    train_network = args.train_network == 'yes'
    reference = anchor if train_network else begin_params

    asset = load_extrasynaptic_asset()
    if list(asset['neuron_ids']) != neuron_ids:
        parser.error('Peptide asset neuron order differs from the model')
    sender, receiver = jnp.asarray(asset['sender'], float_type), jnp.asarray(asset['receiver'], float_type)
    pairs = sender.shape[0]
    asset_sha = sha256(Path(str(files('r_elegans.assets').joinpath('extrasynaptic_v1.npz'))))

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
                        test_animals=len(test_subjects), targets=len(targets), fit_pairs=len(split_stats['train']['pairs']),
                        validation_pairs=len(split_stats['val']['pairs']), test_pairs=len(test_stats['pairs']),
                        scored_fit_pairs=int(w_train.sum()), z_clip=args.clip_z, peptide_pairs=int(pairs))
    print(json.dumps(data_summary), flush=True)

    devices = jax.devices()
    padded = int(math.ceil(len(targets) / len(devices)) * len(devices))
    onehot = np.zeros((padded, 302))
    onehot[np.arange(len(targets)), target_index] = args.stimulus_nS
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    stimulus = jax.device_put(jnp.asarray(onehot, float_type), NamedSharding(Mesh(np.array(devices), ('b',)), P('b')))
    lo, hi = (int(round(w / DT_S)) for w in WINDOW_S)
    stim_frames = int(round(STIM_S / DT_S))

    if train_network:
        masks, policy = conservative_fit_constraints(fixed, neuron_ids, neuron_ids)
    else:
        none = lambda shape: jnp.zeros(shape, bool)
        masks, policy = UnknownMasks(none(begin_params.log_channel_scale.shape), none(begin_params.raw_chemical.shape),
                                     none(begin_params.raw_gap.shape)), dict(version='network-frozen')
    (output / 'fit_policy.json').write_text(json.dumps(policy, indent=2, default=str) + '\n')
    unit = config.synaptic_unit_nS

    def make_responses(substeps):
        def run(p, c_exc, c_inh, tau, state0, level0, conductance):
            def step(carry, k):
                state, level = carry
                on = (k < stim_frames).astype(conductance.dtype)
                peptide = peptide_membrane_input(level, c_exc, c_inh, unit)
                membrane = MembraneConductanceInput(conductance * on + peptide.conductance_nS,
                                                    conductance * on * REVERSAL_MV + peptide.weighted_reversal_pA)
                state = advance_network(state, p, fixed, DT_S, substeps, config, membrane_input=membrane)
                return (state, peptide_level_step(level, state.voltage_mV, fixed, DT_S, tau)), state.indicator
            return jax.lax.scan(jax.checkpoint(step), (state0, level0), jnp.arange(FRAMES))[1]

        def responses(p, c_exc, c_inh, tau, state0, level0, conductances):
            control = run(p, c_exc, c_inh, tau, state0, level0, jnp.zeros(302, float_type))
            stimulated = jax.vmap(lambda g: run(p, c_exc, c_inh, tau, state0, level0, g))(conductances)
            return (stimulated - control[None])[:, lo:hi].mean(axis=1)
        return responses

    def make_settled(substeps):
        @jax.jit
        def settled(p, c_exc, c_inh, tau):
            def step(carry, _):
                state, level = carry
                membrane = peptide_membrane_input(level, c_exc, c_inh, unit)
                state = advance_network(state, p, fixed, DT_S, substeps, config, membrane_input=membrane)
                return (state, peptide_level_step(level, state.voltage_mV, fixed, DT_S, tau)), None
            start = (initial_network_state(fixed, config), jnp.zeros(302, float_type))
            return jax.lax.scan(step, start, None, length=int(round(60. / DT_S)))[0]
        return settled

    responses, settled = make_responses(args.substeps), make_settled(args.substeps)
    couplings = lambda pep: coupling_matrices(pep, sender, receiver)
    off = lambda pep: PeptideParams(jnp.full_like(pep.raw_exc, -50.), jnp.full_like(pep.raw_inh, -50.), pep.log_tau_s)

    def objective(theta, state0, level0, conductances, z, w):
        q = project_unknowns(theta.network, reference, masks)
        c_exc, c_inh = couplings(theta.pep)
        response = responses(q, c_exc, c_inh, theta.pep.log_tau_s, state0, level0, conductances)[:len(targets)]
        loss, (pooled, per_target) = correlation_loss(response, z, w)
        prior = jnp.sum(((q.log_channel_scale - reference.log_channel_scale) * masks.channels) ** 2) / jnp.maximum(masks.channels.sum(), 1.)
        for value, start, mask in ((q.raw_chemical, reference.raw_chemical, masks.chemical), (q.raw_gap, reference.raw_gap, masks.gap)):
            prior += jnp.sum(((value - start) * mask) ** 2) / jnp.maximum(mask.sum(), 1.)
        gains = jnp.mean(jax.nn.softplus(theta.pep.raw_exc) ** 2) + jnp.mean(jax.nn.softplus(theta.pep.raw_inh) ** 2)
        return loss + .001 * prior + args.peptide_prior * gains, (loss, pooled, per_target, response)

    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(args.learning_rate))

    @jax.jit
    def update(theta, opt_state, state0, level0, conductances, z, w):
        (_, (loss, pooled, per_target, response)), grad = jax.value_and_grad(objective, has_aux=True)(theta, state0, level0, conductances, z, w)
        grad = Theta(project_unknowns(grad.network, jax.tree.map(jnp.zeros_like, grad.network), masks), grad.pep)
        norms = [jnp.linalg.norm(g) for g in (*grad.network, *grad.pep)]
        finite = jnp.all(jnp.asarray([jnp.isfinite(g).all() for g in (*grad.network, *grad.pep)]))
        updates, opt_state = optimizer.update(grad, opt_state, theta)
        new = optax.apply_updates(theta, updates)
        net = new.network
        if train_network:
            net = net._replace(
                log_channel_scale=jnp.clip(net.log_channel_scale, reference.log_channel_scale - CHANNEL_BOUND, reference.log_channel_scale + CHANNEL_BOUND),
                raw_chemical=jnp.where(fixed.chemical_mask, jnp.clip(net.raw_chemical, reference.raw_chemical - EDGE_BOUND, reference.raw_chemical + EDGE_BOUND), reference.raw_chemical),
                raw_gap=jnp.where(fixed.gap_mask, jnp.clip(net.raw_gap, reference.raw_gap - EDGE_BOUND, reference.raw_gap + EDGE_BOUND), reference.raw_gap))
        return Theta(project_unknowns(net, reference, masks), clip_peptide_params(new.pep)), opt_state, loss, pooled, per_target, response, norms, finite

    def run_settle(theta):
        c_exc, c_inh = couplings(theta.pep)
        return jax.lax.stop_gradient(settled(project_unknowns(theta.network, reference, masks), c_exc, c_inh, theta.pep.log_tau_s))

    def evaluate_theta(theta, pep_override=None):
        pep = theta.pep if pep_override is None else pep_override
        c_exc, c_inh = couplings(pep)
        q = project_unknowns(theta.network, reference, masks)
        state0, level0 = jax.lax.stop_gradient(settled(q, c_exc, c_inh, pep.log_tau_s))
        return np.asarray(jax.jit(responses)(q, c_exc, c_inh, pep.log_tau_s, state0, level0, stimulus))[:len(targets)]

    def save(stem, theta):
        save_network(output / f'{stem}.npz', project_unknowns(theta.network, reference, masks), fixed, config)
        np.savez_compressed(output / f'{stem}_peptides.npz', raw_exc=np.asarray(theta.pep.raw_exc), raw_inh=np.asarray(theta.pep.raw_inh),
                            log_tau_s=np.asarray(theta.pep.log_tau_s), asset_sha256=asset_sha)

    def evaluate(response, split, permutations):
        stats = split_stats[split]
        result = pair_metrics(model_pairs(np.asarray(response), targets, neuron_ids, stats), stats, seed=args.seed, permutations=permutations)
        return result if permutations else {k: v for k, v in result.items() if not k.startswith('permutation')}

    def snapshot(response, permutations):
        return dict(train=evaluate(response, 'train', permutations), val=evaluate(response, 'val', permutations),
                    test=evaluate(response, 'test', permutations), self_response_fraction=self_response_fraction(np.asarray(response), targets, neuron_ids))

    rng = np.random.default_rng(args.seed)
    theta = Theta(begin_params, initial_peptide_params(pairs, args.peptide_tau_s, float_type))
    opt_state = optimizer.init(theta)
    history = (output / 'history.jsonl').open('w')
    evaluations, best_val = [], dict(score=-math.inf)
    status, stop_reason, step, initial_response = 'completed', 'max_steps', 0, None
    state0 = level0 = None
    z_all, w_all = jnp.asarray(z_train, float_type), jnp.asarray(w_train, float_type)
    while step < args.max_steps:
        if step and time.monotonic() > deadline:
            stop_reason = 'deadline'
            break
        resettled = step % args.resettle_every == 0
        if resettled:
            state0, level0 = run_settle(theta)
        w_step = w_all * jnp.asarray(rng.random(w_all.shape) < args.pair_fraction, float_type)
        new, new_opt, loss, pooled, per_target, response, norms, finite = update(theta, opt_state, state0, level0, stimulus, z_all, w_step)
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
                      per_target_corr=float(per_target), grad_norms=[float(n) for n in norms], resettled=resettled,
                      peptide_tau_s=float(jnp.exp(theta.pep.log_tau_s)))
        history.write(json.dumps(record) + '\n')
        history.flush()
        if step % 5 == 0:
            print(json.dumps(record), flush=True)
        theta, opt_state, step = new, new_opt, step + 1
        if step % args.checkpoint_every == 0:
            save('network_latest', theta)
            (output / 'optimizer_latest.msgpack').write_bytes(serialization.to_bytes(jax.device_get(opt_state)))
            if 'theta' in best_val:
                save('network_best_val', best_val['theta'])
    history.close()

    best_theta = best_val.get('theta', theta)
    final_response, best_response = evaluate_theta(theta), evaluate_theta(best_theta)
    final_off, best_off = evaluate_theta(theta, off(theta.pep)), evaluate_theta(best_theta, off(best_theta.pep))
    save('network_final', theta)
    save('network_best_val', best_theta)
    (output / 'optimizer_final.msgpack').write_bytes(serialization.to_bytes(jax.device_get(opt_state)))
    np.savez_compressed(output / 'responses.npz', targets=np.array(targets), initial=initial_response if initial_response is not None else np.zeros(0),
                        final=final_response, best_val=best_response, final_peptides_off=final_off, best_val_peptides_off=best_off,
                        stimulus_nS=args.stimulus_nS)
    # Substep resolution check with the final parameters.
    convergence_targets = min(13, len(targets))
    c_exc, c_inh = couplings(theta.pep)
    q = project_unknowns(theta.network, reference, masks)
    state_f, level_f = make_settled(2 * args.substeps)(q, c_exc, c_inh, theta.pep.log_tau_s)
    fine = np.asarray(jax.jit(make_responses(2 * args.substeps))(q, c_exc, c_inh, theta.pep.log_tau_s, state_f, level_f,
                                                                 stimulus[:convergence_targets])) if convergence_targets else None
    coarse = final_response[:convergence_targets]
    exc, inh = np.asarray(jax.nn.softplus(theta.pep.raw_exc)), np.asarray(jax.nn.softplus(theta.pep.raw_inh))
    labels = asset['metadata']['pair_labels']
    top = np.argsort(-(exc + inh))[:15]
    net_delta = jax.tree.map(lambda a, b: np.asarray(a) - np.asarray(b), q, reference)
    summary = dict(
        experiment=args.experiment_id, scope=SCOPE, train_network=train_network, status=status, stop_reason=stop_reason, steps=step,
        elapsed_minutes=(time.monotonic() - started) / 60., budget_minutes=args.minutes, deadline_reached=stop_reason == 'deadline',
        parent_network=str(args.network.resolve()), parent_sha256=sha256(args.network),
        anchor_network=str((args.anchor_network or args.network).resolve()), anchor_sha256=sha256(args.anchor_network or args.network),
        optimizer_start='fresh (not a mid-run state resume)', peptide_asset_sha256=asset_sha,
        final_sha256=sha256(output / 'network_final.npz'), final_peptides_sha256=sha256(output / 'network_final_peptides.npz'),
        best_val_sha256=sha256(output / 'network_best_val.npz'), best_val_peptides_sha256=sha256(output / 'network_best_val_peptides.npz'),
        best_val=dict(spearman=best_val['score'] if 'theta' in best_val else None, step=best_val.get('step')),
        devices=[str(d) for d in devices], data=data_summary, stimulus_nS=args.stimulus_nS, learning_rate=args.learning_rate,
        pair_fraction=args.pair_fraction, resettle_every=args.resettle_every, substeps=args.substeps,
        peptide_prior=args.peptide_prior,
        trainable=dict(channels=int(masks.channels.sum()), chemical=int(masks.chemical.sum()), gap=int(masks.gap.sum()),
                       peptide_pairs=int(pairs), peptide_values=int(2 * pairs + 1)),
        changed_entries={n: int(np.count_nonzero(d)) for n, d in zip(('log_channel_scale', 'raw_chemical', 'raw_gap'), net_delta)},
        frozen_parameter_changes=frozen_parameter_changes(q, reference, masks),
        peptide=dict(tau_s=float(np.exp(np.asarray(theta.pep.log_tau_s))), excitatory_gain_sum=float(exc.sum()), inhibitory_gain_sum=float(inh.sum()),
                     pairs_excitatory_above_0p1=int((exc > .1).sum()), pairs_inhibitory_above_0p1=int((inh > .1).sum()),
                     pairs_at_max_gain=int(((exc > 2.9) | (inh > 2.9)).sum()),
                     top_pairs=[dict(peptide=labels[i]['peptide'], receptor=labels[i]['receptor'], excitatory=float(exc[i]), inhibitory=float(inh[i])) for i in top]),
        evaluations=evaluations, final=snapshot(final_response, 500), best_val_checkpoint=snapshot(best_response, 500),
        final_peptides_off=snapshot(final_off, 500), best_val_peptides_off=snapshot(best_off, 500),
        substep_convergence=None if fine is None else dict(
            targets=convergence_targets, correlation=float(np.corrcoef(fine.ravel(), coarse.ravel())[0, 1]),
            max_abs_difference_over_max_abs_response=float(np.abs(fine - coarse).max() / max(np.abs(coarse).max(), 1e-30))),
        selection_rule='best validation-animal Spearman over evaluation steps; final also kept; test animals never used for selection',
        unavailable=['body/behavior responses', 'sensory start/stop control', 'unc-31 atlas comparison (peptide term fitted to wild type only)',
                     'range/diffusion of peptides (all-to-all within expression)', 'amplitude calibration of the calcium indicator'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    print(json.dumps(dict(status=status, steps=step, final=summary['final'], best_val_checkpoint=summary['best_val_checkpoint'],
                          best_val_peptides_off=summary['best_val_peptides_off']), default=float), flush=True)
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
