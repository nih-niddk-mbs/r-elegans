"""E100: unchanged-checkpoint baseline against the Randi 2023 signal-propagation atlas, with stimulus-strength calibration.

No parameter of the network is trained. One scalar (the added membrane conductance of the simulated stimulus) is chosen
on training animals only; metrics are reported on held-out animals. Windows, the z-score null, the success filter and
the metrics are documented choices (r_elegans.atlas_evaluation), not the paper's definitions.
"""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import jax
import numpy as np

from r_elegans.assets import load_asset_document
from r_elegans.atlas_evaluation import (animal_trials, model_pairs, pair_metrics, pair_statistics, self_response_fraction,
                                        split_subjects)
from r_elegans.brain.conductance_network import load_network
from r_elegans.brain.stimulation_response import settle, stimulation_windows
from r_elegans.data.randi_atlas import load_randi_animal

warnings.filterwarnings('ignore', category=RuntimeWarning)  # NaN windows are expected
SCOPE = ('Unchanged-checkpoint comparison of simulated single-neuron stimulation with calcium responses; '
         'no fit, no biological-fidelity or behavior claim; atlas responses are calcium ratios, model responses are indicator changes.')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def find_checkpoint(run, wanted):
    for path in sorted(Path(run).rglob('network.npz')):
        if sha256(path) == wanted:
            return path
    raise SystemExit(f'No network.npz with sha256 {wanted} under {run}')


def build_statistics(root, neuron_ids, seed):
    trials, animals = [], 0
    for path in sorted(Path(root).glob('sub-*/*.nwb')):
        animal = load_randi_animal(path, neuron_ids)
        trials += animal_trials(animal, seed=seed)
        animals += 1
    train_subjects, test_subjects = split_subjects([t['subject'] for t in trials])
    train = [t for t in trials if t['subject'] in train_subjects]
    test = [t for t in trials if t['subject'] in test_subjects]
    return animals, trials, pair_statistics(train), pair_statistics(test), train_subjects, test_subjects, len(train), len(test)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--budget-minutes', type=float, default=170.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-targets', type=int, default=None, help='dry runs only')
    parser.add_argument('--phases', type=int, default=8)
    parser.add_argument('--substeps', type=int, default=20)
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started + args.budget_minutes * 60. - 600.
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.expanduser().resolve()
    if output == repo or repo in output.parents:
        parser.error('Keep outputs external')
    output.mkdir(parents=True, exist_ok=False)
    try:
        from .experiment_records import capture_run_record
    except ImportError:
        from experiment_records import capture_run_record
    capture_run_record(output, repo, args)

    neuron_ids = list(load_asset_document('runtime_model_v1.json')['neuron_ids'])
    checkpoint = find_checkpoint(args.run, args.checkpoint_sha256)
    params, fixed, config = load_network(checkpoint)
    root = args.data_root / 'raw/functional/randi2023_dandi001075_segmentation'
    animals, trials, train_stats, test_stats, train_subjects, test_subjects, n_train, n_test = build_statistics(root, neuron_ids, args.seed)
    targets = sorted({t['target'] for t in trials})
    if args.max_targets:
        targets = targets[:args.max_targets]
    indices = np.array([neuron_ids.index(t) for t in targets])
    data_summary = dict(animals=animals, successful_trials=len(trials), train_trials=n_train, test_trials=n_test,
                        train_animals=len(train_subjects), test_animals=len(test_subjects), targets=len(targets),
                        train_pairs=len(train_stats['pairs']), test_pairs=len(test_stats['pairs']))
    print(json.dumps(data_summary), flush=True)
    stages = []
    summary = dict(experiment='E100', scope=SCOPE, checkpoint=str(checkpoint), checkpoint_sha256=args.checkpoint_sha256,
                   data_root=str(args.data_root), data=data_summary, stages=stages, budget_minutes=args.budget_minutes,
                   unavailable=['trial-level latency comparison', 'amplitude calibration of calcium indicator', 'parent loss of E099'])
    options = dict(dt_s=.1, substeps=args.substeps)

    def evaluate(response, split):
        stats = train_stats if split == 'train' else test_stats
        return pair_metrics(model_pairs(response, targets, neuron_ids, stats), stats, seed=args.seed)

    def sweep(state0, grid, **kwargs):
        t0 = time.monotonic()
        k_targets = np.tile(indices, len(grid))
        k_grid = np.repeat(grid, len(indices))
        values, _ = stimulation_windows(state0, params, fixed, config, k_targets, k_grid, **{**options, **kwargs})
        return values.reshape(len(grid), len(indices), -1), time.monotonic() - t0

    warm = settle(params, fixed, config, warmup_s=60., **options)
    grid = np.geomspace(.003, 3., 8)
    responses, seconds = sweep(warm, grid)
    train_scores = [evaluate(r, 'train') for r in responses]
    stages.append(dict(stage='coarse_sweep', conductance_nS=grid.tolist(), seconds=seconds, train=train_scores,
                       test=[evaluate(r, 'test') for r in responses], self_response=[self_response_fraction(r, targets, neuron_ids) for r in responses]))
    best = int(np.nanargmax([s.get('spearman_mean_z', np.nan) for s in train_scores]))
    chosen = float(grid[best])
    summary['selection_rule'] = 'maximum train Spearman between model response and pooled mean z (declared before the run)'
    if time.monotonic() + 3 * seconds / len(grid) * 5 < deadline and 0 < best < len(grid) - 1:
        fine = np.geomspace(grid[best - 1], grid[best + 1], 7)
        fine_responses, fine_seconds = sweep(warm, fine)
        fine_scores = [evaluate(r, 'train') for r in fine_responses]
        stages.append(dict(stage='fine_sweep', conductance_nS=fine.tolist(), seconds=fine_seconds, train=fine_scores,
                           test=[evaluate(r, 'test') for r in fine_responses]))
        best_fine = int(np.nanargmax([s.get('spearman_mean_z', np.nan) for s in fine_scores]))
        chosen = float(fine[best_fine])
    summary['chosen_conductance_nS'] = chosen
    # Phase sensitivity and convergence at the chosen strength, while the budget allows.
    phase_responses = []
    for warmup in (60., 66., 72., 78., 84., 90., 96., 102.)[:args.phases]:
        if time.monotonic() > deadline - 2 * seconds / len(grid):
            stages.append(dict(stage='phase_sensitivity_stopped', reason='time budget', completed=len(phase_responses)))
            break
        state = settle(params, fixed, config, warmup_s=warmup, **options)
        response, _ = sweep(state, np.array([chosen]))
        phase_responses.append(response[0])
        stages.append(dict(stage='phase', warmup_s=warmup, train=evaluate(response[0], 'train'), test=evaluate(response[0], 'test')))
    if phase_responses:
        mean_response = np.mean(phase_responses, axis=0)
        stages.append(dict(stage='phase_averaged', phases=len(phase_responses), train=evaluate(mean_response, 'train'), test=evaluate(mean_response, 'test'),
                           self_response=self_response_fraction(mean_response, targets, neuron_ids)))
        np.savez_compressed(output / 'phase_averaged_responses.npz', targets=np.array(targets), neuron_ids=np.array(neuron_ids), response=mean_response)
    if time.monotonic() < deadline - seconds / len(grid):
        subset = indices[:: max(1, len(indices) // 12)]
        a, _ = stimulation_windows(warm, params, fixed, config, subset, np.full(len(subset), chosen), **options)
        b, _ = stimulation_windows(warm, params, fixed, config, subset, np.full(len(subset), chosen), dt_s=.1, substeps=2 * args.substeps)
        denom = float(np.abs(b).max()) or 1.
        stages.append(dict(stage='substep_convergence', targets=len(subset), max_abs_difference_over_max_abs_response=float(np.abs(a - b).max() / denom),
                           correlation=float(np.corrcoef(a.ravel(), b.ravel())[0, 1])))
    np.savez_compressed(output / 'coarse_responses.npz', targets=np.array(targets), neuron_ids=np.array(neuron_ids), conductance_nS=grid, response=responses)
    summary['elapsed_minutes'] = (time.monotonic() - started) / 60.
    summary['deadline_reached'] = time.monotonic() > deadline
    summary['sha256'] = {p.name: sha256(p) for p in sorted(output.glob('*.npz'))}
    (output / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: summary[k] for k in ('chosen_conductance_nS', 'elapsed_minutes', 'deadline_reached')}), flush=True)


if __name__ == '__main__':
    main()
