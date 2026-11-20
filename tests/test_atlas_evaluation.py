import jax
import numpy as np

from r_elegans.atlas_evaluation import (animal_trials, auc, model_pairs, null_scale, pair_metrics, pair_statistics,
                                        self_response_fraction, split_subjects)


def synthetic_animal(seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(0., 1200., .5)
    green = 1. + .05 * rng.standard_normal((time.size, 3))
    red = np.ones_like(green)
    starts = np.array([300., 600., 900.])
    for s in starts:  # ROI 0 is the target and responds; ROI 1 is excited; ROI 2 is unaffected
        window = (time >= s + 1.5) & (time < s + 12.)
        green[window, 0] += 1.
        green[window, 1] += .5
    return dict(subject='sub-1', time_s=time, green=green, red=red, names=['AWAR', 'AVAL', 'AIYL'], stim_start_s=starts,
                stim_stop_s=starts + .5, stim_target_column=np.array([0, 0, 0]))


def test_null_scale_is_positive_and_trials_keep_only_responding_targets():
    animal = synthetic_animal()
    assert np.all(null_scale(animal) > 0)
    trials = animal_trials(animal)
    assert len(trials) == 3 and {t['target'] for t in trials} == {'AWAR'}
    assert all(t['z']['AVAL'] > 3 for t in trials) and all(abs(t['z']['AIYL']) < 3 for t in trials)
    animal['stim_target_column'] = np.array([2, 2, 2])  # an unresponsive target is not a successful stimulation
    assert animal_trials(animal) == []


def test_model_that_matches_the_data_pattern_scores_perfectly_and_shuffling_does_not():
    trials = [dict(subject=f's{i}', target='A', self_z=9., z={'B': 6. + i, 'C': -6. - i, 'D': .1 * i, 'E': 5., 'G': 4., 'H': -4.}) for i in range(3)]
    trials += [dict(subject=f's{i}', target='F', self_z=9., z={'B': -5. - i, 'C': 5. + i, 'D': .1, 'E': -5., 'G': -4., 'H': 4.}) for i in range(3)]
    stats = pair_statistics(trials)
    neuron_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    matrix = np.zeros((2, 8))
    for row, target in enumerate(['A', 'F']):
        for responder in 'BCDEGH':
            k = stats['pairs'].index((target, responder))
            matrix[row, neuron_ids.index(responder)] = stats['stouffer'][k] * 1e-3
    matched = pair_metrics(model_pairs(matrix, ['A', 'F'], neuron_ids, stats), stats, permutations=50)
    assert matched['spearman_mean_z'] > .99 and matched['auc_excited'] == 1. and matched['sign_agreement_among_significant_active'] == 1.
    assert matched['permutation_p_spearman'] < .1
    shuffled = pair_metrics(-model_pairs(matrix, ['A', 'F'], neuron_ids, stats), stats, permutations=50)
    assert shuffled['spearman_mean_z'] < -.99 and shuffled['sign_agreement_among_significant_active'] == 0.


def test_auc_and_split_and_self_response():
    assert auc(np.array([.1, .2, .9, .8]), np.array([0, 0, 1, 1])) == 1.
    assert np.isnan(auc(np.array([.1, .2]), np.array([0, 0])))
    train, held = split_subjects([f'sub-{i}' for i in range(9)] * 2)
    assert set(train).isdisjoint(held) and len(held) == 3
    assert self_response_fraction(np.array([[1., 0.], [-1., 0.]]), ['A', 'A'], ['A', 'B']) == .5


def test_stimulation_windows_have_zero_response_without_stimulus_and_are_finite_with_it():
    from r_elegans.brain.conductance_network import initialize_conductance_network, ConductanceNetworkConfig
    from r_elegans.brain.stimulation_response import settle, stimulation_windows
    from r_elegans.model import load_builtin_model
    from r_elegans.rl.full_network_actor import init_full_network_actor
    model = load_builtin_model()
    params, fixed = initialize_conductance_network(init_full_network_actor(model, jax.random.PRNGKey(0)), model.neuron_ids)
    config = ConductanceNetworkConfig()
    state = settle(params, fixed, config, warmup_s=1., dt_s=.1, substeps=2)
    values, control = stimulation_windows(state, params, fixed, config, np.array([3, 10]), np.array([0., .5]), dt_s=.1, substeps=2,
                                          record_s=3., window_s=(1., 3.), batch_size=2)
    assert values.shape == (2, 302) and control.shape == (30, 302)
    assert np.allclose(values[0], 0., atol=1e-12) and np.all(np.isfinite(values))
    assert np.abs(values[1]).max() > 0.
