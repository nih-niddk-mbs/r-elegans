"""Compare simulated stimulation responses with the Randi et al. 2023 atlas (numpy only).

Data responses are scaled by a per-ROI null spread from sham windows at times with no stimulation (robust MAD), so
each response is a z score. Only stimulations whose target itself responds (z >= ``min_target_z``) are used. Pair
statistics pool trials; the Stouffer sum z = sum(z)/sqrt(n) is the pair's evidence for excitation (>0) or inhibition (<0).
"""
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata, spearmanr

from r_elegans.data.randi_atlas import ratio_signal, stimulus_responses


def sham_starts(animal, rng, *, count=200, pre_s=10., post_end_s=12., margin_s=5.):
    time, start, stop = animal['time_s'], animal['stim_start_s'], animal['stim_stop_s']
    low, high = time[0] + pre_s, time[-1] - post_end_s
    chosen = []
    for t in rng.uniform(low, high, size=count * 4):
        if np.all((t + post_end_s + margin_s < start - pre_s) | (t - margin_s - pre_s > stop + post_end_s)):
            chosen.append(t)
        if len(chosen) == count:
            break
    return np.array(chosen)


def null_scale(animal, *, pre_s=10., post_window=(1., 11.), count=200, seed=0, stim_s=.5):
    starts = sham_starts(animal, np.random.default_rng(seed), count=count, pre_s=pre_s, post_end_s=stim_s + post_window[1])
    if starts.size < 20:
        return None
    sham = stimulus_responses(animal, pre_s=pre_s, post_window=post_window, starts=starts, stops=starts + stim_s)
    center = np.nanmedian(sham, axis=0)
    spread = 1.4826 * np.nanmedian(np.abs(sham - center), axis=0)
    return np.where(spread > 1e-9, spread, np.nan)


def animal_trials(animal, *, pre_s=10., post_window=(1., 11.), min_target_z=3., seed=0):
    """List of successful named-target trials with z scores of every named ROI."""
    scale = null_scale(animal, pre_s=pre_s, post_window=post_window, seed=seed)
    if scale is None:
        return []
    responses = stimulus_responses(animal, pre_s=pre_s, post_window=post_window)
    z = responses / scale
    names = animal['names']
    trials = []
    for k, column in enumerate(animal['stim_target_column']):
        if column < 0 or not names[column] or not np.isfinite(z[k, column]) or z[k, column] < min_target_z:
            continue
        values = {names[i]: float(z[k, i]) for i in range(len(names)) if names[i] and i != column and np.isfinite(z[k, i])}
        trials.append(dict(subject=animal['subject'], target=names[column], self_z=float(z[k, column]), z=values))
    return trials


def pair_statistics(trials, *, min_trials=2):
    """Pooled (target, responder) statistics: mean z, trial count and Stouffer z."""
    total, count = defaultdict(float), defaultdict(int)
    for trial in trials:
        for responder, value in trial['z'].items():
            total[(trial['target'], responder)] += value
            count[(trial['target'], responder)] += 1
    pairs = sorted(p for p, n in count.items() if n >= min_trials)
    n = np.array([count[p] for p in pairs])
    s = np.array([total[p] for p in pairs])
    return dict(pairs=pairs, n=n, mean_z=s / np.maximum(n, 1), stouffer=s / np.sqrt(np.maximum(n, 1)))


def split_subjects(subjects, every=3):
    ordered = sorted(set(subjects))
    held_out = set(ordered[::every])
    return [s for s in ordered if s not in held_out], sorted(held_out)


def auc(score, positive):
    positive = np.asarray(positive, bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = rankdata(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def model_pairs(model_matrix, targets, neuron_ids, stats):
    """Model response for each statistics pair; NaN where a target was not simulated."""
    row = {name: i for i, name in enumerate(targets)}
    column = {name: i for i, name in enumerate(neuron_ids)}
    return np.array([model_matrix[row[t], column[r]] if t in row and r in column else np.nan for t, r in stats['pairs']])


def pair_metrics(model_values, stats, *, threshold=3., deadband=1e-6, permutations=500, seed=0):
    keep = np.isfinite(model_values)
    m, z, s = model_values[keep], stats['mean_z'][keep], stats['stouffer'][keep]
    targets = np.array([p[0] for p, k in zip(stats['pairs'], keep) if k])
    out = dict(pairs=int(keep.sum()), excited=int((s >= threshold).sum()), inhibited=int((s <= -threshold).sum()))
    if keep.sum() < 10:
        return out
    out['spearman_mean_z'] = float(spearmanr(m, z)[0])
    out['auc_excited'] = auc(m, s >= threshold)
    out['auc_inhibited'] = auc(-m, s <= -threshold)
    significant = np.abs(s) >= threshold
    active = significant & (np.abs(m) > deadband)
    out['sign_agreement_among_significant_active'] = float(np.mean(np.sign(m[active]) == np.sign(s[active]))) if active.any() else float('nan')
    out['model_silent_fraction_among_significant'] = float(np.mean(np.abs(m[significant]) <= deadband)) if significant.any() else float('nan')
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(targets == t) for t in np.unique(targets)]
    null_rho, null_auc = [], []
    for _ in range(permutations):
        shuffled = m.copy()
        for g in groups:
            shuffled[g] = m[rng.permutation(g)]
        null_rho.append(spearmanr(shuffled, z)[0])
        null_auc.append(auc(shuffled, s >= threshold))
    out['permutation_p_spearman'] = float((1 + np.sum(np.array(null_rho) >= out['spearman_mean_z'])) / (1 + permutations))
    out['permutation_p_auc_excited'] = float((1 + np.nansum(np.array(null_auc) >= out['auc_excited'])) / (1 + permutations))
    return out


def self_response_fraction(model_matrix, targets, neuron_ids):
    column = {name: i for i, name in enumerate(neuron_ids)}
    values = [model_matrix[i, column[t]] for i, t in enumerate(targets) if t in column]
    return float(np.mean(np.array(values) > 0)) if values else float('nan')
