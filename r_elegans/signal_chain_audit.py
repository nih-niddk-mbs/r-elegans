"""Descriptive raw reversals in saved trajectories; no simulation or fitting."""
import re

import numpy as np
from scipy.signal import find_peaks


# Engineering reporting floors, not measured physiological thresholds.
FIELDS = {
    'dense_voltage_mV': ('mV', .1),
    'dense_calcium_uM': ('uM', .001),
    'dense_muscles': ('activation', .0001),
    'dense_command': ('command', .0001),
    'dense_filtered_drive': ('drive', .0001),
    'dense_joints': ('rad', .00001),
}


def channel_statistics(times, values, floor, begin_s=4.):
    """Separate raw turning points from detrended range in two late windows."""
    t = np.asarray(times, dtype=float)
    x = np.asarray(values, dtype=float)
    if (t.ndim != 1 or x.ndim != 2 or len(t) != len(x)
            or not np.isfinite(t).all() or not np.isfinite(x).all()
            or np.any(np.diff(t) <= 0) or not np.isfinite(floor) or floor <= 0
            or not np.isfinite(begin_s)):
        raise ValueError('Finite ordered times, time-by-channel values and positive floor required')
    use = t >= begin_s
    t, x = t[use], x[use]
    if len(t) < 40 or x.shape[1] == 0:
        raise ValueError('At least40 observation samples and one channel required')
    prominence = max(1e-10, .1 * floor)
    windows = np.array_split(np.arange(len(t)), 2)
    ranges, turns = [], []
    for indices in windows:
        clock = t[indices] - t[indices].mean()
        w = x[indices]
        residual = w - w.mean(axis=0) - clock[:, None] * (clock @ w / (clock @ clock))
        ranges.append(np.ptp(residual, axis=0))
        turns.append([len(find_peaks(w[:, j], prominence=prominence)[0])
                      + len(find_peaks(-w[:, j], prominence=prominence)[0])
                      for j in range(w.shape[1])])
    ranges = np.asarray(ranges)
    minimum, maximum = ranges.min(axis=0), ranges.max(axis=0)
    ratio = minimum / np.maximum(maximum, 1e-30)
    clock = t - t.mean()
    slopes = clock @ x / (clock @ clock)
    changes = np.diff(x, axis=0)
    up = np.maximum(changes, 0).sum(axis=0)
    down = np.maximum(-changes, 0).sum(axis=0)
    peaks = np.array([len(find_peaks(x[:, j], prominence=prominence)[0]) for j in range(x.shape[1])])
    troughs = np.array([len(find_peaks(-x[:, j], prominence=prominence)[0]) for j in range(x.shape[1])])
    raw_range = np.ptp(x, axis=0)
    repeating = ((minimum >= floor) & (peaks >= 2) & (troughs >= 2)
                 & (np.min(turns, axis=0) > 0) & (ratio >= .5))
    return dict(
        observation_begin_s=float(t[0]), observation_end_s=float(t[-1]), samples=len(t),
        engineering_floor=float(floor), peak_prominence=float(prominence),
        raw_p2p=raw_range.tolist(), minimum_window_detrended_p2p=minimum.tolist(),
        window_amplitude_ratio=ratio.tolist(), peaks=peaks.tolist(), troughs=troughs.tolist(),
        raw_net_change=(x[-1] - x[0]).tolist(), linear_slope_per_s=slopes.tolist(),
        positive_variation=up.tolist(), negative_variation=down.tolist(),
        weaker_direction_variation_fraction=(np.minimum(up, down) / np.maximum(up + down, 1e-30)).tolist(),
        reversing_indices=np.flatnonzero(((peaks + troughs) > 0) & (raw_range >= floor)).tolist(),
        repeating_indices=np.flatnonzero(repeating).tolist(),
        no_raw_extrema_indices=np.flatnonzero((peaks + troughs) == 0).tolist(),
    )


def summarize_trace(trace, neuron_ids, begin_s=4.):
    t = trace['dense_time_s']
    result = {}
    for field, (units, floor) in FIELDS.items():
        x = trace[field]
        if field in ('dense_voltage_mV', 'dense_calcium_uM') and x.shape[1] != len(neuron_ids):
            raise ValueError('Neural channel count differs from checkpoint neuron IDs')
        if field in ('dense_command', 'dense_filtered_drive', 'dense_joints') and x.shape[1] != 11:
            raise ValueError('Expected11 body-joint channels')
        if field == 'dense_muscles' and x.shape[1] != 95:
            raise ValueError('Expected95 muscle activation channels')
        row = channel_statistics(t, x, floor, begin_s)
        row.update(units=units, stored_dtype=str(x.dtype), channels=x.shape[1])
        if field in ('dense_voltage_mV', 'dense_calcium_uM'):
            row['repeating_neurons'] = [neuron_ids[i] for i in row['repeating_indices']]
            row['reversing_neurons'] = [neuron_ids[i] for i in row['reversing_indices']]
            groups = {
                'numbered_body_motor_names': [i for i, n in enumerate(neuron_ids) if re.fullmatch(r'(DA|DB|DD|VA|VB|VD|AS)\d+', n)],
                'RMD_SMD_SMB_names': [i for i, n in enumerate(neuron_ids) if n.startswith(('RMD', 'SMD', 'SMB'))],
                'AVA_AVB_AVD_AVE_PVC_names': [i for i, n in enumerate(neuron_ids) if n.startswith(('AVA', 'AVB', 'AVD', 'AVE', 'PVC'))],
                'AWA_names': [i for i, n in enumerate(neuron_ids) if n.startswith('AWA')],
            }
            row['named_groups'] = {name: dict(
                neurons=[neuron_ids[i] for i in indices],
                repeating_neurons=[neuron_ids[i] for i in indices if i in row['repeating_indices']],
                maximum_raw_p2p=max((row['raw_p2p'][i] for i in indices), default=0.),
            ) for name, indices in groups.items()}
        if field == 'dense_muscles':
            late = np.asarray(x)[np.asarray(t) >= begin_s]
            row['activation_near_zero_fraction'] = float(np.mean(late < .01))
            row['activation_near_one_fraction'] = float(np.mean(late > .99))
        result[field] = row
    return result


def plot_raw_chain(trace, statistics, neuron_ids, destination, title, *, begin_s=4., show_cue=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = trace['dense_time_s']
    fig, axes = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
    for ax, (field, (units, _)) in zip(axes, FIELDS.items()):
        row = statistics[field]
        # Show a repeating channel when present, plus the largest raw ranges.
        selected = row['repeating_indices'][:1]
        order = np.argsort(row['raw_p2p'])[::-1]
        for j in order:
            if int(j) not in selected:
                selected.append(int(j))
            if len(selected) >= 3:
                break
        for j in selected:
            label = neuron_ids[j] if field in ('dense_voltage_mV', 'dense_calcium_uM') else str(j)
            ax.plot(t, trace[field][:, j], label=label, linewidth=.8)
        if show_cue:
            ax.axvspan(2., 3., color='orange', alpha=.15)
        ax.axvline(begin_s, color='black', linestyle=':', linewidth=.8)
        ax.set_ylabel(field.removeprefix('dense_') + '\n' + units)
        ax.legend(loc='upper right', ncol=3, fontsize=8)
    axes[-1].set_xlabel('Saved simulation time (s); '+('shading=cue schedule' if show_cue else 'no ligand cue')+', dotted=analysis start')
    fig.suptitle(title + '\nRaw signals; example channels, complete channel counts in JSON')
    fig.tight_layout()
    fig.savefig(destination, dpi=130)
    plt.close(fig)
