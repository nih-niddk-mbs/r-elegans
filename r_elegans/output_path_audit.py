"""Descriptive analysis of recorded muscle/output signals; no simulator imports."""
import numpy as np
from scipy.signal import detrend
from scipy.special import expit

from .hybrid_assessment import cycle_excursions, peak_frequency


def projection():
    """NumPy reconstruction of the legacy32 anatomical interpolation."""
    positions = np.array(list(range(24))*2 + list(range(23)) + list(range(24)), dtype=np.float32)/23
    joints = 1 - np.linspace(0, 1, 11, dtype=np.float32)
    local = np.maximum(1 - np.abs(joints[:, None]-positions)/np.float32(.1), 0)
    dorsal = np.arange(95) < 48
    result = []
    for mask in (dorsal, ~dorsal):
        weights = local * mask
        result.append((weights/weights.sum(axis=1, keepdims=True)).astype(float))
    return result


def spectral_balance(t, dorsal, ventral, frequency):
    """Net/common oscillatory coefficient ratio: 0 cancels, 1 survives.

    At a single observed motor frequency; not broadband energy or causality.
    """
    if frequency is None:
        return dict(frequency_hz=None, survival_ratio=None, phase_difference_rad=None)
    carrier = np.exp(-2j*np.pi*frequency*(t-t[0]))
    d = detrend(dorsal, axis=0).T @ carrier
    v = detrend(ventral, axis=0).T @ carrier
    denominator = np.abs(d)+np.abs(v)
    good = denominator > 1e-12 * len(t)
    ratio = np.divide(np.abs(d-v), denominator, out=np.zeros_like(denominator), where=good)
    return dict(frequency_hz=frequency,
                survival_ratio=[float(x) if ok else None for x, ok in zip(ratio, good)],
                phase_difference_rad=[float(x) if ok else None for x, ok in zip(np.angle(d*np.conj(v)), good)])


def nmj_transfer(voltage, network, actor, asset):
    """Re-evaluate the fixed algebraic readout, never the neural dynamics."""
    if network['neuron_ids'].tolist() != asset['neuron_ids']:
        raise ValueError('Runtime NMJ asset neuron ordering differs')
    sparse = asset['nmj']
    counts = np.zeros((95, 302), dtype=np.float32)
    signs = np.zeros_like(counts)
    indices = (sparse['muscle_index'], sparse['neuron_index'])
    counts[indices] = np.asarray(sparse['contact_count'], dtype=np.float32)
    signs[indices] = np.asarray(sparse['sign'], dtype=np.float32)
    weights = counts.astype(float)*np.logaddexp(0., actor['raw_nmj'].astype(float))
    polarity = np.where(signs != 0, signs, np.tanh(actor['unknown_polarity'].astype(float)))*(counts > 0)
    release = expit((voltage-network['fixed_release_threshold_mV'].astype(float))/network['fixed_release_slope_mV'].astype(float))
    drive = (release @ (weights*polarity).T)/np.maximum(weights.sum(axis=1), 1.)
    width = .02+np.logaddexp(0., actor['raw_muscle_slope'].astype(float))
    activation = np.clip(expit((drive-actor['muscle_threshold'].astype(float))/width), 1e-5, 1-1e-5)
    return drive, activation


def analyze(trace, neuron_ids, raw_muscle_slope, *, window_seconds=15., reference='RMDR', nmj_drive=None):
    t = np.asarray(trace['dense_time_s'], dtype=float)
    if len(t) < 80 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or t[-1] < 2*window_seconds:
        raise ValueError('Ordered trace covering two full analysis windows required')
    fields = {key: np.asarray(trace[key], dtype=float) for key in (
        'dense_muscles', 'dense_command', 'dense_filtered_drive', 'dense_joints', 'dense_voltage_mV')}
    for key, columns in (('dense_muscles', 95), ('dense_command', 11),
                         ('dense_filtered_drive', 11), ('dense_joints', 11), ('dense_voltage_mV', 302)):
        if fields[key].shape != (len(t), columns) or not np.isfinite(fields[key]).all():
            raise ValueError('Invalid recorded field: '+key)
    wd, wv = projection()
    muscles = fields['dense_muscles']
    dorsal, ventral = muscles @ wd.T, muscles @ wv.T
    error = float(np.max(np.abs(dorsal-ventral-fields['dense_command'])))
    # NumPy and XLA float32 interpolation/reductions may differ slightly.
    # Reject a wrong ordering/projection rather than interpreting it.
    if error > 1e-7:
        raise ValueError(f'Reconstructed anatomical projection disagrees with recorded command: {error}')
    width = .02 + np.logaddexp(0., np.asarray(raw_muscle_slope, dtype=float))
    if width.shape not in ((), (95,)):
        raise ValueError('Invalid frozen muscle sigmoid width')
    windows = []
    for start, end in ((t[-1]-2*window_seconds, t[-1]-window_seconds), (t[-1]-window_seconds, t[-1])):
        use = (t >= start-1e-12) & (t <= end+1e-12)
        clock = t[use]
        freq = peak_frequency(clock, fields['dense_voltage_mV'][use, neuron_ids.index(reference)], .01)
        signals = dict(dorsal=dorsal[use], ventral=ventral[use],
                       net_command=fields['dense_command'][use], filtered_drive=fields['dense_filtered_drive'][use],
                       joints=fields['dense_joints'][use])
        cycles = {key: cycle_excursions(value, late_only=False) for key, value in signals.items()}
        # Existing cycle estimator keys say 'rad'; only joint values have radians.
        amplitudes = {key: dict(per_joint=row['joint_cycle_excursion_rad'],
                                tail_middle_head=row['regional_cycle_excursion_rad']) for key, row in cycles.items()}
        a = muscles[use]
        windows.append(dict(start_s=float(start), end_s=float(end), cycles=amplitudes,
                            balance=spectral_balance(clock, dorsal[use], ventral[use], freq),
                            muscle_raw_p2p=np.ptp(a, axis=0).tolist(),
                            muscle_near_extremes_fraction=np.mean((a <= .01) | (a >= .99), axis=0).tolist(),
                            muscle_mean_local_sigmoid_slope=np.mean(a*(1-a)/width, axis=0).tolist()))
        if nmj_drive is not None:
            windows[-1]['signed_normalized_nmj_drive_raw_p2p'] = np.ptp(nmj_drive[use], axis=0).tolist()
        windows[-1]['neuron_voltage_raw_p2p_mV'] = np.ptp(fields['dense_voltage_mV'][use], axis=0).tolist()
    return dict(reference_motor=reference, projection_maximum_error=error, windows=windows,
                minimum_joint_cycles_tail_middle_head=np.min(
                    [w['cycles']['joints']['tail_middle_head'] for w in windows], axis=0).tolist())
