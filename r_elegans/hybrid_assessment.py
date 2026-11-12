"""Raw-supported phase/amplitude descriptions and prospective rhythm checks."""
import numpy as np
from scipy.signal import coherence, detrend, find_peaks

from .amplitude_analysis import sustained_excursions
from .full_body_assessment import regional_amplitudes


def peak_frequency(times, signal, prominence):
    peaks = find_peaks(np.asarray(signal), prominence=prominence)[0]
    return float(1 / np.median(np.diff(np.asarray(times)[peaks]))) if len(peaks) >= 3 else None


def cycle_excursions(q, prominence=1e-6):
    """Smaller median up/down raw extremum excursion, excluding posture drift."""
    q = np.asarray(q, dtype=float); late = q[len(q)//2:]
    if q.ndim != 2 or q.shape[1] != 11 or len(late) < 40 or not np.isfinite(q).all():
        raise ValueError('Finite eleven-joint trace required')
    values = []
    for signal in late.T:
        peaks = find_peaks(signal, prominence=prominence)[0]
        troughs = find_peaks(-signal, prominence=prominence)[0]
        events = sorted([(int(i), 1) for i in peaks] + [(int(i), -1) for i in troughs])
        up, down = [], []
        for (i, kind), (j, other) in zip(events[:-1], events[1:]):
            if kind == other:
                continue
            delta = signal[j] - signal[i]
            if delta > 0:
                up.append(float(delta))
            else:
                down.append(float(-delta))
        values.append(min(float(np.median(up)), float(np.median(down))) if min(len(up), len(down)) >= 2 else 0.)
    medians = [float(np.median(np.asarray(values)[g])) for g in np.array_split(np.arange(11), 3)]
    return dict(joint_cycle_excursion_rad=values, regional_cycle_excursion_rad=medians,
                all_three_cycle_amplitude_targets_met=bool(np.all(np.asarray(medians) >= .003)),
                scope='Smaller median raw upward/downward excursion, not a detrended range; .003rad comparison descriptive and separate from historical window gate')


def coordination_statistics(times, q, floor=1e-5):
    t = np.asarray(times, dtype=float); q = np.asarray(q, dtype=float)
    if (q.shape != (len(t), 11) or len(t) < 80 or not np.isfinite(q).all()
            or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)):
        raise ValueError('Finite ordered eleven-joint trajectory required')
    dt = float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t), dt, rtol=1e-7, atol=1e-12):
        raise ValueError('Uniform time grid required for spectral descriptions')
    amp, _ = sustained_excursions(q)
    active = np.flatnonzero(amp >= floor)
    result = dict(eligible_joints=active.tolist(), reference_joint=None, frequency_hz=None,
                  detrended_shape_pc1_fraction=None, coherent_phase_span_rad=None, phases=[],
                  scope='Raw-supported frequency and detrended phase modulo period; phase span alone does not establish travelling waves or propulsion')
    if not len(active):
        return result
    start = len(t) // 2; late_t = t[start:]; late = q[start:]
    ref = int(active[np.argmax(amp[active])])
    frequency = peak_frequency(late_t, late[:, ref], max(1e-8, .1 * amp[ref]))
    if frequency is None:
        return result
    y = detrend(late, axis=0, type='linear')
    singular = np.linalg.svd(y[:, active], compute_uv=False)
    power = singular ** 2
    result.update(reference_joint=ref, frequency_hz=frequency,
                  detrended_shape_pc1_fraction=float(power[0] / power.sum()) if power.sum() > 1e-25 else None)
    carrier = np.exp(-2j * np.pi * frequency * (late_t - late_t[0]))
    coeff = y.T @ carrier
    if abs(coeff[ref]) < 1e-20:
        return result
    nperseg = min(len(late_t) // 2, max(64, round(8 / (frequency * dt))))
    for j in active:
        f, c = coherence(y[:, ref], y[:, j], fs=1 / dt, nperseg=nperseg)
        phase = float(np.angle(coeff[j] / coeff[ref]))
        result['phases'].append(dict(joint=int(j), sustained_p2p_rad=float(amp[j]),
                                     coherence=float(c[np.argmin(abs(f - frequency))]),
                                     phase_relative_rad=phase, lag_modulo_period_s=-phase / (2*np.pi*frequency)))
    trusted = [r['phase_relative_rad'] for r in result['phases'] if r['coherence'] >= .8]
    if len(trusted) >= 3:
        result['coherent_phase_span_rad'] = float(np.ptp(np.unwrap(trusted)))
    return result


def assess_hybrid_trace(trace, stats, neuron_ids):
    t = trace['dense_time_s']; use = t >= t[-1] / 2
    voltage = stats['dense_voltage_mV']
    named = voltage['named_groups']
    motor_ids = set(named['numbered_body_motor_names']['repeating_neurons']) | set(named['RMD_SMD_SMB_names']['repeating_neurons'])
    profiles = {}
    for name in sorted(motor_ids):
        j = neuron_ids.index(name)
        profiles[name] = dict(amplitude_mV=voltage['minimum_window_detrended_p2p'][j],
                              frequency_hz=peak_frequency(t[use], trace['dense_voltage_mV'][use, j], .01))
    regions = regional_amplitudes(trace['dense_joints'], trace['dense_filtered_drive'])
    amp = np.asarray(regions['joint_sustained_p2p_rad'])
    coverage = [int(np.sum(amp[g] >= 1e-5)) for g in np.array_split(np.arange(11), 3)]
    return dict(regions=regions, repeated_joints_per_region=coverage, motor_profiles=profiles,
                cycle_excursions=cycle_excursions(trace['dense_joints']),
                rhythm_retained=bool(profiles and all(n > 0 for n in coverage)),
                coordination=coordination_statistics(t, trace['dense_joints']),
                unphased_late_net_displacement_body_lengths=float(np.linalg.norm(
                    trace['dense_position'][-1] - trace['dense_position'][np.flatnonzero(use)[0]]) / .001),
                displacement_scope='Unphased endpoint difference; no locomotion or gait-closure claim')


def validation_checks(results):
    base = results['hybrid-sham']['hybrid_assessment']
    profiles = base['motor_profiles']
    reference = max(profiles, key=lambda n: profiles[n]['amplitude_mV']) if profiles else None
    active_names = [n for n in results if n != 'hybrid-held-voltage']
    retention = all(results[n]['hybrid_assessment']['rhythm_retained']
                    and reference in results[n]['hybrid_assessment']['motor_profiles'] for n in active_names)
    refinement = []
    for name in ('hybrid-neural-half', 'hybrid-body-half'):
        fine = results[name]['hybrid_assessment']
        a = profiles.get(reference, {}); b = fine['motor_profiles'].get(reference, {})
        amplitudes = bool(a and b and abs(a['amplitude_mV'] - b['amplitude_mV']) <= .2 * max(.1, a['amplitude_mV']))
        frequencies = bool(a.get('frequency_hz') and b.get('frequency_hz') and
                           abs(a['frequency_hz'] - b['frequency_hz']) <= .1 * a['frequency_hz'])
        x = np.asarray(base['regions']['regional_joint_median_p2p_rad'])
        y = np.asarray(fine['regions']['regional_joint_median_p2p_rad'])
        bends = bool(np.all(abs(x-y) <= .2 * np.maximum(x, 1e-5)))
        refinement.append(dict(case=name, same_motor_amplitude_consistent=amplitudes,
                               same_motor_frequency_consistent=frequencies, regional_bends_consistent=bends,
                               passed=amplitudes and frequencies and bends))
    held = results['hybrid-held-voltage']['late_repeating_by_stage']
    negative = held['dense_voltage_mV'] == 0 and held['dense_joints'] == 0
    return dict(reference_motor_neuron=reference, all_active_retain_rhythm=bool(retention),
                refinement=refinement, held_voltage_control_passed=negative,
                rhythm_validation_pass=bool(retention and negative and all(r['passed'] for r in refinement)),
                baseline_historical_regional_amplitude_pass=base['regions']['all_three_regional_amplitude_targets_met'],
                scope='Prospective engineering screening; no certified error, biological fidelity, locomotion or automatic fitting promotion')
