"""E086 sensory control of RIAL release variation; not body stopping."""
import numpy as np
from scipy.signal import find_peaks
from scipy.special import expit
from . import rial_sensory
from .hybrid_assessment import cycle_excursions

RUNGS = (.01, .02, .05)


def interpolated_frequency(times, signal, prominence):
    t, y = np.asarray(times), np.asarray(signal)
    peaks = find_peaks(y, prominence=prominence)[0]
    peaks = peaks[(peaks > 0) & (peaks < len(y)-1)]
    if len(peaks) < 3:
        return None
    denom = y[peaks-1]-2*y[peaks]+y[peaks+1]
    shift = np.divide(.5*(y[peaks-1]-y[peaks+1]), denom,
                      out=np.zeros(len(peaks)), where=denom != 0)
    refined = t[peaks]+np.clip(shift, -.5, .5)*np.median(np.diff(t))
    return float(1/np.median(np.diff(refined)))


def features(trace, ids, variant, threshold, slope):
    result = rial_sensory.features(trace, ids, variant,
                                   frequency_estimator=interpolated_frequency)
    ri = ids.index('RIAL')
    for category in ('active', 'response', 'late'):
        for row in result[category]:
            a, b = row['window']
            use = (trace['dense_time_s'] >= a) & (trace['dense_time_s'] < b)
            release = expit((trace['dense_voltage_mV'][use, ri]-threshold)/slope)
            row['RIAL_release_std'] = float(release.std())
            row['RIAL_release_cycle_excursion'] = float(cycle_excursions(np.tile(release[:,None], (1,11)), late_only=False)['joint_cycle_excursion_rad'][0])
            row['RIAL_release_mean'] = float(release.mean())
    return result


def score(paired, parent, target, margin=1.):
    body = rial_sensory.score(paired, parent, target, margin)
    a, b = paired['no_cue'], paired['cue']
    body_ratios = body['response_amplitude_ratios']
    old_error = np.mean([((r-(1-target*margin))/target)**2 for r in body_ratios])
    baseline = parent['active'][0]['RIAL_release_cycle_excursion']
    penalty = 0.
    preserved = True
    for row in a['active']+a['response']+a['late']:
        ratio = row['RIAL_release_cycle_excursion']/max(baseline, 1e-9)
        preserved = preserved and .9 <= ratio <= 1.1 and row['RIAL_release_cycle_excursion'] >= .01
        penalty += (max(0., .9-ratio)/.1)**2+(max(0., ratio-1.1)/.1)**2
        penalty += (max(0., .01-row['RIAL_release_cycle_excursion'])/.01)**2
    for row, ref in zip(b['active'], a['active']):
        ratio = row['RIAL_release_cycle_excursion']/max(ref['RIAL_release_cycle_excursion'], 1e-9)
        preserved = preserved and .9 <= ratio <= 1.1
        penalty += (max(0., .9-ratio)/.1)**2+(max(0., ratio-1.1)/.1)**2
    ratios = [x['RIAL_release_cycle_excursion']/max(y['RIAL_release_cycle_excursion'], 1e-9)
              for x, y in zip(b['response'], a['response'])]
    error = np.mean([((r-(1-target*margin))/target)**2 for r in ratios])
    body.update(loss=float(max(0., body['loss']-old_error)+error+100*penalty),
                body_response_amplitude_ratios=body_ratios,
                response_amplitude_ratios=ratios,
                late_amplitude_ratios=[x['RIAL_release_cycle_excursion']/max(y['RIAL_release_cycle_excursion'], 1e-9)
                                       for x,y in zip(b['late'], a['late'])],
                metric='RIAL_release_cycle_excursion_ratio',
                RIAL_baseline_and_recovery_preserved=bool(preserved))
    body['baseline_and_recovery_preserved'] = bool(body['baseline_and_recovery_preserved'] and preserved)
    eligible = body['safe'] and body['baseline_and_recovery_preserved'] and body['useful_response_preserved']
    body['partial_attenuation_passed'] = bool(eligible and all(r <= 1-target for r in ratios))
    body['training_margin_met'] = bool(eligible and all(r <= 1-target*margin for r in ratios))
    return body
