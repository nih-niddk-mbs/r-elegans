"""Connection-only proposals and raw-cycle amplitude scoring; no pacing."""
import numpy as np
from scipy.signal import find_peaks

from .hybrid_assessment import cycle_excursions, peak_frequency

TARGETS = np.array([1e-4, 3e-4, 3e-4])


def connection_coordinates(anchor, params, spec):
    chemical = np.nonzero(spec.chemical)
    gap = np.nonzero(np.triu(spec.gap, 1))
    def log_strength(raw):
        strength = np.logaddexp(0., np.asarray(raw, dtype=float))
        if np.any(strength <= 0) or not np.isfinite(strength).all():
            raise ValueError('Finite nonzero effective connection strengths required')
        return np.log(strength)
    a = log_strength(np.asarray(params.raw_chemical)[chemical]) - log_strength(np.asarray(anchor.raw_chemical)[chemical])
    def symmetric(raw):
        x = np.asarray(raw, dtype=float)
        return .5 * (x[gap] + x.T[gap])
    b = log_strength(symmetric(params.raw_gap)) - log_strength(symmetric(anchor.raw_gap))
    return np.r_[a, b]


def inherited_bounds(origin, factor=2., tolerance=2e-6):
    limit = np.log(factor)
    excess = max(0., float(np.max(np.abs(origin))) - limit)
    if not np.isfinite(origin).all() or excess > tolerance:
        raise ValueError('Parent exceeds original-anchor effective-strength bounds')
    return np.minimum(-limit, origin), np.maximum(limit, origin), excess


def propose_connections(best, spec, rng, sigma, lower, upper, focus_cells=()):
    rows, cols = np.nonzero(spec.chemical); gi, gj = np.nonzero(np.triu(spec.gap, 1))
    b = len(rows); size = len(best); noise = np.zeros(size)
    mode = int(rng.integers(4)); cells = []
    if mode in (1, 2):
        pool = np.asarray(focus_cells) if len(focus_cells) and rng.random() < .6 else np.arange(spec.chemical.shape[0])
        cells = rng.choice(pool, size=min(2, len(pool)), replace=False)
        selected = np.flatnonzero(np.r_[np.isin(rows, cells) | np.isin(cols, cells),
                                       np.isin(gi, cells) | np.isin(gj, cells)])
    else:
        pool = np.arange(size)
        if mode == 3:
            pool = np.arange(b) if rng.random() < .5 else np.arange(b, size)
        selected = rng.choice(pool, size=min(128, len(pool)), replace=False)
    noise[selected] = rng.normal() if mode in (2, 3) else rng.normal(size=len(selected))
    candidate = np.clip(best + sigma * noise, lower, upper)
    return candidate, dict(mode=mode, cells=np.asarray(cells).tolist(), proposed_coordinates=len(selected))


def amplitude_score(trace, reference_index, *, window_seconds=15., targets=TARGETS, baseline_profiles=None):
    t = np.asarray(trace['dense_time_s']); q = np.asarray(trace['dense_joints'])
    v = np.asarray(trace['dense_voltage_mV'])[:, reference_index]
    if (q.shape != (len(t), 11) or not np.isfinite(q).all() or not np.isfinite(v).all()
            or not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or t[-1] < 2*window_seconds):
        raise ValueError('Finite full trace covering two scoring windows required')
    cycles, profiles, guards = [], [], []
    for number, (a, b) in enumerate(((t[-1]-2*window_seconds, t[-1]-window_seconds),
                                    (t[-1]-window_seconds, t[-1]))):
        use = (t >= a-1e-12) & (t <= b+1e-12)
        signal = v[use]; clock = t[use]
        # Raw excursions, not ranges of a detrended posture.
        cycles.append(cycle_excursions(q[use], late_only=False)['regional_cycle_excursion_rad'])
        pi = find_peaks(signal, prominence=.01)[0]; ti = find_peaks(-signal, prominence=.01)[0]
        peaks, troughs = len(pi), len(ti)
        events = sorted([(int(i), 1) for i in pi]+[(int(i), -1) for i in ti])
        up, down = [], []
        for (i, kind), (j, other) in zip(events[:-1], events[1:]):
            if kind != other:
                delta = float(signal[j]-signal[i])
                (up if delta > 0 else down).append(abs(delta))
        amp = min(float(np.median(up)), float(np.median(down))) if min(len(up),len(down)) >= 2 else 0.
        freq = peak_frequency(clock, signal, .01)
        profile = dict(amplitude_mV=amp, frequency_hz=freq, peaks=peaks, troughs=troughs)
        guard = peaks >= 3 and troughs >= 3 and amp >= .1 and freq is not None
        if baseline_profiles is not None:
            base = baseline_profiles[number]
            guard = guard and amp >= max(.1, .5*base['amplitude_mV']) and .5*base['frequency_hz'] <= freq <= 2*base['frequency_hz']
        profiles.append(profile); guards.append(bool(guard))
    values = np.asarray(cycles)
    deficits = np.maximum(0., np.log(np.asarray(targets) / np.maximum(values, 1e-7)))
    safety = float(np.max(np.abs(q))) < .5
    return dict(data_loss=float(deficits.mean()+deficits.max()), window_cycle_excursions_rad=values.tolist(),
                minimum_cycle_excursions_rad=values.min(axis=0).tolist(), reference_motor_profiles=profiles,
                neural_guard=bool(all(guards)), safety_passed=safety,
                target_passed=bool(np.all(values >= targets) and all(guards) and safety),
                scoring_windows_seconds=[[float(t[-1]-2*window_seconds), float(t[-1]-window_seconds)],
                                         [float(t[-1]-window_seconds), float(t[-1])]])
