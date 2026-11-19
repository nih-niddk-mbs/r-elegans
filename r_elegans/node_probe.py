"""E096/E097 imposed membrane-conductance dose-response at single nodes; diagnostic only, never a learned control."""
import numpy as np

GROUPS = {
    'RIM': ('RIML', 'RIMR'), 'RIB': ('RIBL', 'RIBR'), 'RIS': ('RIS',),
    'AVA': ('AVAL', 'AVAR'), 'AVB': ('AVBL', 'AVBR'), 'PVC': ('PVCL', 'PVCR'),
    'RIA': ('RIAL', 'RIAR'), 'SMD': ('SMDDL', 'SMDDR', 'SMDVL', 'SMDVR'),
    'AIZ': ('AIZL', 'AIZR'), 'AIB': ('AIBL', 'AIBR'),
}
DOSES = (.01, .03, .1)
SIGNS = (('excite', 0.), ('inhibit', -80.))
MAX_DOSE = .1
TARGET = .01
RETAIN_DOSE = .1
EXTRA_GROUPS = {'AIY': ('AIYL', 'AIYR')}
ALL_GROUPS = {**GROUPS, **EXTRA_GROUPS}
CALIBRATION_GROUPS = ('AIZ', 'AIY', 'RIA', 'SMD', 'RIB', 'RIM')
CALIBRATION_DOSES = (.0003, .001, .003, .01)
GRID_CONTROLS = (('AIZ', 'excite', .003), ('AIZ', 'excite', .01))
GRIDS = ('neural-half', 'body-half')
PROFILES = ('e096', 'e097')


def cases(profile='e096'):
    if profile == 'e096':
        return [dict(name=f'{group}-{label}-{dose:g}', group=group, conductance_nS=dose, reversal_mV=reversal)
                for group in GROUPS for label, reversal in SIGNS for dose in DOSES]
    if profile != 'e097':
        raise ValueError('Unknown probe profile')
    reversals = dict(SIGNS)
    result = [dict(name=f'{group}-{label}-{dose:g}', group=group, conductance_nS=dose, reversal_mV=reversal)
              for group in CALIBRATION_GROUPS for label, reversal in SIGNS for dose in CALIBRATION_DOSES]
    for group, label, dose in GRID_CONTROLS:
        for grid in GRIDS:
            result.append(dict(name=f'{group}-{label}-{dose:g}@{grid}', group=group, conductance_nS=dose, reversal_mV=reversals[label], grid=grid))
    return result


def retains_trace(profile, case):
    return True if profile == 'e097' else case['conductance_nS'] == RETAIN_DOSE


def membrane_probe(ids, case, window):
    import jax.numpy as jnp
    from .brain.conductance_network import MembraneConductanceInput
    strength, reversal = float(case['conductance_nS']), float(case['reversal_mV'])
    if case['group'] not in ALL_GROUPS or not np.isfinite([strength, reversal]).all() or not 0 < strength <= MAX_DOSE or reversal not in (0., -80.):
        raise ValueError('Probe outside declared diagnostic envelope')
    mask = np.zeros(len(ids))
    for name in ALL_GROUPS[case['group']]:
        mask[ids.index(name)] = 1.
    mask = jnp.asarray(mask)
    a, b = window

    def probe(t):
        g = mask * strength * jnp.asarray((t >= a) & (t < b))
        return MembraneConductanceInput(g, g * reversal)
    return probe


def reduction_percent(metrics):
    return [float(100 * (1 - r)) for r in metrics['regional_motion_ratios'][0]]


def reaches_target(metrics):
    ratios = np.asarray(metrics['regional_motion_ratios'])
    return bool(metrics['safe'] and metrics['baseline_and_recovery_preserved'] and np.all(ratios <= 1 - TARGET))


def minimum_doses(table):
    """Smallest tested dose per group and sign that reaches the 1% gate in every region."""
    result = {}
    for row in table:
        if row.get('status') == 'completed' and row['target_reached'] and not row.get('grid'):
            key = f"{row['group']}-{row['sign']}"
            result[key] = min(result.get(key, np.inf), row['dose_nS'])
    return result


def predict_at(points, x):
    """Interpolate value at node change x from (abs node change, value) pairs; linear through the origin below the range."""
    pts = sorted((float(a), float(b)) for a, b in points if a > 0)
    if not pts or x <= 0:
        return 0., 'zero'
    xs, ys = np.array([a for a, _ in pts]), np.array([b for _, b in pts])
    if x < xs[0]:
        return float(ys[0] * x / xs[0]), 'extrapolated-below-range'
    if x > xs[-1]:
        return float(ys[-1]), 'above-range'
    return float(np.interp(np.log(x), np.log(xs), ys)), 'interpolated'
