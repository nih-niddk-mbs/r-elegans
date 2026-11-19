"""E096 imposed membrane-conductance dose-response at single nodes; diagnostic only, never a learned control."""
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


def cases():
    return [dict(name=f'{group}-{label}-{dose:g}', group=group, conductance_nS=dose, reversal_mV=reversal)
            for group in GROUPS for label, reversal in SIGNS for dose in DOSES]


def membrane_probe(ids, case, window):
    import jax.numpy as jnp
    from .brain.conductance_network import MembraneConductanceInput
    strength, reversal = float(case['conductance_nS']), float(case['reversal_mV'])
    if case['group'] not in GROUPS or not np.isfinite([strength, reversal]).all() or not 0 < strength <= MAX_DOSE or reversal not in (0., -80.):
        raise ValueError('Probe outside declared diagnostic envelope')
    mask = np.zeros(len(ids))
    for name in GROUPS[case['group']]:
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
        if row.get('status') == 'completed' and row['target_reached']:
            key = f"{row['group']}-{row['sign']}"
            result[key] = min(result.get(key, np.inf), row['dose_nS'])
    return result
