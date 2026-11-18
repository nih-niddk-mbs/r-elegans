"""E095 single-edge strength dose-response; the widened upper bounds are an engineering prior."""
import numpy as np

FACTORS = (.5, .75, 1.5, 2., 3., 4.)
MAX_FACTOR = 4.
TARGET = .01
FINALISTS = 2
# (presynaptic, postsynaptic) existing chemical edges on the cue-to-motor route.
EDGES = (('AWAR', 'AIYR'), ('AIBR', 'AIYR'), ('AWAL', 'AIYL'), ('AIYR', 'RIAR'),
         ('AIZR', 'RIAR'), ('AIZL', 'RIAL'), ('RIAL', 'SMDVL'), ('RIAL', 'SMDVR'))


def edge_coordinate(spec, ids, pre, post):
    ei = np.nonzero(spec.chemical)
    ix = np.flatnonzero((ei[0] == ids.index(post)) & (ei[1] == ids.index(pre)))
    if len(ix) != 1:
        raise ValueError(f'Exactly one existing {pre}->{post} chemical coordinate required')
    return int(np.count_nonzero(spec.channels)) + int(ix[0])


def scan_groups(spec, ids, origin, lower, upper):
    groups = []
    for pre, post in EDGES:
        c = edge_coordinate(spec, ids, pre, post)
        if origin[c] > upper[c] + 1e-10 or origin[c] < lower[c] - 1e-10:
            raise ValueError('Parent outside original prior')
        down = min(float(origin[c] - lower[c]), float(np.log(1 / FACTORS[0])))
        groups.append(dict(name=f'chemical:{pre}->{post}', indices=[c],
                           lower=-max(down, 0.), upper=float(np.log(MAX_FACTOR))))
    return groups


def extend_upper(upper, origin, groups):
    allowed = np.array(upper, copy=True)
    for g in groups:
        c = g['indices'][0]
        allowed[c] = max(allowed[c], origin[c] + g['upper'])
    return allowed


def planned_cases(groups):
    """Every edge x factor; factors below the original lower bound are skipped, not widened."""
    cases = []
    for i, g in enumerate(groups):
        for factor in FACTORS:
            delta = float(np.log(factor))
            if delta >= 0:
                runnable, value = delta <= g['upper'] + 1e-12, delta / g['upper']
            else:
                runnable = g['lower'] < 0 and -delta <= -g['lower'] + 1e-12
                value = delta / -g['lower'] if g['lower'] < 0 else None
            z = np.zeros(len(groups))
            if runnable:
                z[i] = min(1., max(-1., value))
            cases.append(dict(edge=g['name'], factor=factor, group=i, runnable=bool(runnable), z=z.tolist()))
    return cases


def reduction_percent(metrics):
    return float(100 * (1 - max(max(r) for r in metrics['regional_motion_ratios'])))


def eligible(row):
    m = row.get('metrics', {})
    return bool(row.get('status') == 'completed' and m.get('safe') and m.get('baseline_and_recovery_preserved'))


def finalists(rows, k=FINALISTS):
    """Largest weakest-region reduction among cases that keep baseline gait and recovery."""
    good = [r for r in rows if eligible(r)]
    return sorted(good, key=lambda r: (-reduction_percent(r['metrics']), r['output']))[:k]
