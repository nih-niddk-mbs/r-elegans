"""E098 cue-to-RIA-to-motor pathway parameter groups; widened edge bounds are an engineering prior."""
import numpy as np

SOURCES = ('AWAL', 'AWAR')
HOP1 = ('AIZ', 'AIY', 'ADF', 'ADA', 'AFD', 'ASH', 'RIF', 'AIB')
NODES = ('RIAL', 'RIAR')
DENSITY_NEURONS = ('AIZL', 'AIZR', 'RIAL', 'RIAR', 'SMDDL', 'SMDDR', 'SMDVL', 'SMDVR')
MOTOR_PREFIXES = ('SMD', 'RMD')
MAX_FACTOR = 4.
NODE_TARGET = .02
INWARD = [6, 7, 8, 14]
OUTWARD = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 15]


def _hop1_names(ids):
    names = [n + side for n in HOP1 for side in ('L', 'R')]
    missing = [n for n in (*SOURCES, *NODES, *DENSITY_NEURONS, *names) if n not in ids]
    if missing:
        raise ValueError('Pathway neurons missing from the network: ' + ', '.join(missing))
    return names


def pathway_edges(spec, ids):
    """Existing chemical edges (pre, post): AWA to hop-1 interneurons that reach RIA, those to RIA, and RIA to motor targets."""
    mask = np.asarray(spec.chemical, dtype=bool)
    hop1 = _hop1_names(ids)
    index = {n: i for i, n in enumerate(ids)}
    reaches_ria = [h for h in hop1 if any(mask[index[r], index[h]] for r in NODES)]
    edges = []
    for h in reaches_ria:
        edges += [(s, h) for s in SOURCES if mask[index[h], index[s]]]
        edges += [(h, r) for r in NODES if mask[index[r], index[h]]]
    for r in NODES:
        edges += [(r, n) for n in ids if n.startswith(MOTOR_PREFIXES) and mask[index[n], index[r]]]
    return [e for e in dict.fromkeys(edges)]


def hop1_with_edges(spec, ids):
    """Interneurons on trained AWA->h->RIA paths, in a stable order."""
    pre = {e[0] for e in pathway_edges(spec, ids)}
    return [h for h in _hop1_names(ids) if h in pre]


def edge_coordinate(spec, ids, pre, post):
    ei = np.nonzero(spec.chemical)
    ix = np.flatnonzero((ei[0] == ids.index(post)) & (ei[1] == ids.index(pre)))
    if len(ix) != 1:
        raise ValueError(f'Exactly one existing {pre}->{post} chemical coordinate required')
    return int(np.count_nonzero(spec.channels)) + int(ix[0])


def pathway_groups(spec, ids, origin, lower, upper):
    groups = []
    ci = np.nonzero(spec.channels)

    def density(name, indices):
        ix = np.asarray(indices, dtype=int)
        if not len(ix):
            return
        lo = max(-.4, float(np.max(lower[ix] - origin[ix])))
        hi = min(.4, float(np.min(upper[ix] - origin[ix])))
        if lo > 1e-10 or hi < -1e-10:
            raise ValueError('Parent outside declared prior')
        if hi - lo > 1e-8:
            groups.append(dict(name=name, indices=ix.tolist(), lower=min(lo, 0.), upper=max(hi, 0.)))
    for pre, post in pathway_edges(spec, ids):
        c = edge_coordinate(spec, ids, pre, post)
        if origin[c] > upper[c] + 1e-10 or origin[c] < lower[c] - 1e-10:
            # Inherited exceptions (E092) stay frozen rather than becoming trainable.
            continue
        down = min(float(origin[c] - lower[c]), float(np.log(MAX_FACTOR)))
        groups.append(dict(name=f'chemical:{pre}->{post}', indices=[c], lower=-max(down, 0.), upper=float(np.log(MAX_FACTOR)), widened=True))
    for name in DENSITY_NEURONS:
        for label, columns in (('inward', INWARD), ('outward', OUTWARD)):
            density(f'density:{name}:{label}', np.flatnonzero((ci[0] == ids.index(name)) & np.isin(ci[1], columns)))
    used = [i for g in groups for i in g['indices']]
    if len(used) != len(set(used)):
        raise ValueError('Overlapping pathway groups')
    return groups


def extend_pathway_upper(upper, origin, groups):
    allowed = np.array(upper, copy=True)
    for g in groups:
        if g.get('widened'):
            c = g['indices'][0]
            allowed[c] = max(allowed[c], origin[c] + g['upper'])
    return allowed


def widened_coordinates(groups):
    return {g['indices'][0] for g in groups if g.get('widened')}
