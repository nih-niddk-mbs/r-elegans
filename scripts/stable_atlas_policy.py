"""E113 NumPy-only parameter policy and supervised selection rules."""
import numpy as np

BOUND = .1


class Coordinates:
    def __init__(self, arrays):
        self.origin = arrays
        ids = arrays['neuron_ids'].tolist()
        mask = arrays['cells_conductances_nS'] > 0
        mask[:, 16] = False  # LEAK, fixed reference
        mask[[i for i, n in enumerate(ids) if n.startswith('RMD')]] = False
        self.channel = np.where(mask)
        self.chemical = np.where(arrays['fixed_chemical_mask'] > 0)
        self.gap = np.where(np.triu(arrays['fixed_gap_mask'] > 0, 1))
        self.sizes = [len(ix[0]) for ix in (self.channel, self.chemical, self.gap)]
        self.size = sum(self.sizes)

    def apply(self, offsets):
        x = np.asarray(offsets, dtype=np.float64)
        if x.shape != (self.size,) or not np.isfinite(x).all() or np.max(abs(x)) > BOUND + 1e-12:
            raise ValueError('Invalid bounded coordinate vector')
        out = {k: v.copy() for k, v in self.origin.items()}
        parts = np.split(x, np.cumsum(self.sizes)[:-1])
        for key, ix, delta in zip(('params_log_channel_scale', 'params_raw_chemical', 'params_raw_gap'),
                                  (self.channel, self.chemical, self.gap), parts):
            out[key] = out[key].astype(np.float64)
            out[key][ix] += delta
            if key == 'params_raw_gap':out[key][ix[::-1]] += delta
        return out

    def audit(self, arrays, offsets):
        expected = self.apply(offsets)
        failures = [k for k in expected if k not in arrays or not np.array_equal(expected[k], arrays[k])]
        failures += sorted(set(arrays) - set(expected))
        return dict(passed=not failures, mismatched_fields=failures, coordinates=self.size,
                    changed_coordinates=int(np.count_nonzero(offsets)),
                    maximum_offset=float(np.max(abs(offsets))),
                    near_bound=int(np.sum(abs(offsets) >= .99 * BOUND)),
                    fixed_fields_exact=True if not failures else False,
                    channel_chemical_gap_coordinates=self.sizes)


def loss(response, targets, ids, statistics, split):
    """Same pooled/per-target clipped-z Pearson loss as train_atlas_gradient."""
    r = np.asarray(response, dtype=np.float64)
    if r.ndim == 3:r = r.mean(axis=0)
    z = np.zeros_like(r);w = np.zeros_like(r)
    s = statistics['stats'][split];lookup = {t: i for i, t in enumerate(targets)}
    for pair, value in zip(s['pairs'], s['mean_z']):
        t, receiver = pair
        if t in lookup and t != receiver:
            i, j = lookup[t], ids.index(receiver);z[i, j] = np.clip(value, -5, 5);w[i, j] = 1
    if not w.any():raise ValueError('No measured downstream labels')
    n = w.sum();a = (r - (r*w).sum()/n)*w;b = (z - (z*w).sum()/n)*w
    pooled = (a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-30)
    count = np.maximum(w.sum(axis=1), 1)
    a = (r-((r*w).sum(axis=1)/count)[:, None])*w
    b = (z-((z*w).sum(axis=1)/count)[:, None])*w
    per = (a*b).sum(axis=1)/np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1)+1e-30)
    valid = w.sum(axis=1) >= 5
    if not valid.any():raise ValueError('At least one target with five measured pairs required')
    return float(1-.5*(pooled+(per*valid).sum()/valid.sum()))


def prior(x):return float(.001*np.mean((np.asarray(x)/BOUND)**2))


def adam_step(x, m, v, gradient, iteration):
    gradient = np.asarray(gradient, dtype=float)
    gradient = gradient/max(1., float(np.linalg.norm(gradient)))
    m = .9*m+.1*gradient;v = .999*v+.001*gradient**2
    step = .002*(m/(1-.9**iteration))/(np.sqrt(v/(1-.999**iteration))+1e-8)
    return np.clip(x-step, -BOUND, BOUND), m, v


def eligible_targets(statistics, split):
    counts = {}
    for t, r in statistics['stats'][split]['pairs']:
        if t != r:counts.setdefault(t, set()).add(r)
    return sorted(t for t, receivers in counts.items() if len(receivers) >= 5)


def choose_batch(rng, pool, hard, iteration):
    if len(pool) < 2:raise ValueError('Two eligible training targets required')
    fine = sorted(set(pool) & set(hard))
    if iteration % 5 == 0 and fine:
        first = str(rng.choice(fine));second = str(rng.choice([t for t in pool if t != first]))
        return sorted([first, second])
    return sorted(rng.choice(pool, 2, replace=False).tolist())


def preserved(reference, candidate):
    if not reference.get('passed') or not candidate.get('passed'):return False
    p, q = reference['median_period_s'], candidate['median_period_s']
    a, b = np.mean(reference['window_amplitudes_mV']), np.mean(candidate['window_amplitudes_mV'])
    return bool(p and abs(q/p-1) <= .1 and abs(b/a-1) <= .25)


def admit(now, deadline, recent_seconds):
    return deadline-now > max(120., 1.3*recent_seconds)
