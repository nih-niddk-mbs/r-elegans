"""Prospective E109 downstream-pair checks; no self-response normalization."""
import numpy as np
from scripts.selected_atlas_audit_checks import compare_responses


def observed_pairs(statistics, targets, ids):
    pairs = sorted({tuple(p) for split in statistics['stats'].values() for p in split['pairs'] if p[0] != p[1]})
    if not pairs or len(pairs) != len(set(pairs)):
        raise ValueError('Nonempty unique downstream support required')
    if any(t not in targets or r not in ids for t, r in pairs):
        raise ValueError('Observed pair coverage gap; refusing silent exclusion')
    return pairs, np.asarray([targets.index(t) for t, _ in pairs]), np.asarray([ids.index(r) for _, r in pairs])


def compare_downstream(a, b, pairs):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.shape != (len(pairs),) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite observed-pair vectors required')
    error = abs(a - b)
    tolerance = 1e-7 + .1 * np.maximum(abs(a), abs(b))
    ratios = error / tolerance
    active = np.maximum(abs(a), abs(b)) >= 1e-6
    flips = active & (a * b < 0)
    worst = np.argsort(-ratios)[:10]
    summary = dict(vector=compare_responses(a, b), pair_count=len(pairs),
        mixed_tolerance_passed=bool(np.all(ratios <= 1)), mixed_failures=int((ratios > 1).sum()),
        mixed_pass_fraction=float(np.mean(ratios <= 1)), maximum_tolerance_ratio=float(ratios.max()),
        modeled_active_pairs=int(active.sum()), active_sign_flips=int(flips.sum()),
        maximum_downstream_response=float(max(abs(a).max(), abs(b).max())),
        worst_pairs=[dict(target=pairs[i][0], receiver=pairs[i][1], coarse=float(a[i]), fine=float(b[i]),
                          abs_difference=float(error[i]), tolerance=float(tolerance[i]), tolerance_ratio=float(ratios[i])) for i in worst],
        absolute_tolerance=1e-7, relative_tolerance=.1,
        scope='Mixed tolerance is a prospective engineering resolution convention, not biological noise; vector check remains separate')
    return summary, ratios, flips
