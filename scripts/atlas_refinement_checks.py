"""E110 failure-selection and saved-layer checks; no fitting or solver calls."""
import math
import numpy as np
from scripts.atlas_shared_checks import observed_pairs


def choose_targets(statistics, ids, masks, source_audit):
    pairs, _, _ = observed_pairs(statistics, statistics['targets'], ids)
    targets, details = set(), {}
    for label in ('selected', 'parent'):
        data = masks[label]
        if data['pairs'].tolist() != [list(p) for p in pairs]:
            raise ValueError('Source pair-mask identities differ from cached observed support')
        ratios, flips = data['tolerance_ratios'], data['active_sign_flips']
        if ratios.shape != (3, 5, len(pairs)) or flips.shape != ratios.shape or flips.dtype != np.bool_ or not np.isfinite(ratios).all() or np.any(ratios < 0):
            raise ValueError('Invalid source failure mask')
        if len(source_audit['models'][label]['comparisons']) != 3:
            raise ValueError('All source comparisons required')
        for i, row in enumerate(source_audit['models'][label]['comparisons']):
            if row['levels'] != [i, i + 1] or len(row['per_phase']) != 4:raise ValueError('Unexpected source comparison ordering')
            for phase, check in enumerate(row['per_phase'] + [row['phase_mean']]):
                if check['mixed_failures'] != int((ratios[i, phase] > 1).sum()) or check['active_sign_flips'] != int(flips[i, phase].sum()) or not np.isclose(check['maximum_tolerance_ratio'], ratios[i, phase].max(), rtol=1e-12, atol=1e-12):
                    raise ValueError('Source mask disagrees with source audit')
        rows = []
        for k, (target, receiver) in enumerate(pairs):
            use = (ratios[1:, :, k] > 1) | flips[1:, :, k]
            if use.any():
                targets.add(target)
                locations = [dict(source_levels=[int(i) + 1, int(i) + 2], phase=int(phase) if phase < 4 else 'mean',
                                  tolerance_ratio=float(ratios[i + 1, phase, k]), active_sign_flip=bool(flips[i + 1, phase, k]))
                             for i, phase in np.argwhere(use)]
                rows.append(dict(target=target, receiver=receiver, locations=locations))
        details[label] = rows
    if not targets:raise ValueError('No remaining failed targets; targeted follow-up unnecessary')
    targets.update(('ADAL', 'AIYL', 'AIYR'))
    if not targets.issubset(statistics['targets']):raise ValueError('Fixed target/control absent from original atlas')
    return sorted(targets), details


def layer_values(data, targets, ids, step_ms, anchor_sha):
    if data['targets'].tolist() != targets or str(data['anchors_sha256']) != anchor_sha or float(data['neural_step_ms']) != step_ms or float(data['sample_dt_s']) != .001:
        raise ValueError('Layer target/grid/anchor metadata mismatch')
    count = 2 * math.ceil(len(targets) / 8) + (-len(targets)) % 8
    means, cm, delta, base = (data[k] for k in ('indicator_means', 'control_means', 'control_voltage_delta', 'reference_sham_rmd_mV'))
    if means.shape != (4, len(targets), len(ids)) or cm.shape != (count, 4, len(ids)) or delta.shape != (count, 4, 12000, 2) or base.shape != (4, 12000, 2):
        raise ValueError('Layer pulse/control coverage incomplete')
    labels = [f'batch-{i // 8}-condition-{k}' for i in range(0, len(targets), 8) for k in range(min(8, len(targets) - i), 10)]
    if data['control_labels'].tolist() != labels:raise ValueError('Layer control labels differ')
    if any(x.dtype != np.float64 or not np.isfinite(x).all() for x in (means, cm, delta, base)):
        raise ValueError('Nonfinite or non-float64 layer')
    error, trace_error = float(abs(cm - cm[0]).max()), float(abs(delta).max())
    return means - cm[0][:, None], dict(duplicate_mean_error=error, duplicate_trace_error=trace_error,
                                       controls_passed=bool(error <= 1e-12 and trace_error <= 1e-12))
