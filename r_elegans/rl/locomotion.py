"""Phase-matched locomotion diagnostics for normalized, externally paced worms."""
import numpy as np


def cycle_metrics(position, heading, angles, *, period_steps, warmup_cycles=4,
                  measured_cycles=6, body_length=1.):
    """States include t=0; discard startup and compare identical command phases.

    Heading in mechanics is the first rod's angle, not the animal's axis.
    Project displacement on the actual tail-to-head segment-center axis.
    No position is taken from a moving head tip.
    """
    position, heading, angles = map(lambda x: np.asarray(x, dtype=float), (position, heading, angles))
    final = (warmup_cycles + measured_cycles) * period_steps
    if (period_steps < 2 or warmup_cycles < 1 or measured_cycles < 3 or body_length <= 0
            or len(position) <= final or len(heading) != len(position) or len(angles) != len(position)):
        raise ValueError("Need initial state and at least three complete measured cycles after warmup")
    if not all(np.all(np.isfinite(x)) for x in (position, heading, angles)):
        raise ValueError("Nonfinite body trajectory")
    endpoints = np.arange(warmup_cycles, warmup_cycles + measured_cycles + 1) * period_steps
    q = angles[endpoints]
    orientations = np.concatenate([np.zeros((len(q), 1)), np.cumsum(q, axis=1)], axis=1)
    weights = np.ones(orientations.shape[1])
    weights[[0, -1]] = .5
    axis = heading[endpoints] + np.arctan2(np.sin(orientations) @ weights, np.cos(orientations) @ weights)
    displacement = np.diff(position[endpoints], axis=0) / body_length
    forward = displacement[:, 0] * np.cos(axis[:-1]) + displacement[:, 1] * np.sin(axis[:-1])
    lateral = -displacement[:, 0] * np.sin(axis[:-1]) + displacement[:, 1] * np.cos(axis[:-1])
    window = angles[endpoints[0]:endpoints[-1] + 1]
    oscillation = np.sqrt(np.mean((window - window.mean(axis=0)) ** 2))
    closure = np.sqrt(np.mean(np.diff(q, axis=0) ** 2, axis=1)) / max(oscillation, 1e-6)
    yaw = np.arctan2(np.sin(np.diff(axis)), np.cos(np.diff(axis)))
    mean = float(np.mean(forward))
    return dict(forward_per_cycle=forward.tolist(), lateral_per_cycle=lateral.tolist(),
        mean_forward_body_lengths=mean, mean_lateral_body_lengths=float(np.mean(lateral)),
        std_forward_body_lengths=float(np.std(forward)),
        direction_agreement=float(np.mean(forward * mean > 0)),
        max_shape_closure_relative=float(np.max(closure)),
        mean_abs_yaw_per_cycle=float(np.mean(np.abs(yaw))),
        total_displacement_body_lengths=((position[endpoints[-1]] - position[endpoints[0]]) / body_length).tolist(),
        warmup_cycles=warmup_cycles, measured_cycles=measured_cycles)


def assess_propulsion(coarse, fine, rest, *, isotropic=None, minimum_progress=1e-3):
    """Separate detectable translation from the stronger useful-motion gate.

    Fine dt is half coarse dt. Numerical/rest floors and cycle consistency
    prevent startup relaxation or integration drift from counting as swimming.
    Thresholds are engineering assay criteria, not biological measurements.
    """
    a, b = coarse["mean_forward_body_lengths"], fine["mean_forward_body_lengths"]
    numerical_error = abs(a - b)
    rest_drift = abs(rest["mean_forward_body_lengths"])
    isotropic_drift = abs(isotropic["mean_forward_body_lengths"]) if isotropic else 0.
    periodic = max(coarse["max_shape_closure_relative"], fine["max_shape_closure_relative"]) < .1
    stable = (fine["direction_agreement"] >= .8
              and fine["std_forward_body_lengths"] <= .2 * abs(b) + 1e-9)
    detectable = bool(periodic and stable and a*b > 0
        and abs(b) > max(1e-6, 5*rest_drift, 5*isotropic_drift, 3*numerical_error))
    straight = (fine["mean_abs_yaw_per_cycle"] < .05
                and abs(fine["mean_lateral_body_lengths"]) < .5 * abs(b) + 1e-9)
    ready = bool(detectable and straight and abs(b) >= minimum_progress)
    return dict(passed=ready, detectable_translation=detectable, periodic_shape=bool(periodic),
        consistent_cycles=bool(stable), approximately_straight=bool(straight),
        direction=("forward" if b > 0 else "reverse") if detectable else "unresolved",
        refined_forward_body_lengths_per_cycle=b, timestep_difference=numerical_error,
        rest_drift_body_lengths_per_cycle=rest_drift,
        isotropic_drift_body_lengths_per_cycle=isotropic_drift, minimum_useful_progress=minimum_progress)
