"""Continuous-state stop/restart commands and body-level acceptance criteria."""
import numpy as np

from .locomotion import cycle_metrics, assess_propulsion


def stop_restart_commands(period_steps, *, warmup_cycles=3, measured_cycles=3):
    """Three equal blocks: sine, zero, sine. No neural/body state resets."""
    if period_steps < 2 or warmup_cycles < 1 or measured_cycles < 3:
        raise ValueError("Need warmup and at least three measured cycles per block")
    block = (warmup_cycles + measured_cycles) * period_steps
    sine = .85 * np.sin(2 * np.pi * np.arange(block) / period_steps)
    return np.concatenate([sine, np.zeros(block), sine])


def stop_restart_metrics(position, heading, angles, *, period_steps, dt,
                         warmup_cycles=3, measured_cycles=3):
    """Measure late windows, excluding transitions, without hiding reciprocal motion."""
    position, heading, angles = map(np.asarray, (position, heading, angles))
    block = (warmup_cycles + measured_cycles) * period_steps
    if len(position) != 3 * block + 1 or dt <= 0:
        raise ValueError("Need initial state and three complete blocks, with positive dt")
    reports = {}
    for index, label in enumerate(("go", "stop", "restart")):
        start, end = index * block, (index + 1) * block
        metrics = cycle_metrics(position[start:end+1], heading[start:end+1], angles[start:end+1],
            period_steps=period_steps, warmup_cycles=warmup_cycles, measured_cycles=measured_cycles)
        first = start + warmup_cycles * period_steps
        metrics["centroid_path_per_cycle"] = float(np.linalg.norm(
            np.diff(position[first:end+1], axis=0), axis=1).sum() / measured_cycles)
        metrics["shape_speed_rms"] = float(np.sqrt(np.mean(
            (np.diff(angles[first:end+1], axis=0) / dt) ** 2)))
        reports[label] = metrics
    return reports


def assess_stop_restart(coarse, fine, isotropic):
    """Keep the propulsion floor; require <=10% residual motion and >=80% recovery.

    The stop window starts three pacing cycles after command removal. These
    are engineering gates, not a biological stopping latency claim.
    """
    phases = {label: assess_propulsion(coarse[label], fine[label], fine["stop"],
        isotropic=isotropic[label]) for label in ("go", "restart")}
    ratios = {}
    for name, report in (("coarse", coarse), ("fine", fine)):
        ratios[name] = dict(
            stop_path_ratio=report["stop"]["centroid_path_per_cycle"] /
                max(report["go"]["centroid_path_per_cycle"], 1e-12),
            stop_shape_speed_ratio=report["stop"]["shape_speed_rms"] /
                max(report["go"]["shape_speed_rms"], 1e-12),
            restart_progress_ratio=report["restart"]["mean_forward_body_lengths"] /
                max(report["go"]["mean_forward_body_lengths"], 1e-12))
    stopped = all(r["stop_path_ratio"] <= .1 and r["stop_shape_speed_ratio"] <= .1
                  for r in ratios.values())
    recovered = all(.8 <= r["restart_progress_ratio"] <= 1.2 for r in ratios.values())
    forward = all(p["detectable_translation"] and p["approximately_straight"]
                  and p["direction"] == "forward" for p in phases.values())
    control_passed = bool(stopped and recovered and forward)
    return dict(passed=bool(control_passed and all(p["passed"] for p in phases.values())),
        control_passed=control_passed, stopped=bool(stopped), recovered=bool(recovered),
        ratios=ratios, propulsion=phases)
