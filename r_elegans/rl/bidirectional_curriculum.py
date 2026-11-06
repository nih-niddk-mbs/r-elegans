"""Broader commands and joint validation for forward/reverse refinement."""
import numpy as np

from .motor_curriculum import traveling_passes


def command_bank(steps=384, *, held_out=False):
    """Independent tuning sequences; final physics assays use separate long runs."""
    if steps < 128:
        raise ValueError("At least 128 samples required")
    rng = np.random.default_rng(421 if held_out else 119)
    t = np.arange(steps) * .02
    periods = (2.3, 3.1, 3.9, 4.5) if held_out else (2.1, 2.7, 3.3, 4.1)
    commands = []
    for i, period in enumerate(periods):
        sine = rng.uniform(.65, .95) * np.sin(2*np.pi*t/period + rng.uniform(-1., 1.))
        sine[:steps//16] = 0
        if i % 2:
            sine[steps//2:steps*3//4] = 0
        commands.append(sine)
    for _ in range(4):
        count = 9 if held_out else 7
        levels = rng.choice([-.95, -.65, -.4, 0., .5, .75, .95], size=count)
        levels[[0, count//2, -1]] = 0
        boundaries = np.r_[0., np.sort(rng.uniform(.08, .92, count-1)), 1.]
        index = np.minimum(np.searchsorted(boundaries[1:], np.arange(steps)/steps, side="right"), count-1)
        commands.append(levels[index])
    return np.asarray(commands, dtype=np.float32)


def validation_key(forward, reverse):
    """Prefer fewer failed cases, then the worst normalized error, then the mean.

    No training loss or final propulsion-assay result participates in selection.
    Both directions must pass the existing wave criteria for a passed fit.
    """
    if not forward or len(forward) != len(reverse):
        raise ValueError("Need matching forward and reverse validation cases")
    failed, errors = 0, []
    for backward, metrics in ((False, forward), (True, reverse)):
        sign = -1 if backward else 1
        for m in metrics:
            if not all(np.isfinite(v) for v in m.values()):
                raise ValueError("Nonfinite validation")
            failed += not traveling_passes(m, reverse=backward)
            errors.extend((m["muscle_rmse"]/.025, m["body_relative_rmse"]/.25,
                (1-m["bend_correlation"])/.1, (1-m["direction_agreement"])/.05,
                abs(m["muscle_lag"]-m["target_muscle_lag"])/.06,
                abs(m["body_lag"]-m["target_body_lag"])/.06,
                max(0., .04-sign*m["muscle_lag"])/.06,
                max(0., .04-sign*m["body_lag"])/.06))
    return (int(failed), float(max(errors)), float(np.mean(errors)))
