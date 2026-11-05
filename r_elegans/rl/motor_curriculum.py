"""Supervised first motor milestone: a small bend, with no reward or food task."""

import jax
import jax.numpy as jnp
import optax

from r_elegans.body import (
    build_muscle_projection, default_muscle_body_params, initialize_muscle_body,
    muscle_body_step, project_muscles_to_joints,
)
from r_elegans.body.neuromuscular import BODY_WALL_MUSCLE_NAMES, muscle_longitudinal_positions


def bend_target(steps=64, *, amplitude=0.04):
    """An explicitly synthetic, mild dorsal bend around 40% resting activation.

    This is a constant motor target, not a biological recording or a gait.
    The spatial envelope avoids concentrating bend at the body's endpoints.
    """
    if steps < 2 or not 0 < amplitude <= 0.1:
        raise ValueError("Use at least two steps and bend amplitude in (0, 0.1]")
    dorsal = jnp.asarray([n.startswith("d") for n in BODY_WALL_MUSCLE_NAMES])
    profile = jnp.sin(jnp.pi * muscle_longitudinal_positions())
    muscles = 0.4 + amplitude * jnp.where(dorsal, 1.0, -1.0) * profile
    return jnp.broadcast_to(muscles, (steps, 95))


def neural_muscle_sequence(actor, interface, steps):
    """Free-run the recurrent brain from rest, without food, clock or teacher input."""
    def step(voltage, _):
        muscles, voltage = interface.deterministic_step(actor, voltage, jnp.zeros(3))
        return voltage, muscles
    return jax.lax.scan(step, interface.init_carry(actor), None, length=steps)[1]


def muscle_sequence_body(muscles, *, dt=0.02, body_params=None):
    params = default_muscle_body_params(12) if body_params is None else body_params
    projection = build_muscle_projection(11, head_at_end=True)
    def step(body, activation):
        dorsal, ventral = project_muscles_to_joints(activation, projection)
        body = muscle_body_step(body, params, dorsal, ventral, dt)
        return body, body
    return jax.lax.scan(step, initialize_muscle_body(12), muscles)[1]


def make_bend_update(interface, target, optimizer):
    """Temporal supervision through the full recurrent network and fixed NMJs."""
    def loss(actor):
        predicted = neural_muscle_sequence(actor, interface, target.shape[0])
        return jnp.mean((predicted - target) ** 2)

    @jax.jit
    def update(actor, opt_state):
        value, grads = jax.value_and_grad(loss)(actor)
        updates, opt_state = optimizer.update(grads, opt_state, actor)
        return optax.apply_updates(actor, updates), opt_state, value
    return update


def bend_metrics(predicted, target):
    """Check muscle matching AND the resulting body, including a longer hold."""
    body = muscle_sequence_body(predicted)
    reference = muscle_sequence_body(target)
    rmse = jnp.sqrt(jnp.mean((predicted - target) ** 2))
    body_rmse = jnp.sqrt(jnp.mean((body.joint_angles - reference.joint_angles) ** 2))
    scale = jnp.sqrt(jnp.mean(reference.joint_angles ** 2))
    # Last quarter checks maintenance rather than only a transient bend.
    tail = predicted.shape[0] * 3 // 4
    hold_error = jnp.sqrt(jnp.mean((body.joint_angles[tail:] - reference.joint_angles[tail:]) ** 2))
    return {"muscle_rmse": rmse, "body_relative_rmse": body_rmse / jnp.maximum(scale, 1e-8),
            "hold_relative_rmse": hold_error / jnp.maximum(scale, 1e-8),
            "mean_final_joint_angle": jnp.mean(body.joint_angles[-1]),
            "target_mean_final_joint_angle": jnp.mean(reference.joint_angles[-1])}


def bend_passes(metrics):
    return (metrics["muscle_rmse"] < 0.03 and metrics["body_relative_rmse"] < 0.2
            and metrics["hold_relative_rmse"] < 0.2
            and metrics["mean_final_joint_angle"] * metrics["target_mean_final_joint_angle"] > 0)


def motor_stimulation_vector(model, gain=0.1):
    """Fixed research input into NMJ-projecting neurons, inferred from anatomy.

    A positive command excites neurons whose signed outgoing contacts favor
    dorsal over ventral muscle activation, and suppresses the converse. It is
    artificial distributed stimulation, NOT a known sensory/motor command organ.
    It never supplies target muscle activations to the forward computation.
    """
    dorsal = jnp.asarray([n.startswith("d") for n in BODY_WALL_MUSCLE_NAMES])
    weights = model.neuromuscular_params.synapse_weights
    signed = weights * model.neuromuscular_params.synapse_signs
    score = jnp.sum(signed * jnp.where(dorsal, 1., -1.)[:, None], axis=0)
    score /= jnp.maximum(weights.sum(axis=0), 1.)
    return gain * score


def commanded_bend_target(commands, amplitude=0.04):
    spatial = bend_target(2, amplitude=amplitude)[0] - 0.4
    return 0.4 + jnp.asarray(commands)[:, None] * spatial[None, :]


def commanded_muscle_sequence(actor, interface, commands, stimulation):
    def step(voltage, command):
        muscles, voltage = interface.deterministic_step(actor, voltage, command * stimulation)
        return voltage, muscles
    return jax.lax.scan(step, interface.init_carry(actor), commands)[1]


def alternating_metrics(predicted, target):
    body = muscle_sequence_body(predicted)
    reference = muscle_sequence_body(target)
    angles, expected = body.joint_angles, reference.joint_angles
    scale = jnp.sqrt(jnp.mean(expected ** 2))
    mean, desired = angles.mean(axis=1), expected.mean(axis=1)
    centered, centered_target = mean - mean.mean(), desired - desired.mean()
    correlation = jnp.sum(centered * centered_target) / jnp.maximum(
        jnp.linalg.norm(centered) * jnp.linalg.norm(centered_target), 1e-8)
    active = jnp.abs(desired) > 0.3 * jnp.max(jnp.abs(desired))
    direction = jnp.sum(active * (mean * desired > 0)) / jnp.maximum(jnp.sum(active), 1)
    return dict(muscle_rmse=jnp.sqrt(jnp.mean((predicted - target) ** 2)),
                body_relative_rmse=jnp.sqrt(jnp.mean((angles - expected) ** 2)) / jnp.maximum(scale, 1e-8),
                bend_correlation=correlation, direction_agreement=direction)


def alternating_passes(metrics):
    return (metrics["muscle_rmse"] < 0.025 and metrics["body_relative_rmse"] < 0.25
            and metrics["bend_correlation"] > 0.9 and metrics["direction_agreement"] > 0.95)


def traveling_bend_target(commands, *, delay=0.4, dt=0.02, amplitude=0.04):
    """A causal anterior-to-posterior delayed command, for supervision only.

    No delayed commands or target phases are injected into neurons. The brain
    receives the SAME scalar stimulation as in alternation, and must learn the
    spatial timing through its own dynamics. This is an externally paced wave.
    """
    if delay <= 0 or dt <= 0:
        raise ValueError("Positive propagation delay and dt required")
    times = jnp.arange(commands.shape[0]) * dt
    delayed = jax.vmap(lambda p: jnp.interp(times - delay * p, times, commands, left=0.0))(
        muscle_longitudinal_positions()).T
    return 0.4 + delayed * (bend_target(2, amplitude=amplitude)[0] - 0.4)


def signal_lag(front, rear, *, dt=0.02, max_lag_steps=30):
    """Normalized overlap correlation; positive lag means posterior follows.

    Ignore initial settling. Both lag signs are searched, so reverse waves
    cannot be mistaken for forward propagation. Resolution is one time step.
    """
    indices = jnp.arange(front.shape[0])
    lags = jnp.arange(-max_lag_steps, max_lag_steps + 1)
    def correlation(lag):
        shifted = indices + lag
        mask = (indices >= front.shape[0] // 4) & (shifted >= 0) & (shifted < front.shape[0])
        count = jnp.maximum(mask.sum(), 1)
        b = rear[jnp.clip(shifted, 0, front.shape[0] - 1)]
        a = front - jnp.sum(front * mask) / count
        b = b - jnp.sum(b * mask) / count
        return jnp.sum(a * b * mask) / jnp.maximum(
            jnp.sqrt(jnp.sum(a*a*mask) * jnp.sum(b*b*mask)), 1e-10)
    scores = jax.vmap(correlation)(lags)
    return lags[jnp.argmax(scores)] * dt


def traveling_metrics(predicted, target):
    metrics = alternating_metrics(predicted, target)
    positions = muscle_longitudinal_positions()
    dorsal = jnp.asarray([n.startswith("d") for n in BODY_WALL_MUSCLE_NAMES])
    def band(muscles, low, high):
        selected = (positions >= low) & (positions <= high)
        d, v = selected & dorsal, selected & ~dorsal
        return (jnp.sum(muscles * d, axis=1) / d.sum()
                - jnp.sum(muscles * v, axis=1) / v.sum())
    def muscle_lag(m):
        return signal_lag(band(m, .2, .4), band(m, .6, .8))
    def body_lag(m):
        angles = muscle_sequence_body(m).joint_angles
        return signal_lag(angles[:, 6:9].mean(axis=1), angles[:, 2:5].mean(axis=1))
    metrics.update(muscle_lag=muscle_lag(predicted), target_muscle_lag=muscle_lag(target),
                   body_lag=body_lag(predicted), target_body_lag=body_lag(target))
    return metrics


def traveling_passes(metrics):
    # Float32 lag ticks (20 ms) must not fail exactly on a gate boundary.
    eps = 1e-6
    return (alternating_passes(metrics) and metrics["muscle_lag"] + eps >= .04
            and metrics["body_lag"] + eps >= .04
            and abs(metrics["muscle_lag"] - metrics["target_muscle_lag"]) <= .06 + eps
            and abs(metrics["body_lag"] - metrics["target_body_lag"]) <= .06 + eps)
