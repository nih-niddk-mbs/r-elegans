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


def muscle_sequence_body(muscles, *, dt=0.02):
    params = default_muscle_body_params(12)
    projection = build_muscle_projection(11)
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
