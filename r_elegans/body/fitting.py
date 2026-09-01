"""Compact differentiable controllers for fitting body motion primitives."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .actuation import (
    MuscleBodyParams,
    MuscleBodyState,
    initialize_muscle_body,
    muscle_body_step,
)
from .neuromuscular import (
    BODY_WALL_MUSCLE_NAMES,
    build_muscle_projection,
    muscle_longitudinal_positions,
    project_muscles_to_joints,
)

Array = jax.Array


class PeriodicControllerParams(NamedTuple):
    """Physical parameters of a traveling anatomical muscle wave."""

    amplitude: Array
    frequency: Array
    waves_along_body: Array
    phase_offset: Array
    steering_bias: Array


class BodyMotionLossWeights(NamedTuple):
    """Weights for fitting a forward, straight, economical motion."""

    lateral: Array = jnp.asarray(1.0)
    heading: Array = jnp.asarray(0.25)
    energy: Array = jnp.asarray(1e-3)
    curvature: Array = jnp.asarray(1e-3)


def decode_periodic_controller(raw: Array) -> PeriodicControllerParams:
    """Map five unconstrained fit parameters to stable physical ranges."""

    raw = jnp.asarray(raw)
    if raw.shape != (5,):
        raise ValueError("A periodic controller requires five raw parameters")
    return PeriodicControllerParams(
        amplitude=jax.nn.sigmoid(raw[0]),
        frequency=0.25 + 1.75 * jax.nn.sigmoid(raw[1]),
        waves_along_body=2.0 * jnp.tanh(raw[2]),
        phase_offset=raw[3],
        steering_bias=0.5 * jnp.tanh(raw[4]),
    )


def periodic_muscle_activations(
    time: Array,
    params: PeriodicControllerParams,
) -> Array:
    """Generate bounded activation for each of the 95 anatomical muscles."""

    positions = muscle_longitudinal_positions()
    phase = 2.0 * jnp.pi * (
        params.waves_along_body * positions - params.frequency * time
    ) + params.phase_offset
    signed = params.amplitude * jnp.sin(phase) + params.steering_bias
    dorsal_mask = jnp.asarray(
        [name.startswith("d") for name in BODY_WALL_MUSCLE_NAMES]
    )
    activation = jnp.where(
        dorsal_mask,
        jnp.maximum(signed, 0.0),
        jnp.maximum(-signed, 0.0),
    )
    return jnp.clip(activation, 0.0, 1.0)


@partial(jax.jit, static_argnames=("num_segments", "steps"))
def simulate_periodic_controller(
    controller_params: PeriodicControllerParams,
    body_params: MuscleBodyParams,
    *,
    num_segments: int = 12,
    steps: int = 250,
    dt: float = 0.02,
) -> tuple[MuscleBodyState, MuscleBodyState, Array]:
    """Roll out a 95-muscle periodic controller through body physics."""

    initial = initialize_muscle_body(num_segments)
    projection = build_muscle_projection(num_segments - 1)

    def advance(
        state: MuscleBodyState, _: Array
    ) -> tuple[MuscleBodyState, tuple[MuscleBodyState, Array]]:
        activation = periodic_muscle_activations(state.time, controller_params)
        dorsal, ventral = project_muscles_to_joints(activation, projection)
        next_state = muscle_body_step(
            state, body_params, dorsal, ventral, dt
        )
        return next_state, (next_state, activation)

    final, (trajectory, activations) = jax.lax.scan(
        advance, initial, xs=jnp.arange(steps)
    )
    return final, trajectory, activations


def body_motion_loss(
    raw_controller: Array,
    body_params: MuscleBodyParams,
    loss_weights: BodyMotionLossWeights = BodyMotionLossWeights(),
    *,
    num_segments: int = 12,
    steps: int = 250,
    dt: float = 0.02,
) -> Array:
    """Loss for positive-x locomotion with straightness and effort penalties."""

    controller = decode_periodic_controller(raw_controller)
    final, trajectory, activations = simulate_periodic_controller(
        controller,
        body_params,
        num_segments=num_segments,
        steps=steps,
        dt=dt,
    )
    forward_loss = -final.position[0]
    lateral_loss = loss_weights.lateral * final.position[1] ** 2
    heading_loss = loss_weights.heading * final.heading ** 2
    energy_loss = loss_weights.energy * jnp.mean(activations**2)
    curvature_loss = loss_weights.curvature * jnp.mean(
        trajectory.joint_angles**2
    )
    return (
        forward_loss
        + lateral_loss
        + heading_loss
        + energy_loss
        + curvature_loss
    )
