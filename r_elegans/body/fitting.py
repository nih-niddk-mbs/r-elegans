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


class CommandedControllerParams(NamedTuple):
    """Forward/reverse gaits and direction-specific steering gains."""

    forward: PeriodicControllerParams
    reverse: PeriodicControllerParams
    forward_steering_gain: Array
    reverse_steering_gain: Array
    forward_low_speed_steering_gain: Array
    reverse_low_speed_steering_gain: Array


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


def decode_commanded_controller(raw: Array) -> CommandedControllerParams:
    """Map 14 unconstrained parameters to a continuous command controller."""

    raw = jnp.asarray(raw)
    if raw.shape != (14,):
        raise ValueError("A commanded controller requires 14 raw parameters")
    return CommandedControllerParams(
        forward=decode_periodic_controller(raw[:5]),
        reverse=decode_periodic_controller(raw[5:10]),
        forward_steering_gain=0.25 * jnp.tanh(raw[10]),
        reverse_steering_gain=0.25 * jnp.tanh(raw[11]),
        forward_low_speed_steering_gain=0.25 * jnp.tanh(raw[12]),
        reverse_low_speed_steering_gain=0.25 * jnp.tanh(raw[13]),
    )


def controller_for_command(
    command: Array,
    params: CommandedControllerParams,
) -> PeriodicControllerParams:
    """Convert ``[speed, steering]`` in ``[-1, 1]`` to a periodic gait."""

    command = jnp.asarray(command)
    if command.shape != (2,):
        raise ValueError("A motion command must contain [speed, steering]")
    speed, steering = jnp.clip(command, -1.0, 1.0)
    magnitude = jnp.abs(speed)
    forward = speed >= 0.0

    def select(forward_value: Array, reverse_value: Array) -> Array:
        return jnp.where(forward, forward_value, reverse_value)

    base = PeriodicControllerParams(
        amplitude=select(params.forward.amplitude, params.reverse.amplitude),
        frequency=select(params.forward.frequency, params.reverse.frequency),
        waves_along_body=select(
            params.forward.waves_along_body,
            params.reverse.waves_along_body,
        ),
        phase_offset=select(
            params.forward.phase_offset, params.reverse.phase_offset
        ),
        steering_bias=select(
            params.forward.steering_bias, params.reverse.steering_bias
        ),
    )
    steering_gain = select(
        params.forward_steering_gain, params.reverse_steering_gain
    )
    low_speed_steering_gain = select(
        params.forward_low_speed_steering_gain,
        params.reverse_low_speed_steering_gain,
    )
    return PeriodicControllerParams(
        amplitude=jnp.sqrt(magnitude) * base.amplitude,
        frequency=0.25 + magnitude * (base.frequency - 0.25),
        waves_along_body=base.waves_along_body,
        phase_offset=base.phase_offset,
        steering_bias=(
            magnitude * base.steering_bias
            + (
                jnp.sqrt(magnitude) * steering_gain
                + 4.0
                * magnitude
                * (1.0 - magnitude)
                * low_speed_steering_gain
            )
            * steering
        ),
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


def commanded_muscle_activations(
    phase: Array,
    command: Array,
    params: CommandedControllerParams,
) -> tuple[Array, PeriodicControllerParams]:
    """Generate 95 muscles from a command while preserving external phase."""

    controller = controller_for_command(command, params)
    phase_controller = controller._replace(
        frequency=jnp.asarray(0.0), phase_offset=phase
    )
    return periodic_muscle_activations(0.0, phase_controller), controller


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


@partial(jax.jit, static_argnames=("num_segments",))
def simulate_muscle_trajectory(
    muscle_activations: Array,
    body_params: MuscleBodyParams,
    *,
    num_segments: int = 12,
    dt: float = 0.02,
) -> tuple[MuscleBodyState, MuscleBodyState]:
    """Roll out an externally supplied ``[time, 95]`` muscle trajectory."""

    initial = initialize_muscle_body(num_segments)
    projection = build_muscle_projection(num_segments - 1)

    def advance(
        state: MuscleBodyState, activation: Array
    ) -> tuple[MuscleBodyState, MuscleBodyState]:
        dorsal, ventral = project_muscles_to_joints(activation, projection)
        next_state = muscle_body_step(
            state, body_params, dorsal, ventral, dt
        )
        return next_state, next_state

    return jax.lax.scan(advance, initial, muscle_activations)


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


def commanded_body_motion_loss(
    raw_controller: Array,
    body_params: MuscleBodyParams,
    commands: Array,
    *,
    num_segments: int = 12,
    steps: int = 250,
    dt: float = 0.02,
) -> Array:
    """Fit shared forward/reverse gaits across ``[speed, steering]`` commands."""

    params = decode_commanded_controller(raw_controller)

    def rollout(command: Array) -> tuple[Array, Array, Array, Array, Array]:
        controller = controller_for_command(command, params)
        final, trajectory, activations = simulate_periodic_controller(
            controller,
            body_params,
            num_segments=num_segments,
            steps=steps,
            dt=dt,
        )
        return (
            final.position,
            final.heading,
            jnp.mean(activations**2),
            jnp.mean(trajectory.joint_angles**2),
            command,
        )

    positions, headings, energies, curvatures, bounded_commands = jax.vmap(
        rollout
    )(commands)
    speeds = bounded_commands[:, 0]
    steering = bounded_commands[:, 1]
    target_x = 0.6 * speeds * (1.0 - 0.3 * jnp.abs(steering))
    target_y = 0.35 * speeds * steering
    return jnp.mean(
        2.0 * (positions[:, 0] - target_x) ** 2
        + 4.0 * (positions[:, 1] - target_y) ** 2
        + 1e-3 * headings**2
        + 1e-3 * energies
        + 1e-3 * curvatures
    )
