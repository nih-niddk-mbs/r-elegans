"""Dorsal/ventral muscle actuation for the planar worm body.

The passive bend is a discrete Kelvin--Voigt element: elastic torque resists
joint angle and viscous torque resists joint rate. Opposing dorsal and ventral
activations produce a signed active bending moment. Environmental force and
torque balance are delegated to :mod:`r_elegans.body.mechanics`.

Parameters are normalized by default. Empirical stiffness, damping, muscle
moment, and substrate drag calibrations belong in the external data catalog.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .mechanics import (
    BodyParams,
    BodyState,
    body_velocity,
    torque_driven_body_velocity,
)

Array = jax.Array


class MuscleBodyParams(NamedTuple):
    """Mechanical and muscle parameters for an actuated body."""

    mechanics: BodyParams
    bending_stiffness: Array
    bending_damping: Array
    muscle_moment_scale: Array
    activation_time_constant: Array
    max_joint_angle: Array


class MuscleBodyState(NamedTuple):
    """World pose, body shape, and signed dorsal-minus-ventral activation."""

    position: Array
    heading: Array
    joint_angles: Array
    muscle_activation: Array
    time: Array


def default_muscle_body_params(
    num_segments: int = 12,
    *,
    body_length: float = 1.0,
) -> MuscleBodyParams:
    """Return stable normalized parameters for development and control tests."""

    if num_segments < 2:
        raise ValueError("An actuated body requires at least two segments")
    return MuscleBodyParams(
        mechanics=BodyParams(
            segment_length=jnp.asarray(body_length / num_segments),
            parallel_drag=jnp.asarray(1.0),
            perpendicular_drag=jnp.asarray(2.0),
            solve_regularization=jnp.asarray(1e-6),
        ),
        bending_stiffness=jnp.asarray(4.0),
        bending_damping=jnp.asarray(1.0),
        muscle_moment_scale=jnp.asarray(3.0),
        activation_time_constant=jnp.asarray(0.05),
        max_joint_angle=jnp.asarray(0.75),
    )


def initialize_muscle_body(
    num_segments: int = 12,
    *,
    position: Array | None = None,
    heading: float | Array = 0.0,
) -> MuscleBodyState:
    """Create a straight, relaxed body state."""

    if num_segments < 2:
        raise ValueError("An actuated body requires at least two segments")
    joint_angles = jnp.zeros((num_segments - 1,))
    return MuscleBodyState(
        position=jnp.zeros((2,)) if position is None else jnp.asarray(position),
        heading=jnp.asarray(heading),
        joint_angles=joint_angles,
        muscle_activation=jnp.zeros_like(joint_angles),
        time=jnp.asarray(0.0),
    )


def _validate_step_inputs(
    state: MuscleBodyState,
    params: MuscleBodyParams,
    dorsal_activation: Array,
    ventral_activation: Array,
) -> None:
    expected = state.joint_angles.shape
    if dorsal_activation.shape != expected or ventral_activation.shape != expected:
        raise ValueError(
            "Dorsal and ventral activation must have one value per body joint"
        )
    if state.muscle_activation.shape != expected:
        raise ValueError("Muscle state must have one value per body joint")
    if params.mechanics.segment_length.ndim != 0:
        raise ValueError("The current body model requires one shared segment length")


def muscle_body_step(
    state: MuscleBodyState,
    params: MuscleBodyParams,
    dorsal_activation: Array,
    ventral_activation: Array,
    dt: float | Array,
) -> MuscleBodyState:
    """Advance one interval from opposing muscle commands in ``[0, 1]``.

    Positive signed activation bends toward the dorsal side. Muscle activation
    follows the command with exact first-order kinetics; joint motion then
    satisfies the overdamped balance between active, elastic, and viscous bend
    moments.
    """

    dorsal = jnp.asarray(dorsal_activation)
    ventral = jnp.asarray(ventral_activation)
    _validate_step_inputs(state, params, dorsal, ventral)
    dt_array = jnp.asarray(dt)

    signed_command = jnp.clip(dorsal, 0.0, 1.0) - jnp.clip(
        ventral, 0.0, 1.0
    )
    activation_fraction = -jnp.expm1(
        -dt_array / params.activation_time_constant
    )
    activation = state.muscle_activation + activation_fraction * (
        signed_command - state.muscle_activation
    )

    active_moment = params.muscle_moment_scale * activation
    _, unconstrained_rate = torque_driven_body_velocity(
        state.joint_angles,
        active_moment,
        params.mechanics,
        params.bending_stiffness,
        params.bending_damping,
        dt_array,
    )
    unconstrained_angles = state.joint_angles + dt_array * unconstrained_rate
    next_angles = jnp.clip(
        unconstrained_angles,
        -params.max_joint_angle,
        params.max_joint_angle,
    )
    realized_rate = (next_angles - state.joint_angles) / dt_array

    # Clipping changes the realized shape velocity, so recompute the force-free
    # rigid motion from that velocity only when a joint hits its hard limit.
    velocity = body_velocity(state.joint_angles, realized_rate, params.mechanics)
    cosine, sine = jnp.cos(state.heading), jnp.sin(state.heading)
    body_to_world = jnp.array(((cosine, -sine), (sine, cosine)))
    return MuscleBodyState(
        position=state.position + dt_array * (body_to_world @ velocity[:2]),
        heading=state.heading + dt_array * velocity[2],
        joint_angles=next_angles,
        muscle_activation=activation,
        time=state.time + dt_array,
    )


def prescribed_muscle_wave(
    time: Array,
    num_joints: int,
    *,
    amplitude: float | Array = 1.0,
    frequency: float | Array = 1.0,
    waves_along_body: float | Array = 1.0,
    coactivation: float | Array = 0.0,
) -> tuple[Array, Array]:
    """Generate bounded anti-phase dorsal and ventral muscle commands."""

    phase = 2.0 * jnp.pi * (
        waves_along_body * jnp.arange(num_joints) / max(num_joints - 1, 1)
        - frequency * time
    )
    signed = jnp.clip(amplitude, 0.0, 1.0) * jnp.sin(phase)
    baseline = 0.5 * jnp.clip(coactivation, 0.0, 1.0)
    dorsal = jnp.clip(baseline + jnp.maximum(signed, 0.0), 0.0, 1.0)
    ventral = jnp.clip(baseline + jnp.maximum(-signed, 0.0), 0.0, 1.0)
    return dorsal, ventral


@partial(jax.jit, static_argnames=("num_segments", "steps"))
def simulate_muscle_wave(
    params: MuscleBodyParams,
    *,
    num_segments: int = 12,
    steps: int = 500,
    dt: float = 0.01,
    amplitude: float | Array = 1.0,
    frequency: float | Array = 1.0,
    waves_along_body: float | Array = 1.0,
) -> tuple[MuscleBodyState, MuscleBodyState]:
    """Run an end-to-end muscle-driven locomotion simulation."""

    initial = initialize_muscle_body(num_segments)

    def advance(
        state: MuscleBodyState, _: Array
    ) -> tuple[MuscleBodyState, MuscleBodyState]:
        dorsal, ventral = prescribed_muscle_wave(
            state.time,
            num_segments - 1,
            amplitude=amplitude,
            frequency=frequency,
            waves_along_body=waves_along_body,
        )
        next_state = muscle_body_step(
            state, params, dorsal, ventral, dt
        )
        return next_state, next_state

    return jax.lax.scan(advance, initial, xs=jnp.arange(steps))


def as_body_state(state: MuscleBodyState) -> BodyState:
    """Drop muscle state for APIs that consume the common body pose schema."""

    return BodyState(
        position=state.position,
        heading=state.heading,
        joint_angles=state.joint_angles,
        time=state.time,
    )
