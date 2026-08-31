"""Differentiable planar locomotion using resistive-force theory.

The body is a chain of equal-length segments. At each instant, force and
torque balance determine the rigid-body velocity induced by changing joint
angles. This overdamped model intentionally omits inertia, which is negligible
in the low-Reynolds-number regime targeted by the project.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class BodyParams(NamedTuple):
    """Physical parameters shared by every segment."""

    segment_length: Array
    parallel_drag: Array
    perpendicular_drag: Array
    solve_regularization: Array


class BodyState(NamedTuple):
    """World pose and internal joint angles for a planar body."""

    position: Array
    heading: Array
    joint_angles: Array
    time: Array


def segment_orientations(joint_angles: Array) -> Array:
    """Return segment orientations relative to the body's heading."""

    return jnp.concatenate((jnp.zeros((1,), dtype=joint_angles.dtype), jnp.cumsum(joint_angles)))


def relative_segment_centers(joint_angles: Array, segment_length: Array) -> Array:
    """Return segment centers in a centered body-fixed frame."""

    orientations = segment_orientations(joint_angles)
    tangents = jnp.stack((jnp.cos(orientations), jnp.sin(orientations)), axis=1)
    starts = jnp.concatenate(
        (
            jnp.zeros((1, 2), dtype=joint_angles.dtype),
            jnp.cumsum(segment_length * tangents[:-1], axis=0),
        ),
        axis=0,
    )
    centers = starts + 0.5 * segment_length * tangents
    return centers - jnp.mean(centers, axis=0, keepdims=True)


def _wrench(positions: Array, drag_tensors: Array, velocities: Array) -> Array:
    forces = -jnp.einsum("nij,nj->ni", drag_tensors, velocities)
    torque = jnp.sum(positions[:, 0] * forces[:, 1] - positions[:, 1] * forces[:, 0])
    return jnp.concatenate((jnp.sum(forces, axis=0), torque[None]))


def body_velocity(
    joint_angles: Array,
    joint_rates: Array,
    params: BodyParams,
) -> Array:
    """Solve for body-frame ``[forward, lateral, angular]`` velocity."""

    positions = relative_segment_centers(joint_angles, params.segment_length)
    orientations = segment_orientations(joint_angles)
    tangents = jnp.stack((jnp.cos(orientations), jnp.sin(orientations)), axis=1)
    normals = jnp.stack((-tangents[:, 1], tangents[:, 0]), axis=1)
    drag_tensors = params.segment_length * (
        params.parallel_drag * jnp.einsum("ni,nj->nij", tangents, tangents)
        + params.perpendicular_drag * jnp.einsum("ni,nj->nij", normals, normals)
    )

    shape_jacobian = jax.jacfwd(relative_segment_centers, argnums=0)(
        joint_angles, params.segment_length
    )
    shape_velocity = jnp.einsum("nck,k->nc", shape_jacobian, joint_rates)

    count = positions.shape[0]
    x_translation = jnp.broadcast_to(jnp.array([1.0, 0.0]), (count, 2))
    y_translation = jnp.broadcast_to(jnp.array([0.0, 1.0]), (count, 2))
    rotation = jnp.stack((-positions[:, 1], positions[:, 0]), axis=1)

    resistance = jnp.stack(
        (
            _wrench(positions, drag_tensors, x_translation),
            _wrench(positions, drag_tensors, y_translation),
            _wrench(positions, drag_tensors, rotation),
        ),
        axis=1,
    )
    shape_wrench = _wrench(positions, drag_tensors, shape_velocity)
    stabilized = resistance - params.solve_regularization * jnp.eye(3)
    return jnp.linalg.solve(stabilized, -shape_wrench)


def prescribed_traveling_wave(
    time: Array,
    num_joints: int,
    *,
    amplitude: float = 0.35,
    frequency: float = 1.0,
    waves_along_body: float = 1.0,
) -> tuple[Array, Array]:
    """Return joint angles and rates for a head-to-tail sinusoidal wave."""

    phase = 2.0 * jnp.pi * (
        waves_along_body * jnp.arange(num_joints) / max(num_joints, 1)
        - frequency * time
    )
    angles = amplitude * jnp.sin(phase)
    rates = -2.0 * jnp.pi * frequency * amplitude * jnp.cos(phase)
    return angles, rates


@partial(jax.jit, static_argnames=("num_segments", "steps"))
def simulate_traveling_wave(
    params: BodyParams,
    *,
    num_segments: int = 12,
    steps: int = 200,
    dt: float = 0.01,
    amplitude: float = 0.35,
    frequency: float = 1.0,
    waves_along_body: float = 1.0,
) -> tuple[BodyState, BodyState]:
    """Simulate a prescribed wave and return final state plus trajectory."""

    initial_angles, _ = prescribed_traveling_wave(
        jnp.array(0.0),
        num_segments - 1,
        amplitude=amplitude,
        frequency=frequency,
        waves_along_body=waves_along_body,
    )
    initial = BodyState(
        position=jnp.zeros(2),
        heading=jnp.array(0.0),
        joint_angles=initial_angles,
        time=jnp.array(0.0),
    )

    def advance(state: BodyState, _: Array) -> tuple[BodyState, BodyState]:
        angles, rates = prescribed_traveling_wave(
            state.time,
            num_segments - 1,
            amplitude=amplitude,
            frequency=frequency,
            waves_along_body=waves_along_body,
        )
        velocity = body_velocity(angles, rates, params)
        cosine, sine = jnp.cos(state.heading), jnp.sin(state.heading)
        body_to_world = jnp.array(((cosine, -sine), (sine, cosine)))
        next_time = state.time + dt
        next_angles, _ = prescribed_traveling_wave(
            next_time,
            num_segments - 1,
            amplitude=amplitude,
            frequency=frequency,
            waves_along_body=waves_along_body,
        )
        next_state = BodyState(
            position=state.position + dt * (body_to_world @ velocity[:2]),
            heading=state.heading + dt * velocity[2],
            joint_angles=next_angles,
            time=next_time,
        )
        return next_state, next_state

    return jax.lax.scan(advance, initial, xs=jnp.arange(steps))

