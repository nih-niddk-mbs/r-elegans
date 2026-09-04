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


def world_segment_centers(state: BodyState, params: BodyParams) -> Array:
    """Return segment centers transformed into world coordinates."""

    centers = relative_segment_centers(
        state.joint_angles, params.segment_length
    )
    cosine, sine = jnp.cos(state.heading), jnp.sin(state.heading)
    body_to_world = jnp.array(((cosine, -sine), (sine, cosine)))
    return centers @ body_to_world.T + state.position


def _regularized(matrix: Array, relative_regularization: Array) -> Array:
    """Add a dimensionally consistent relative diagonal regularizer."""

    diagonal = jnp.diag(matrix)
    scale = jnp.maximum(jnp.abs(diagonal), jnp.finfo(matrix.dtype).tiny)
    return matrix + relative_regularization * jnp.diag(scale)


def generalized_resistance_matrix(
    joint_angles: Array,
    params: BodyParams,
) -> Array:
    """Return the RFT resistance matrix for rigid and joint velocities.

    Generalized velocities are ``[vx, vy, L*omega, L*qdot...]``. Scaling the
    angular coordinates by body length keeps the SI-valued matrix well
    conditioned. Besides translational drag at each segment center, the
    calculation includes each finite rod's rotational drag about its center.
    """

    positions = relative_segment_centers(joint_angles, params.segment_length)
    orientations = segment_orientations(joint_angles)
    tangents = jnp.stack((jnp.cos(orientations), jnp.sin(orientations)), axis=1)
    normals = jnp.stack((-tangents[:, 1], tangents[:, 0]), axis=1)
    segment_drag = params.segment_length * (
        params.parallel_drag * jnp.einsum("ni,nj->nij", tangents, tangents)
        + params.perpendicular_drag * jnp.einsum("ni,nj->nij", normals, normals)
    )

    count = positions.shape[0]
    body_length = params.segment_length * count
    shape_jacobian = jax.jacfwd(relative_segment_centers, argnums=0)(
        joint_angles, params.segment_length
    )
    translation = jnp.broadcast_to(jnp.eye(2), (count, 2, 2))
    rotation = jnp.stack((-positions[:, 1], positions[:, 0]), axis=1)[..., None]
    point_jacobian = jnp.concatenate(
        (translation, rotation / body_length, shape_jacobian / body_length), axis=2
    )
    resistance = jnp.einsum(
        "nci,ncd,ndj->ij", point_jacobian, segment_drag, point_jacobian
    )

    # A finite segment rotating about its center also dissipates energy. The
    # coefficient is the exact integral of c_perp*x^2 over a uniform rod.
    joint_count = joint_angles.shape[0]
    joint_rotation = jnp.tril(
        jnp.ones((count, joint_count), dtype=joint_angles.dtype), k=-1
    )
    angular_jacobian = jnp.concatenate(
        (
            jnp.zeros((count, 2), dtype=joint_angles.dtype),
            jnp.ones((count, 1), dtype=joint_angles.dtype) / body_length,
            joint_rotation / body_length,
        ),
        axis=1,
    )
    rotational_drag = (
        params.perpendicular_drag * params.segment_length**3 / 12.0
    )
    return resistance + rotational_drag * jnp.einsum(
        "ni,nj->ij", angular_jacobian, angular_jacobian
    )


def body_velocity(
    joint_angles: Array,
    joint_rates: Array,
    params: BodyParams,
) -> Array:
    """Solve for body-frame ``[forward, lateral, angular]`` velocity."""

    resistance = generalized_resistance_matrix(joint_angles, params)
    rigid = resistance[:3, :3]
    coupling = resistance[:3, 3:]
    body_length = params.segment_length * (joint_angles.shape[0] + 1)
    scaled_shape_rate = body_length * joint_rates
    scaled_rigid_velocity = jnp.linalg.solve(
        _regularized(rigid, params.solve_regularization),
        -(coupling @ scaled_shape_rate),
    )
    return scaled_rigid_velocity.at[2].divide(body_length)


def torque_driven_body_velocity(
    joint_angles: Array,
    active_moments: Array,
    params: BodyParams,
    bending_stiffness: Array,
    bending_damping: Array,
    dt: Array,
) -> tuple[Array, Array]:
    """Solve coupled low-Re body motion from joint moments.

    The Schur complement removes force-free rigid motion from the full RFT
    resistance matrix. A backward-Euler elastic term keeps the overdamped
    solve stable for physical SI parameters. Returns body-frame rigid velocity
    and joint angular rates.
    """

    resistance = generalized_resistance_matrix(joint_angles, params)
    rigid = resistance[:3, :3]
    coupling = resistance[:3, 3:]
    shape = resistance[3:, 3:]
    rigid_solve = jnp.linalg.solve(
        _regularized(rigid, params.solve_regularization), coupling
    )
    reduced = shape - coupling.T @ rigid_solve

    body_length = params.segment_length * (joint_angles.shape[0] + 1)
    joint_count = joint_angles.shape[0]
    identity = jnp.eye(joint_count, dtype=joint_angles.dtype)
    # Coordinates in the resistance matrix are L*qdot; transform the physical
    # joint torques and Kelvin--Voigt coefficients into those coordinates.
    system = reduced + (
        bending_damping / body_length**2
        + dt * bending_stiffness / body_length**2
    ) * identity
    right_hand_side = (
        active_moments - bending_stiffness * joint_angles
    ) / body_length
    scaled_shape_rate = jnp.linalg.solve(
        _regularized(system, params.solve_regularization), right_hand_side
    )
    scaled_rigid_velocity = -(rigid_solve @ scaled_shape_rate)
    rigid_velocity = scaled_rigid_velocity.at[2].divide(body_length)
    return rigid_velocity, scaled_shape_rate / body_length


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
