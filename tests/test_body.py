import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.body import (
    BodyParams,
    as_body_state,
    body_velocity,
    default_muscle_body_params,
    initialize_muscle_body,
    muscle_body_step,
    prescribed_muscle_wave,
    prescribed_traveling_wave,
    relative_segment_centers,
    simulate_traveling_wave,
    simulate_muscle_wave,
    world_segment_centers,
)


def body_params() -> BodyParams:
    return BodyParams(
        segment_length=jnp.array(1.0 / 12.0),
        parallel_drag=jnp.array(1.0),
        perpendicular_drag=jnp.array(2.0),
        solve_regularization=jnp.array(1e-6),
    )


def test_straight_body_has_regular_segment_centers() -> None:
    centers = relative_segment_centers(jnp.zeros(11), jnp.array(1.0 / 12.0))
    separations = jnp.linalg.norm(jnp.diff(centers, axis=0), axis=1)

    np.testing.assert_allclose(separations, jnp.full((11,), 1.0 / 12.0), rtol=1e-6)
    np.testing.assert_allclose(jnp.mean(centers, axis=0), jnp.zeros(2), atol=1e-7)


def test_zero_shape_rate_produces_zero_body_velocity() -> None:
    angles, _ = prescribed_traveling_wave(jnp.array(0.0), 11)
    velocity = jax.jit(body_velocity)(angles, jnp.zeros_like(angles), body_params())

    np.testing.assert_allclose(velocity, jnp.zeros(3), atol=1e-7)


def test_body_velocity_vectorizes_over_shape_rates() -> None:
    angles, rates = prescribed_traveling_wave(jnp.array(0.0), 11)
    batched_rates = jnp.stack((rates, -rates))
    velocities = jax.vmap(lambda rate: body_velocity(angles, rate, body_params()))(
        batched_rates
    )

    assert velocities.shape == (2, 3)
    np.testing.assert_allclose(velocities[0], -velocities[1], rtol=1e-5, atol=1e-6)


def test_prescribed_wave_generates_finite_displacement() -> None:
    final_state, trajectory = simulate_traveling_wave(
        body_params(), num_segments=12, steps=200, dt=0.01
    )

    assert trajectory.position.shape == (200, 2)
    assert jnp.all(jnp.isfinite(trajectory.position))
    assert jnp.linalg.norm(final_state.position) > 1e-4


def test_locomotion_is_differentiable_with_respect_to_wave_amplitude() -> None:
    def final_forward_position(amplitude: jax.Array) -> jax.Array:
        final_state, _ = simulate_traveling_wave(
            body_params(),
            num_segments=12,
            steps=50,
            dt=0.01,
            amplitude=amplitude,
        )
        return final_state.position[0]

    gradient = jax.grad(final_forward_position)(jnp.array(0.3))

    assert jnp.isfinite(gradient)
    assert jnp.abs(gradient) > 1e-7


def test_balanced_muscles_leave_straight_body_stationary() -> None:
    params = default_muscle_body_params(12)
    state = initialize_muscle_body(12)
    balanced = jnp.full((11,), 0.6)

    final = jax.jit(muscle_body_step)(
        state, params, balanced, balanced, jnp.asarray(0.01)
    )

    np.testing.assert_allclose(final.position, state.position, atol=1e-8)
    np.testing.assert_allclose(final.joint_angles, state.joint_angles, atol=1e-8)
    np.testing.assert_allclose(final.muscle_activation, 0.0, atol=1e-8)


def test_opposing_muscles_bend_in_opposite_directions() -> None:
    params = default_muscle_body_params(4)
    state = initialize_muscle_body(4)
    active = jnp.ones((3,))
    relaxed = jnp.zeros((3,))

    dorsal = muscle_body_step(state, params, active, relaxed, 0.01)
    ventral = muscle_body_step(state, params, relaxed, active, 0.01)

    assert jnp.all(dorsal.joint_angles > 0.0)
    np.testing.assert_allclose(
        dorsal.joint_angles, -ventral.joint_angles, rtol=1e-6
    )


def test_muscle_wave_generates_bounded_shape_and_displacement() -> None:
    params = default_muscle_body_params(12)

    final, trajectory = simulate_muscle_wave(
        params, num_segments=12, steps=500, dt=0.01
    )

    assert trajectory.position.shape == (500, 2)
    assert trajectory.joint_angles.shape == (500, 11)
    assert jnp.all(jnp.isfinite(trajectory.position))
    assert jnp.max(jnp.abs(trajectory.joint_angles)) <= params.max_joint_angle
    assert jnp.linalg.norm(final.position) > 1e-3


def test_muscle_driven_locomotion_is_differentiable() -> None:
    params = default_muscle_body_params(8)

    def forward_displacement(amplitude: jax.Array) -> jax.Array:
        final, _ = simulate_muscle_wave(
            params,
            num_segments=8,
            steps=100,
            dt=0.01,
            amplitude=amplitude,
        )
        return final.position[0]

    gradient = jax.grad(forward_displacement)(jnp.asarray(0.7))

    assert jnp.isfinite(gradient)
    assert jnp.abs(gradient) > 1e-7


def test_reversing_wave_direction_reverses_locomotion() -> None:
    params = default_muscle_body_params(12)
    forward, _ = simulate_muscle_wave(
        params, num_segments=12, steps=300, dt=0.01, waves_along_body=1.0
    )
    reverse, _ = simulate_muscle_wave(
        params, num_segments=12, steps=300, dt=0.01, waves_along_body=-1.0
    )

    assert forward.position[0] * reverse.position[0] < 0.0
    np.testing.assert_allclose(
        jnp.abs(forward.position[0]),
        jnp.abs(reverse.position[0]),
        rtol=1e-5,
    )


def test_world_centers_follow_body_pose() -> None:
    params = default_muscle_body_params(4)
    muscle_state = initialize_muscle_body(
        4, position=jnp.asarray([2.0, -1.0]), heading=jnp.pi / 2.0
    )

    centers = world_segment_centers(as_body_state(muscle_state), params.mechanics)

    np.testing.assert_allclose(jnp.mean(centers, axis=0), [2.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(centers[:, 0], 2.0, atol=1e-6)


def test_prescribed_muscle_commands_are_bounded() -> None:
    dorsal, ventral = prescribed_muscle_wave(
        jnp.asarray(0.2), 11, amplitude=1.0, coactivation=0.2
    )

    assert jnp.all((dorsal >= 0.0) & (dorsal <= 1.0))
    assert jnp.all((ventral >= 0.0) & (ventral <= 1.0))
