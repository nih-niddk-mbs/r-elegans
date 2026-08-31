import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.body import (
    BodyParams,
    body_velocity,
    prescribed_traveling_wave,
    relative_segment_centers,
    simulate_traveling_wave,
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

