import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.body import (
    BODY_WALL_MUSCLE_NAMES,
    body_motion_loss,
    decode_periodic_controller,
    default_muscle_body_params,
    periodic_muscle_activations,
    simulate_periodic_controller,
)


def test_controller_decoding_has_stable_bounds() -> None:
    controller = decode_periodic_controller(jnp.zeros((5,)))

    assert 0.0 < controller.amplitude < 1.0
    assert 0.25 < controller.frequency < 2.0
    assert controller.waves_along_body == 0.0
    assert controller.steering_bias == 0.0


def test_periodic_controller_activates_opposing_muscle_sides() -> None:
    controller = decode_periodic_controller(
        jnp.asarray([2.0, 0.0, -0.5, 0.0, 0.0])
    )
    activation = periodic_muscle_activations(jnp.asarray(0.0), controller)
    dorsal = jnp.asarray(
        [name.startswith("d") for name in BODY_WALL_MUSCLE_NAMES]
    )

    assert activation.shape == (95,)
    assert jnp.all((activation >= 0.0) & (activation <= 1.0))
    assert jnp.any(activation[dorsal] > 0.0)
    assert jnp.any(activation[~dorsal] > 0.0)


def test_periodic_controller_rollout_is_finite_and_differentiable() -> None:
    raw = jnp.asarray([0.0, -0.5, -0.35, 0.0, 0.0])
    body_params = default_muscle_body_params(12)
    controller = decode_periodic_controller(raw)

    final, trajectory, activations = simulate_periodic_controller(
        controller, body_params, steps=40
    )
    gradient = jax.grad(
        lambda value: body_motion_loss(value, body_params, steps=40)
    )(raw)

    assert trajectory.position.shape == (40, 2)
    assert activations.shape == (40, 95)
    assert jnp.all(jnp.isfinite(final.position))
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.linalg.norm(gradient) > 1e-7


def test_negative_spatial_wave_moves_in_positive_x_direction() -> None:
    controller = decode_periodic_controller(
        jnp.asarray([4.0, -0.5, -0.55, 0.0, 0.0])
    )
    final, _, _ = simulate_periodic_controller(
        controller, default_muscle_body_params(12), steps=250
    )

    assert final.position[0] > 0.05
    np.testing.assert_allclose(final.time, 5.0, atol=1e-5)
