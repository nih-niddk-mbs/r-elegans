import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.body import (
    NeuromuscularParams,
    decode_commanded_controller,
    default_muscle_body_params,
)
from r_elegans.envs import (
    decode_sensory_policy,
    default_petri_dish_params,
    food_concentration,
    petri_navigation_loss,
    simulate_neural_petri_dish,
    simulate_petri_dish,
)


RAW_GAIT = jnp.asarray(
    [
        2.34,
        -0.45,
        -0.256,
        0.64,
        0.011,
        2.47,
        -0.535,
        0.369,
        0.714,
        -0.018,
        0.2,
        -0.2,
        0.5,
        0.0,
    ]
)


def test_food_pulse_peaks_at_source_and_diffuses() -> None:
    dish = default_petri_dish_params(jnp.asarray([0.4, -0.2]))
    source_early, relative = food_concentration(
        dish.source_position, jnp.asarray(0.0), dish
    )
    away, _ = food_concentration(jnp.asarray([-0.4, 0.2]), 0.0, dish)
    source_late, _ = food_concentration(dish.source_position, 10.0, dish)

    assert relative == 1.0
    assert source_early > away
    assert source_late < source_early


def test_sensory_policy_has_bounded_outputs() -> None:
    policy = decode_sensory_policy(jnp.asarray([100.0] * 7))

    assert 0.5 <= policy.base_speed <= 1.0
    assert -1.0 <= policy.steering_bias <= 1.0
    assert 0.0 <= policy.food_slowing <= 1.0


def test_direct_petri_rollout_is_finite_and_respects_dish() -> None:
    dish = default_petri_dish_params(jnp.asarray([0.0, 0.8]))
    body_params = default_muscle_body_params(12)
    gait = decode_commanded_controller(RAW_GAIT)

    final, trajectory, observations = simulate_petri_dish(
        jnp.zeros((7,)), dish, body_params, gait, steps=30
    )

    assert trajectory.body.position.shape == (30, 2)
    assert observations.muscle_activation.shape == (30, 95)
    assert jnp.all(jnp.isfinite(observations.head_position))
    assert jnp.linalg.norm(final.body.position) <= dish.radius - dish.body_margin + 1e-6


def test_navigation_loss_is_differentiable() -> None:
    sources = jnp.asarray([[0.0, 0.8], [0.7, -0.2]])
    headings = jnp.asarray([0.0, 1.0])
    body_params = default_muscle_body_params(12)
    gait = decode_commanded_controller(RAW_GAIT)

    loss, gradient = jax.value_and_grad(
        lambda raw: petri_navigation_loss(
            raw, sources, headings, body_params, gait, steps=20
        )
    )(jnp.zeros((7,)))

    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.linalg.norm(gradient) > 1e-7


def test_neural_rollout_records_voltage_and_muscles() -> None:
    neuron_count = 4
    weights = jnp.ones((95, neuron_count))
    neuromuscular = NeuromuscularParams(
        synapse_weights=weights,
        synapse_signs=jnp.ones_like(weights),
        neuron_threshold=jnp.full((neuron_count,), -20.0),
        neuron_slope=jnp.full((neuron_count,), 5.0),
        muscle_threshold=jnp.full((95,), 0.05),
        muscle_slope=jnp.full((95,), 0.1),
    )
    final, _, observations, voltage = simulate_neural_petri_dish(
        jnp.zeros((7,)),
        default_petri_dish_params(jnp.asarray([0.0, 0.8])),
        default_muscle_body_params(12),
        decode_commanded_controller(RAW_GAIT),
        jnp.zeros((neuron_count, 13)),
        neuromuscular,
        steps=3,
    )

    assert voltage.shape == (3, neuron_count)
    assert observations.muscle_activation.shape == (3, 95)
    np.testing.assert_allclose(final.body.time, 0.06, atol=1e-6)
