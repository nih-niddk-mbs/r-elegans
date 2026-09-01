import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.body import (
    NeuromuscularParams,
    decode_commanded_controller,
    muscle_activations_from_voltage,
)
from r_elegans.brain import (
    MOTOR_FEATURE_COUNT,
    motor_command_features,
    neural_motor_loss,
    neural_motor_voltage,
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


def test_motor_features_encode_commands_and_time() -> None:
    commands = jnp.asarray([[1.0, 0.0], [-0.5, 0.5]])
    times = jnp.linspace(0.0, 1.0, 6)

    features = motor_command_features(
        commands, times, decode_commanded_controller(RAW_GAIT)
    )

    assert features.shape == (2, 6, MOTOR_FEATURE_COUNT)
    np.testing.assert_allclose(features[..., 0], 1.0)
    np.testing.assert_allclose(features[0, :, 2], 0.0)
    np.testing.assert_allclose(features[1, :, 1], 0.0)


def test_neural_voltage_and_nmj_projection_support_batches() -> None:
    features = jnp.ones((2, 3, MOTOR_FEATURE_COUNT))
    coefficients = jnp.zeros((4, MOTOR_FEATURE_COUNT))
    voltage = neural_motor_voltage(coefficients, features)
    weights = jnp.ones((95, 4))
    params = NeuromuscularParams(
        synapse_weights=weights,
        synapse_signs=jnp.ones_like(weights),
        neuron_threshold=jnp.full((4,), -20.0),
        neuron_slope=jnp.full((4,), 5.0),
        muscle_threshold=jnp.full((95,), 0.05),
        muscle_slope=jnp.full((95,), 0.1),
    )
    muscles = muscle_activations_from_voltage(voltage, params)

    assert voltage.shape == (2, 3, 4)
    assert muscles.shape == (2, 3, 95)
    assert jnp.all((muscles >= 0.0) & (muscles <= 1.0))


def test_supervised_neural_motor_loss_is_differentiable() -> None:
    features = jnp.ones((1, 4, MOTOR_FEATURE_COUNT))
    coefficients = jnp.zeros((3, MOTOR_FEATURE_COUNT))
    weights = jnp.ones((95, 3))
    params = NeuromuscularParams(
        synapse_weights=weights,
        synapse_signs=jnp.ones_like(weights),
        neuron_threshold=jnp.full((3,), -20.0),
        neuron_slope=jnp.full((3,), 5.0),
        muscle_threshold=jnp.full((95,), 0.05),
        muscle_slope=jnp.full((95,), 0.1),
    )
    targets = jnp.full((1, 4, 95), 0.5)

    loss, gradient = jax.value_and_grad(
        lambda value: neural_motor_loss(
            value, features, targets, params, jnp.ones((3,), dtype=bool)
        )[0]
    )(coefficients)

    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.linalg.norm(gradient) > 1e-7
