"""Supervised neural-output controllers for anatomical muscle projection."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from r_elegans.body import (
    CommandedControllerParams,
    NeuromuscularParams,
    controller_for_command,
    muscle_activations_from_voltage,
)

Array = jax.Array
MOTOR_FEATURE_COUNT = 13


def motor_command_features(
    commands: Array,
    times: Array,
    gait_params: CommandedControllerParams,
) -> Array:
    """Encode command and gait phase as ``[commands, time, 13]`` features."""

    commands = jnp.asarray(commands)
    times = jnp.asarray(times)
    if commands.ndim != 2 or commands.shape[1] != 2:
        raise ValueError("Commands must have shape [batch, 2]")
    if times.ndim != 1:
        raise ValueError("Times must be one-dimensional")

    def encode(command: Array) -> Array:
        gait = controller_for_command(command, gait_params)
        phase = gait.phase_offset - 2.0 * jnp.pi * gait.frequency * times
        sine, cosine = jnp.sin(phase), jnp.cos(phase)
        speed, steering = command
        forward = jnp.maximum(speed, 0.0)
        reverse = jnp.maximum(-speed, 0.0)
        ones = jnp.ones_like(times)
        return jnp.stack(
            (
                ones,
                forward * ones,
                reverse * ones,
                forward * sine,
                forward * cosine,
                reverse * sine,
                reverse * cosine,
                forward * steering * ones,
                reverse * steering * ones,
                forward * steering * sine,
                forward * steering * cosine,
                reverse * steering * sine,
                reverse * steering * cosine,
            ),
            axis=-1,
        )

    return jax.vmap(encode)(commands)


def neural_motor_voltage(
    coefficients: Array,
    features: Array,
    *,
    voltage_min: float = -60.0,
    voltage_max: float = 20.0,
) -> Array:
    """Generate bounded neuron voltage trajectories from motor features."""

    coefficients = jnp.asarray(coefficients)
    features = jnp.asarray(features)
    if coefficients.ndim != 2 or coefficients.shape[1] != MOTOR_FEATURE_COUNT:
        raise ValueError("Coefficients must have shape [neurons, 13]")
    if features.shape[-1] != MOTOR_FEATURE_COUNT:
        raise ValueError("Motor features must have a trailing dimension of 13")
    logits = jnp.einsum("...f,nf->...n", features, coefficients)
    return voltage_min + (voltage_max - voltage_min) * jax.nn.sigmoid(logits)


def neural_motor_loss(
    coefficients: Array,
    features: Array,
    target_muscles: Array,
    neuromuscular_params: NeuromuscularParams,
    trainable_neurons: Array,
) -> tuple[Array, tuple[Array, Array]]:
    """Return supervised muscle loss and predicted neural/muscle trajectories."""

    mask = jnp.asarray(trainable_neurons)[:, None]
    resting = jnp.zeros_like(coefficients).at[:, 0].set(-4.0)
    effective_coefficients = resting + mask * coefficients
    voltage = neural_motor_voltage(effective_coefficients, features)
    muscles = muscle_activations_from_voltage(voltage, neuromuscular_params)
    muscle_mse = jnp.mean((muscles - target_muscles) ** 2)
    presynaptic = jax.nn.sigmoid(
        (voltage - neuromuscular_params.neuron_threshold)
        / neuromuscular_params.neuron_slope
    )
    activity_penalty = 1e-5 * jnp.mean(presynaptic**2)
    coefficient_penalty = 1e-7 * jnp.mean((mask * coefficients) ** 2)
    return (
        muscle_mse + activity_penalty + coefficient_penalty,
        (voltage, muscles),
    )
