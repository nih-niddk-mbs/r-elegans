"""Connect neural activity, 95 body-wall muscles, and body joints."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .actuation import MuscleBodyParams, MuscleBodyState, muscle_body_step

Array = jax.Array


def _canonical_muscle_names() -> tuple[str, ...]:
    return tuple(
        [f"dBWML{index}" for index in range(1, 25)]
        + [f"dBWMR{index}" for index in range(1, 25)]
        + [f"vBWML{index}" for index in range(1, 24)]
        + [f"vBWMR{index}" for index in range(1, 25)]
    )


BODY_WALL_MUSCLE_NAMES = _canonical_muscle_names()


class MuscleProjection(NamedTuple):
    """Fixed anatomical projection from individual muscles to body joints."""

    dorsal_weights: Array
    ventral_weights: Array


class NeuromuscularParams(NamedTuple):
    """Neuron-to-muscle transform in ``[muscle, neuron]`` order.

    ``synapse_signs`` must be -1, 0, or +1. A zero explicitly represents
    unknown or excluded polarity and contributes no drive.
    """

    synapse_weights: Array
    synapse_signs: Array
    neuron_threshold: Array
    neuron_slope: Array
    muscle_threshold: Array
    muscle_slope: Array


def muscle_longitudinal_positions() -> Array:
    """Return normalized anterior-to-posterior positions for all 95 muscles."""

    row_indices = jnp.asarray(
        list(range(24))
        + list(range(24))
        + list(range(23))
        + list(range(24)),
        dtype=jnp.float32,
    )
    return row_indices / 23.0


def build_muscle_projection(num_joints: int = 11) -> MuscleProjection:
    """Build local, normalized interpolation from muscle rows to joints."""

    if num_joints < 1:
        raise ValueError("A muscle projection requires at least one joint")
    positions = muscle_longitudinal_positions()
    joint_positions = jnp.linspace(0.0, 1.0, num_joints)
    spacing = 1.0 if num_joints == 1 else 1.0 / (num_joints - 1)
    local = jnp.maximum(
        1.0 - jnp.abs(joint_positions[:, None] - positions[None, :]) / spacing,
        0.0,
    )
    dorsal_mask = jnp.asarray(
        [name.startswith("d") for name in BODY_WALL_MUSCLE_NAMES]
    )
    ventral_mask = ~dorsal_mask

    def normalized(mask: Array) -> Array:
        weights = local * mask[None, :]
        return weights / jnp.maximum(jnp.sum(weights, axis=1, keepdims=True), 1e-8)

    return MuscleProjection(normalized(dorsal_mask), normalized(ventral_mask))


def project_muscles_to_joints(
    muscle_activation: Array,
    projection: MuscleProjection,
) -> tuple[Array, Array]:
    """Average 95 anatomical muscle activations into joint commands."""

    activation = jnp.asarray(muscle_activation)
    if activation.shape != (len(BODY_WALL_MUSCLE_NAMES),):
        raise ValueError("Expected one activation for each of 95 body-wall muscles")
    if projection.dorsal_weights.shape != projection.ventral_weights.shape:
        raise ValueError("Dorsal and ventral projections must have matching shapes")
    if projection.dorsal_weights.ndim != 2 or projection.dorsal_weights.shape[1] != 95:
        raise ValueError("Muscle projections must have shape [joints, 95]")
    bounded = jnp.clip(activation, 0.0, 1.0)
    return projection.dorsal_weights @ bounded, projection.ventral_weights @ bounded


def validate_neuromuscular_params(
    params: NeuromuscularParams,
    *,
    expected_muscles: int = 95,
) -> None:
    """Validate NMJ shapes and fixed topology constraints."""

    weights = jnp.asarray(params.synapse_weights)
    signs = jnp.asarray(params.synapse_signs)
    if weights.ndim != 2 or weights.shape[0] != expected_muscles:
        raise ValueError("NMJ weights must have shape [95, neurons]")
    if signs.shape != weights.shape:
        raise ValueError("NMJ signs must match NMJ weights")
    if params.neuron_threshold.shape != (weights.shape[1],):
        raise ValueError("Neuron thresholds must have one value per neuron")
    if params.neuron_slope.shape != (weights.shape[1],):
        raise ValueError("Neuron slopes must have one value per neuron")
    if params.muscle_threshold.shape not in ((), (expected_muscles,)):
        raise ValueError("Muscle threshold must be scalar or one value per muscle")
    if params.muscle_slope.shape not in ((), (expected_muscles,)):
        raise ValueError("Muscle slope must be scalar or one value per muscle")
    if bool(jnp.any(weights < 0)):
        raise ValueError("NMJ weights must be nonnegative")
    if bool(jnp.any(jnp.asarray(params.neuron_slope) <= 0)):
        raise ValueError("Neuron slopes must be positive")
    if bool(jnp.any(jnp.asarray(params.muscle_slope) <= 0)):
        raise ValueError("Muscle slopes must be positive")
    if bool(jnp.any(~jnp.isin(signs, jnp.asarray([-1.0, 0.0, 1.0])))):
        raise ValueError("NMJ signs must be -1, 0, or +1")


def muscle_activations_from_voltage(
    voltage: Array,
    params: NeuromuscularParams,
) -> Array:
    """Transform neural voltage into bounded muscle activation."""

    voltage = jnp.asarray(voltage)
    if voltage.shape != params.neuron_threshold.shape:
        raise ValueError("Voltage must have one value per neuron")
    presynaptic = jax.nn.sigmoid(
        (voltage - params.neuron_threshold) / params.neuron_slope
    )
    signed_weights = params.synapse_weights * params.synapse_signs
    scale = jnp.maximum(jnp.sum(params.synapse_weights, axis=1), 1.0)
    drive = (signed_weights @ presynaptic) / scale
    connected = jnp.any(params.synapse_weights > 0.0, axis=1)
    raw_activation = jax.nn.sigmoid(
        (drive - params.muscle_threshold) / params.muscle_slope
    )
    resting_activation = jax.nn.sigmoid(
        -params.muscle_threshold / params.muscle_slope
    )
    activation = jnp.clip(
        (raw_activation - resting_activation)
        / jnp.maximum(1.0 - resting_activation, 1e-8),
        0.0,
        1.0,
    )
    return jnp.where(connected, activation, 0.0)


def neuromuscular_body_step(
    state: MuscleBodyState,
    body_params: MuscleBodyParams,
    voltage: Array,
    neuromuscular_params: NeuromuscularParams,
    projection: MuscleProjection,
    dt: float | Array,
) -> tuple[MuscleBodyState, Array]:
    """Advance the body directly from neural voltage and return muscle activity."""

    muscle_activation = muscle_activations_from_voltage(
        voltage, neuromuscular_params
    )
    dorsal, ventral = project_muscles_to_joints(
        muscle_activation, projection
    )
    return (
        muscle_body_step(state, body_params, dorsal, ventral, dt),
        muscle_activation,
    )
