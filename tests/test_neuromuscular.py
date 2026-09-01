from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from r_elegans.body import (
    BODY_WALL_MUSCLE_NAMES,
    NeuromuscularParams,
    build_muscle_projection,
    default_muscle_body_params,
    initialize_muscle_body,
    muscle_activations_from_voltage,
    neuromuscular_body_step,
    project_muscles_to_joints,
    validate_neuromuscular_params,
)
from r_elegans.data import (
    NeuromuscularConnectome,
    infer_nmj_signs,
    load_neuromuscular_connectome,
    save_neuromuscular_connectome,
)


def synthetic_params() -> NeuromuscularParams:
    weights = (
        jnp.zeros((95, 2))
        .at[0, 0]
        .set(4.0)
        .at[48, :]
        .set(4.0)
    )
    signs = (
        jnp.zeros((95, 2))
        .at[0, 0]
        .set(1.0)
        .at[48, 0]
        .set(1.0)
        .at[48, 1]
        .set(-1.0)
    )
    return NeuromuscularParams(
        synapse_weights=weights,
        synapse_signs=signs,
        neuron_threshold=jnp.zeros((2,)),
        neuron_slope=jnp.ones((2,)),
        muscle_threshold=jnp.zeros((95,)),
        muscle_slope=jnp.ones((95,)),
    )


def test_canonical_body_wall_muscle_inventory() -> None:
    assert len(BODY_WALL_MUSCLE_NAMES) == 95
    assert len(set(BODY_WALL_MUSCLE_NAMES)) == 95
    assert BODY_WALL_MUSCLE_NAMES[0] == "dBWML1"
    assert BODY_WALL_MUSCLE_NAMES[-1] == "vBWMR24"
    assert sum(name.startswith("d") for name in BODY_WALL_MUSCLE_NAMES) == 48
    assert sum(name.startswith("v") for name in BODY_WALL_MUSCLE_NAMES) == 47


def test_projection_is_local_normalized_and_side_specific() -> None:
    projection = build_muscle_projection(11)

    assert projection.dorsal_weights.shape == (11, 95)
    np.testing.assert_allclose(projection.dorsal_weights.sum(axis=1), 1.0, atol=2e-7)
    np.testing.assert_allclose(projection.ventral_weights.sum(axis=1), 1.0, atol=2e-7)
    np.testing.assert_allclose(projection.dorsal_weights[:, 48:], 0.0)
    np.testing.assert_allclose(projection.ventral_weights[:, :48], 0.0)


def test_uniform_muscles_project_to_uniform_joint_commands() -> None:
    projection = build_muscle_projection(11)
    activation = jnp.asarray(
        [1.0 if name.startswith("d") else 0.0 for name in BODY_WALL_MUSCLE_NAMES]
    )

    dorsal, ventral = project_muscles_to_joints(activation, projection)

    np.testing.assert_allclose(dorsal, 1.0, atol=1e-6)
    np.testing.assert_allclose(ventral, 0.0, atol=1e-6)


def test_signed_synapses_drive_excitation_and_inhibition() -> None:
    params = synthetic_params()
    resting = muscle_activations_from_voltage(jnp.asarray([-30.0, -30.0]), params)
    excited = muscle_activations_from_voltage(jnp.asarray([8.0, -8.0]), params)
    inhibited = muscle_activations_from_voltage(jnp.asarray([8.0, 8.0]), params)

    np.testing.assert_allclose(resting, 0.0, atol=1e-7)
    assert excited[0] > resting[0]
    assert inhibited[48] < excited[48]
    np.testing.assert_allclose(excited[1], 0.0)


def test_neuromuscular_transform_is_jittable_and_differentiable() -> None:
    params = synthetic_params()

    def total_activation(voltage: jax.Array) -> jax.Array:
        return jnp.sum(muscle_activations_from_voltage(voltage, params))

    voltage = jnp.asarray([0.2, -0.1])
    value = jax.jit(total_activation)(voltage)
    gradient = jax.grad(total_activation)(voltage)

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(jnp.abs(gradient) > 1e-7)


def test_neuromuscular_body_step_connects_voltage_to_body() -> None:
    params = synthetic_params()
    state = initialize_muscle_body(12)

    next_state, activation = jax.jit(neuromuscular_body_step)(
        state,
        default_muscle_body_params(12),
        jnp.asarray([8.0, -8.0]),
        params,
        build_muscle_projection(11),
        jnp.asarray(0.01),
    )

    assert activation.shape == (95,)
    assert next_state.joint_angles.shape == (11,)
    assert jnp.all(jnp.isfinite(next_state.joint_angles))
    assert jnp.any(next_state.joint_angles > 0.0)


def test_validation_rejects_ambiguous_signs_and_zero_slopes() -> None:
    params = synthetic_params()
    with pytest.raises(ValueError, match="signs"):
        validate_neuromuscular_params(
            params._replace(
                synapse_signs=params.synapse_signs.at[0, 0].set(0.5)
            )
        )
    with pytest.raises(ValueError, match="slopes"):
        validate_neuromuscular_params(params._replace(neuron_slope=jnp.zeros((2,))))


def test_processed_connectome_round_trip(tmp_path: Path) -> None:
    connectome = NeuromuscularConnectome(
        neuron_ids=("A", "B"),
        muscle_ids=BODY_WALL_MUSCLE_NAMES,
        chemical_counts=jnp.arange(190, dtype=jnp.float32).reshape(95, 2),
        synapse_signs=jnp.where(
            jnp.arange(190, dtype=jnp.float32).reshape(95, 2) > 0, 1.0, 0.0
        ),
    )
    path = tmp_path / "nmj.npz"

    save_neuromuscular_connectome(connectome, path)
    loaded = load_neuromuscular_connectome(root=tmp_path, relative_path="nmj.npz")

    assert loaded.neuron_ids == connectome.neuron_ids
    assert loaded.muscle_ids == connectome.muscle_ids
    np.testing.assert_array_equal(loaded.chemical_counts, connectome.chemical_counts)
    np.testing.assert_array_equal(loaded.synapse_signs, connectome.synapse_signs)


def test_nmj_sign_inference_is_conservative_and_topology_masked() -> None:
    counts = jnp.zeros((95, 4)).at[0, :].set(1.0).at[1, 0].set(2.0)
    connectome = NeuromuscularConnectome(
        neuron_ids=("DA1", "VD1", "AVA", "DB1"),
        muscle_ids=BODY_WALL_MUSCLE_NAMES,
        chemical_counts=counts,
    )
    transmitters = {
        "DA1": "ACh",
        "VD1": "GABA",
        "AVA": "Glu",
        "class:DB": "ACh",
    }

    signs = infer_nmj_signs(connectome, transmitters)

    np.testing.assert_array_equal(signs[0], [1.0, -1.0, 0.0, 1.0])
    np.testing.assert_array_equal(signs[1], [1.0, 0.0, 0.0, 0.0])
