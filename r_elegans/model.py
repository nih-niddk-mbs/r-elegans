"""Load the compact, versioned model checkpoint bundled with the package."""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp

from r_elegans.assets import load_asset_document
from r_elegans.body import (
    CommandedControllerParams,
    NeuromuscularParams,
    decode_commanded_controller,
)
from r_elegans.data import load_builtin_neuromuscular_connectome
from r_elegans.data.connectome import Connectome, load_connectome

Array = jax.Array
BUILTIN_MODEL_ASSET = "runtime_model_v1.json"


class RuntimeModel(NamedTuple):
    """Parameters required for inference with the current closed-loop model."""

    model_id: str
    neuron_ids: tuple[str, ...]
    connectome: Connectome
    gait_params: CommandedControllerParams
    neural_motor_coefficients: Array
    neuromuscular_params: NeuromuscularParams
    raw_sensory_policy: Array


@lru_cache(maxsize=1)
def load_builtin_model() -> RuntimeModel:
    """Load the repository-shipped checkpoint without an external data root."""

    document = load_asset_document(BUILTIN_MODEL_ASSET)
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported built-in model schema")
    connectome = load_builtin_neuromuscular_connectome()
    neural_connectome = load_connectome()
    neuron_ids = tuple(document["neuron_ids"])
    if neuron_ids != connectome.neuron_ids:
        raise ValueError("Built-in neural coefficients and NMJ ordering differ")
    if neuron_ids != neural_connectome.neuron_ids:
        raise ValueError("Built-in neural coefficients and connectome ordering differ")

    motor = document["neural_motor"]
    active_indices = jnp.asarray(motor["active_neuron_index"], dtype=jnp.int32)
    active_coefficients = jnp.asarray(
        motor["active_coefficients"], dtype=jnp.float32
    )
    coefficients = jnp.zeros((302, 13), dtype=jnp.float32)
    coefficients = coefficients.at[:, 0].set(-4.0)
    coefficients = coefficients.at[active_indices].add(active_coefficients)

    return RuntimeModel(
        model_id=str(document["model_id"]),
        neuron_ids=neuron_ids,
        connectome=neural_connectome,
        gait_params=decode_commanded_controller(
            jnp.asarray(document["body_controller_raw"], dtype=jnp.float32)
        ),
        neural_motor_coefficients=coefficients,
        neuromuscular_params=NeuromuscularParams(
            synapse_weights=connectome.chemical_counts,
            synapse_signs=connectome.synapse_signs,
            neuron_threshold=jnp.full((302,), -20.0),
            neuron_slope=jnp.full((302,), 5.0),
            muscle_threshold=jnp.full((95,), 0.05),
            muscle_slope=jnp.full((95,), 0.1),
        ),
        raw_sensory_policy=jnp.asarray(
            document["sensory_policy_raw"], dtype=jnp.float32
        ),
    )


__all__ = ["BUILTIN_MODEL_ASSET", "RuntimeModel", "load_builtin_model"]
