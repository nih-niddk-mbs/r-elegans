"""Connectome data structures and validation.

Downloading and parsing Cook et al. data is intentionally deferred until the
source artifact, license, checksum, and neuron/muscle mappings are pinned.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.assets import load_asset_document


class Connectome(NamedTuple):
    """Canonical neuron-neuron connectivity in ``[post, pre]`` order."""

    neuron_ids: tuple[str, ...]
    chemical_counts: jax.Array
    gap_counts: jax.Array


def validate_connectome(
    neuron_ids: Sequence[str],
    chemical_counts: object,
    gap_counts: object,
    *,
    expected_neurons: int = 302,
) -> None:
    """Raise ``ValueError`` when a processed connectome violates invariants."""

    chemical = np.asarray(chemical_counts)
    gap = np.asarray(gap_counts)
    expected_shape = (expected_neurons, expected_neurons)
    if len(neuron_ids) != expected_neurons or len(set(neuron_ids)) != expected_neurons:
        raise ValueError("neuron_ids must contain unique canonical identifiers")
    if chemical.shape != expected_shape or gap.shape != expected_shape:
        raise ValueError(f"connectivity matrices must have shape {expected_shape}")
    if not np.all(np.isfinite(chemical)) or not np.all(np.isfinite(gap)):
        raise ValueError("connectivity matrices must be finite")
    if np.any(chemical < 0) or np.any(gap < 0):
        raise ValueError("synapse counts must be nonnegative")
    if not np.array_equal(gap, gap.T):
        raise ValueError("gap-junction counts must be symmetric")


def load_connectome() -> Connectome:
    """Load the bundled 302-neuron Cook topology in ``[post, pre]`` order."""

    document = load_asset_document("runtime_model_v1.json")
    neuron_ids = tuple(document["neuron_ids"])
    sparse = document["connectome"]
    chemical = np.zeros((302, 302), dtype=np.float32)
    chemical[
        np.asarray(sparse["chemical_post"], dtype=np.int32),
        np.asarray(sparse["chemical_pre"], dtype=np.int32),
    ] = np.asarray(sparse["chemical_count"], dtype=np.float32)
    gap = np.zeros((302, 302), dtype=np.float32)
    gap_a = np.asarray(sparse["gap_neuron_a"], dtype=np.int32)
    gap_b = np.asarray(sparse["gap_neuron_b"], dtype=np.int32)
    gap_count = np.asarray(sparse["gap_count"], dtype=np.float32)
    gap[gap_a, gap_b] = gap_count
    gap[gap_b, gap_a] = gap_count
    result = Connectome(neuron_ids, jnp.asarray(chemical), jnp.asarray(gap))
    validate_connectome(*result)
    return result
