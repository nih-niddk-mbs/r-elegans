"""A small, real chemotaxis subcircuit sliced from the bundled connectome.

Rather than fitting all 302 neurons, this selects fourteen identified neurons
along the textbook klinotaxis pathway -- chemosensory AWC/ASE, first- and
second-order interneurons AIY/AIZ, the head-oscillator integrator RIA, and
dorsal/ventral head-motor readout RMDD/RMDV -- and slices the real anatomical
topology (:class:`r_elegans.data.connectome.Connectome`) down to just these
fourteen rows/columns. The connectivity is not assumed: every edge counted
here is a real entry in the bundled Cook et al. topology.

The dorsal/ventral (not left/right) RMD subclasses are used for the readout
because the simulator's ``steering`` command is a dorsal-minus-ventral bending
bias (see ``r_elegans.body.fitting.periodic_muscle_activations``), and RIA is
the documented site where left/right sensory asymmetry is transformed into
that dorsal/ventral bias -- so reading out RMDD/RMDV asymmetry is reading out
exactly the quantity the rest of the pipeline expects.

Parameters follow ``r_elegans.brain.dynamics.NeuralParams``'s existing
normalized (non-mV/ms) unit convention, matching the constants already used
by ``tests/test_brain.py``'s and ``tests/test_differentiability.py``'s
synthetic fixtures -- the closest existing precedent, since no real fit of
this network exists yet. ``synapse_reversal`` (excitatory/inhibitory
character) has no real per-edge sign dataset in this repo; it defaults to
mostly excitatory with the one well-established literature fact (AIY inhibits
AIZ) hand-set, and is fully trainable thereafter -- a disclosed judgment call,
not a fitted quantity.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp

from r_elegans.data.connectome import Connectome

from .dynamics import NeuralParams

Array = jax.Array

SUBCIRCUIT_NEURON_NAMES: tuple[str, ...] = (
    "AWCL", "AWCR", "ASEL", "ASER",  # 0-3: chemosensory (sensory)
    "AIYL", "AIYR",                  # 4-5: first-order interneuron
    "AIZL", "AIZR",                  # 6-7: second-order interneuron
    "RIAL", "RIAR",                  # 8-9: L/R-to-D/V integrator
    "RMDDL", "RMDDR",                # 10-11: dorsal head-motor readout
    "RMDVL", "RMDVR",                # 12-13: ventral head-motor readout
)
SENSORY_LOCAL_INDICES: tuple[int, ...] = (0, 1, 2, 3)
READOUT_DORSAL_LOCAL_INDICES: tuple[int, ...] = (10, 11)
READOUT_VENTRAL_LOCAL_INDICES: tuple[int, ...] = (12, 13)
INHIBITORY_NEURON_NAMES: tuple[str, ...] = ("AIYL", "AIYR")

_DEFAULT_TIME_CONSTANT = 0.1
_DEFAULT_SLOPE = 0.1


class RawSubcircuitParams(NamedTuple):
    """Trainable subcircuit parameters, with positivity-constrained fields raw."""

    raw_chemical: Array
    chemical_mask: Array
    raw_gap: Array
    gap_mask: Array
    leak_reversal: Array
    synapse_reversal: Array
    raw_time_constant: Array
    threshold: Array
    raw_slope: Array


def _inverse_softplus(target: float, floor: float) -> Array:
    """Return the unconstrained value whose ``floor + softplus(x)`` is ``target``."""

    if target <= floor:
        raise ValueError("target must exceed floor for a valid inverse-softplus")
    return jnp.log(jnp.expm1(jnp.asarray(target - floor)))


def decode_subcircuit_params(
    raw: RawSubcircuitParams,
    *,
    tau_min: float = 0.02,
    slope_min: float = 0.02,
) -> NeuralParams:
    """Map trainable raw fields to a physically valid ``NeuralParams``."""

    return NeuralParams(
        raw_chemical=raw.raw_chemical,
        chemical_mask=raw.chemical_mask,
        raw_gap=raw.raw_gap,
        gap_mask=raw.gap_mask,
        leak_reversal=raw.leak_reversal,
        synapse_reversal=raw.synapse_reversal,
        time_constant=tau_min + jax.nn.softplus(raw.raw_time_constant),
        threshold=raw.threshold,
        slope=slope_min + jax.nn.softplus(raw.raw_slope),
    )


def build_subcircuit_params(
    connectome: Connectome,
    neuron_names: Sequence[str] = SUBCIRCUIT_NEURON_NAMES,
    *,
    tau_min: float = 0.02,
    slope_min: float = 0.02,
) -> RawSubcircuitParams:
    """Slice the real connectome's topology down to ``neuron_names``.

    Anatomical masks come directly from nonzero synapse/gap-junction counts
    in ``connectome`` -- restricted to the chosen neurons, not invented. All
    other fields are initialized to the same normalized-unit constants this
    repository's existing synthetic ``NeuralParams`` test fixtures use, since
    no real fit of this network exists yet; every field remains trainable.
    """

    name_to_index = {name: index for index, name in enumerate(connectome.neuron_ids)}
    try:
        global_indices = jnp.asarray(
            [name_to_index[name] for name in neuron_names], dtype=jnp.int32
        )
    except KeyError as error:
        raise ValueError(
            f"Neuron {error.args[0]!r} not found in connectome.neuron_ids"
        ) from error

    chemical_sub = connectome.chemical_counts[global_indices][:, global_indices]
    gap_sub = connectome.gap_counts[global_indices][:, global_indices]
    chemical_mask = (chemical_sub > 0).astype(jnp.float32)
    gap_mask = (gap_sub > 0).astype(jnp.float32)

    count = len(neuron_names)
    synapse_reversal = jnp.full((count,), 0.2)
    inhibitory_local = [
        neuron_names.index(name)
        for name in INHIBITORY_NEURON_NAMES
        if name in neuron_names
    ]
    if inhibitory_local:
        synapse_reversal = synapse_reversal.at[jnp.asarray(inhibitory_local)].set(-0.8)

    return RawSubcircuitParams(
        raw_chemical=jnp.full((count, count), -2.0),
        chemical_mask=chemical_mask,
        raw_gap=jnp.full((count, count), -3.0),
        gap_mask=gap_mask,
        leak_reversal=jnp.full((count,), -0.35),
        synapse_reversal=synapse_reversal,
        raw_time_constant=jnp.full(
            (count,), _inverse_softplus(_DEFAULT_TIME_CONSTANT, tau_min)
        ),
        threshold=jnp.full((count,), -0.2),
        raw_slope=jnp.full((count,), _inverse_softplus(_DEFAULT_SLOPE, slope_min)),
    )


__all__ = [
    "INHIBITORY_NEURON_NAMES",
    "READOUT_DORSAL_LOCAL_INDICES",
    "READOUT_VENTRAL_LOCAL_INDICES",
    "RawSubcircuitParams",
    "SENSORY_LOCAL_INDICES",
    "SUBCIRCUIT_NEURON_NAMES",
    "build_subcircuit_params",
    "decode_subcircuit_params",
]
