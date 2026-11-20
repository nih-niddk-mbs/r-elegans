"""Data-based synaptic polarity for the conductance network.

The bundled asset ``edge_polarity_v1.npz`` holds, for each chemical edge of the runtime connectome, a sign from Fenyves et
al. 2020 (NT+receptor prediction first, then transmitter-only) or 0 when no method gives one (complex, unpredicted, absent).
Excitatory and inhibitory reversal values are project priors, not verified literature values. Undetermined edges keep the
network's existing presynaptic reversal ('keep') or take a neutral value; they are never treated as known.
"""
import json
from importlib.resources import files

import jax.numpy as jnp
import numpy as np

EXCITATORY_MV = 0.
INHIBITORY_MV = -85.
NEUTRAL_MV = -40.
SOURCE_NT_RECEPTOR, SOURCE_NT_ONLY = 1, 2


def load_edge_polarity():
    """Sign, source code and conflict flag [post, pre] over the 302 model neurons, plus the asset's metadata."""
    package = files('r_elegans.assets')
    with package.joinpath('edge_polarity_v1.npz').open('rb') as stream, np.load(stream, allow_pickle=False) as data:
        arrays = {k: data[k] for k in ('sign', 'source', 'conflict', 'neuron_ids')}
    metadata = json.loads(package.joinpath('edge_polarity_v1.json').read_text(encoding='utf-8'))
    return arrays | dict(metadata=metadata)


def polarized_reversal(old, sign, *, excitatory_mV=EXCITATORY_MV, inhibitory_mV=INHIBITORY_MV, undetermined='keep', neutral_mV=NEUTRAL_MV):
    """Per-edge [post, pre] reversal from the old reversal (vector or matrix) and edge signs."""
    old = np.asarray(old, float)
    matrix = np.broadcast_to(old[None, :], sign.shape).copy() if old.ndim == 1 else old.copy()
    if undetermined == 'neutral':
        matrix[sign == 0] = neutral_mV
    elif undetermined != 'keep':
        raise ValueError("undetermined must be 'keep' or 'neutral'")
    matrix[sign > 0] = excitatory_mV
    matrix[sign < 0] = inhibitory_mV
    return matrix


def apply_edge_polarity(fixed, **options):
    """Copy of ``fixed`` with a per-edge reversal matrix; nothing else changes."""
    polarity = load_edge_polarity()
    edges = np.asarray(fixed.chemical_mask) > 0
    if np.any((polarity['sign'] != 0) & ~edges):
        raise ValueError('Polarity asset has signs on edges absent from this network')
    matrix = polarized_reversal(fixed.synapse_reversal_mV, polarity['sign'], **options)
    return fixed._replace(synapse_reversal_mV=jnp.asarray(matrix, jnp.asarray(fixed.synapse_reversal_mV).dtype))
