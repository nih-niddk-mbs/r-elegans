"""Extrasynaptic (peptide) signaling term for single-neuron stimulation simulations.

Model: each sender neuron's activity-dependent release fraction r_s = sigmoid((V_s - threshold_s) / slope_s) (the same release
function as chemical edges, used as a proxy for dense-core-vesicle release) is low-pass filtered with one shared slow time
constant to a peptide level p_s. A receiver neuron r receives a modulatory membrane conductance
    g_exc[r] = unit * sum_s C_exc[r, s] p_s   (reversal E_exc),   g_inh[r] likewise (reversal E_inh),
    C[r, s] = sum_k gain_k * receiver[k, r] * sender[k, s]
over peptide-receptor pairs k (Beets EC50 <= cutoff, CeNGEN expression; `scripts/build_extrasynaptic_asset.py`). Excitatory and
inhibitory gains are separate non-negative parameters per pair, so no sign is assumed. The term enters only through
`MembraneConductanceInput`; the core network step is unchanged. Gains, time constant and reversals are project priors or fitted,
not measurements; range/diffusion is not modeled (all-to-all within expression).
"""
import json
from importlib.resources import files
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .conductance_network import MembraneConductanceInput

E_EXCITATORY_MV, E_INHIBITORY_MV = 0., -85.
RAW_GAIN_INIT, RAW_GAIN_MAX, RAW_GAIN_MIN = -9., 2.95, -12.  # softplus(-9) ~ 1e-4 (tonic peptide conductance ~0.1 x leak at init), softplus(2.95) ~ 3
TAU_RANGE_S = (1., 30.)


class PeptideParams(NamedTuple):
    raw_exc: jax.Array
    raw_inh: jax.Array
    log_tau_s: jax.Array


def load_extrasynaptic_asset():
    package = files('r_elegans.assets')
    with package.joinpath('extrasynaptic_v1.npz').open('rb') as stream, np.load(stream, allow_pickle=False) as data:
        arrays = {k: data[k] for k in ('sender', 'receiver', 'ec50_nM', 'neuron_ids')}
    metadata = json.loads(package.joinpath('extrasynaptic_v1.json').read_text(encoding='utf-8'))
    return arrays | dict(metadata=metadata)


def initial_peptide_params(pairs, tau_s=5., dtype=float):
    return PeptideParams(jnp.full(pairs, RAW_GAIN_INIT, dtype), jnp.full(pairs, RAW_GAIN_INIT, dtype), jnp.asarray(np.log(tau_s), dtype))


def clip_peptide_params(p):
    return PeptideParams(jnp.clip(p.raw_exc, RAW_GAIN_MIN, RAW_GAIN_MAX), jnp.clip(p.raw_inh, RAW_GAIN_MIN, RAW_GAIN_MAX),
                         jnp.clip(p.log_tau_s, np.log(TAU_RANGE_S[0]), np.log(TAU_RANGE_S[1])))


def coupling_matrices(p, sender, receiver):
    """C_exc, C_inh [post, pre]; sender/receiver are [K, neurons] 0/1 arrays."""
    receiver, sender = jnp.asarray(receiver, p.raw_exc.dtype), jnp.asarray(sender, p.raw_exc.dtype)
    exc, inh = jax.nn.softplus(p.raw_exc), jax.nn.softplus(p.raw_inh)
    return (receiver.T * exc[None, :]) @ sender, (receiver.T * inh[None, :]) @ sender


def peptide_membrane_input(level, c_exc, c_inh, unit_nS):
    g_exc, g_inh = unit_nS * (c_exc @ level), unit_nS * (c_inh @ level)
    return MembraneConductanceInput(g_exc + g_inh, g_exc * E_EXCITATORY_MV + g_inh * E_INHIBITORY_MV)


def peptide_level_step(level, voltage_mV, fixed, dt_s, log_tau_s):
    release = jax.nn.sigmoid((voltage_mV - fixed.release_threshold_mV) / fixed.release_slope_mV)
    decay = jnp.exp(-dt_s / jnp.exp(log_tau_s))
    return level * decay + (1. - decay) * release
