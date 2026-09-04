"""Sensory transduction into, and steering readout from, the chemotaxis subcircuit.

Honest accounting of what is biologically grounded here versus engineered:

- ``response`` (adapted log-concentration) and ``derivative`` (its recent
  rate of change) are the genuine transduced-signal quantities real ASE/AWC
  neurons are documented to encode as ON/OFF adaptation -- though the
  adaptation time constant that produces them (see
  ``r_elegans.envs.petri_dish``) is itself engineered, not fit to recordings.
- ``sin(phase)``/``cos(phase)`` is an explicit engineered stand-in for
  proprioceptive/head-oscillator coupling: gait phase is not itself a
  modeled biological quantity anywhere in this repository.
- AWC and ASE currently receive the same underlying ``response``/
  ``derivative`` signal (through independently trainable per-neuron gains)
  because the environment does not yet model distinct per-modality
  transduction -- a disclosed simplification, not a claim that these two
  chemosensory classes are redundant in the real animal.
- ``relative_concentration`` (a proxy for food proximity) is deliberately
  *not* injected into the subcircuit -- see the speed/steering split below.

Only ``steering`` is connectome-driven in this first pass. ``speed`` keeps
the existing simple food-proximity-slowing formula
(``r_elegans.envs.petri_dish.decode_sensory_policy``'s
``base_speed * (1 - food_slowing * relative_concentration)``), computed
directly from plain trainable scalars rather than routed through the
subcircuit. This keeps scope and risk bounded to steering, which is what
klinotaxis is actually about, and keeps a saturated/runaway subcircuit
voltage from also breaking forward locomotion.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .circuit import (
    READOUT_DORSAL_LOCAL_INDICES,
    READOUT_VENTRAL_LOCAL_INDICES,
    SENSORY_LOCAL_INDICES,
    SUBCIRCUIT_NEURON_NAMES,
)

Array = jax.Array

SUBCIRCUIT_NEURON_COUNT = len(SUBCIRCUIT_NEURON_NAMES)
SENSORY_OBSERVATION_CHANNELS = 4  # response, derivative, sin(phase), cos(phase)


class SensoryGains(NamedTuple):
    """Trainable map from sensed observation channels to sensory current."""

    weights: Array  # [len(SENSORY_LOCAL_INDICES), SENSORY_OBSERVATION_CHANNELS]
    bias: Array     # [len(SENSORY_LOCAL_INDICES)]


def init_sensory_gains(key: Array, *, scale: float = 0.1) -> SensoryGains:
    """Small random initial gains; supervised pretraining shapes these."""

    sensory_count = len(SENSORY_LOCAL_INDICES)
    weights = scale * jax.random.normal(
        key, (sensory_count, SENSORY_OBSERVATION_CHANNELS)
    )
    return SensoryGains(weights=weights, bias=jnp.zeros((sensory_count,)))


def inject_sensory_current(gains: SensoryGains, observation: Array) -> Array:
    """Return an external-current vector, nonzero only at sensory neurons.

    ``observation`` is ``[response, derivative, sin(phase), cos(phase),
    relative_concentration]`` (see ``PetriDishGymnaxEnv.get_obs``); only the
    first four channels are used here.
    """

    obs_channels = observation[:SENSORY_OBSERVATION_CHANNELS]
    sensory_current = gains.weights @ obs_channels + gains.bias
    return jnp.zeros((SUBCIRCUIT_NEURON_COUNT,)).at[
        jnp.asarray(SENSORY_LOCAL_INDICES)
    ].set(sensory_current)


def steering_from_voltage(voltage: Array, scale: Array, bias: Array) -> Array:
    """Read dorsal-minus-ventral RMD voltage into a bounded steering command.

    This is the quantity RIA is documented to compute: a transformation of
    left/right sensory asymmetry into a dorsal/ventral head-bending bias,
    matching the simulator's own dorsal-vs-ventral ``steering`` convention
    (see ``r_elegans.body.fitting.periodic_muscle_activations``).
    """

    dorsal = jnp.mean(voltage[jnp.asarray(READOUT_DORSAL_LOCAL_INDICES)])
    ventral = jnp.mean(voltage[jnp.asarray(READOUT_VENTRAL_LOCAL_INDICES)])
    return jnp.tanh(scale * (dorsal - ventral) + bias)


__all__ = [
    "SENSORY_OBSERVATION_CHANNELS",
    "SUBCIRCUIT_NEURON_COUNT",
    "SensoryGains",
    "init_sensory_gains",
    "inject_sensory_current",
    "steering_from_voltage",
]
