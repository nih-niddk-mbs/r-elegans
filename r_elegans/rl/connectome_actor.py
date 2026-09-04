"""A recurrent connectome-subcircuit actor sharing the analytic controller's
``[speed, steering]`` interface.

The subcircuit's voltage is genuine recurrent state that must persist across
environment steps -- it lives here, in the actor, rather than inside the
Gymnax environment, because ``r_elegans.envs.gymnax_petri_dish.PetriDishGymnaxEnv.step_env``
deliberately stops gradients on everything it returns (the environment is a
black box, by design, for model-free RL). Any trainable connectome parameter
must therefore live in ``agent.actor`` so that ``r_elegans.rl.training``'s
loss functions -- which freshly and differentiably recompute the action
distribution from stored ``(obs, raw_action, carry_in)`` -- can differentiate
it. See ``r_elegans.rl.actor_interface`` for how this voltage carry is
threaded through the rollout and update loop alongside the analytic
controller's (trivial, zero-size) carry.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp

from r_elegans.brain.circuit import (
    RawSubcircuitParams,
    build_subcircuit_params,
    decode_subcircuit_params,
)
from r_elegans.brain.dynamics import integrate_neural_fixed_step
from r_elegans.brain.sensory import (
    SensoryGains,
    init_sensory_gains,
    inject_sensory_current,
    steering_from_voltage,
)
from r_elegans.data.connectome import Connectome

Array = jax.Array


class RecurrentConnectomeActorParams(NamedTuple):
    """The trainable subcircuit plus its sensory/readout/speed interface."""

    neural_params: RawSubcircuitParams
    sensory_gains: SensoryGains
    readout_scale: Array
    readout_bias: Array
    base_speed_raw: Array
    food_slowing_raw: Array
    log_std: Array


def init_connectome_actor_params(
    connectome: Connectome,
    key: Array,
    *,
    log_std: tuple[float, float] = (-1.0, -1.0),
) -> RecurrentConnectomeActorParams:
    """Slice the real connectome and initialize the trainable interface.

    ``base_speed_raw=0.0``/``food_slowing_raw=-2.0`` decode (via the same
    sigmoid conventions as ``r_elegans.envs.petri_dish.decode_sensory_policy``)
    to the same initial speed behavior used to seed the analytic controller's
    fits, for a comparable starting point.
    """

    return RecurrentConnectomeActorParams(
        neural_params=build_subcircuit_params(connectome),
        sensory_gains=init_sensory_gains(key),
        readout_scale=jnp.asarray(1.0),
        readout_bias=jnp.asarray(0.0),
        base_speed_raw=jnp.asarray(0.0),
        food_slowing_raw=jnp.asarray(-2.0),
        log_std=jnp.asarray(log_std),
    )


def initial_voltage(actor: RecurrentConnectomeActorParams) -> Array:
    """The subcircuit's resting voltage -- used to (re)start each episode."""

    return decode_subcircuit_params(actor.neural_params).leak_reversal


def connectome_action_mean_and_next_voltage(
    actor: RecurrentConnectomeActorParams,
    voltage: Array,
    observation: Array,
    *,
    dt: float = 0.02,
    substeps: int = 4,
) -> tuple[Array, Array]:
    """One shared forward step: advance the subcircuit, read out ``[speed, steering]``.

    Reused, unmodified, by both the stochastic rollout step and the loss
    functions' fresh, differentiable recomputation (see
    ``r_elegans.rl.actor_interface.make_connectome_actor_interface``).
    """

    params = decode_subcircuit_params(actor.neural_params)
    external_current = inject_sensory_current(actor.sensory_gains, observation)
    next_voltage = integrate_neural_fixed_step(
        voltage, params, external_current, dt, substeps=substeps
    )
    steering = steering_from_voltage(next_voltage, actor.readout_scale, actor.readout_bias)

    relative_concentration = observation[4]
    base_speed = 0.5 + 0.5 * jax.nn.sigmoid(actor.base_speed_raw)
    food_slowing = jax.nn.sigmoid(actor.food_slowing_raw)
    speed = base_speed * (1.0 - food_slowing * relative_concentration)

    return jnp.stack((speed, steering)), next_voltage


def actor_to_arrays(actor: RecurrentConnectomeActorParams) -> dict[str, Array]:
    """Flatten to a flat, ``np.savez``-compatible dict for checkpointing."""

    neural = actor.neural_params
    gains = actor.sensory_gains
    return {
        "raw_chemical": neural.raw_chemical,
        "chemical_mask": neural.chemical_mask,
        "raw_gap": neural.raw_gap,
        "gap_mask": neural.gap_mask,
        "leak_reversal": neural.leak_reversal,
        "synapse_reversal": neural.synapse_reversal,
        "raw_time_constant": neural.raw_time_constant,
        "threshold": neural.threshold,
        "raw_slope": neural.raw_slope,
        "sensory_weights": gains.weights,
        "sensory_bias": gains.bias,
        "readout_scale": actor.readout_scale,
        "readout_bias": actor.readout_bias,
        "base_speed_raw": actor.base_speed_raw,
        "food_slowing_raw": actor.food_slowing_raw,
        "log_std": actor.log_std,
    }


def actor_from_arrays(data: Mapping[str, Array]) -> RecurrentConnectomeActorParams:
    """Inverse of :func:`actor_to_arrays`, e.g. for loading a saved ``.npz``."""

    neural = RawSubcircuitParams(
        raw_chemical=jnp.asarray(data["raw_chemical"]),
        chemical_mask=jnp.asarray(data["chemical_mask"]),
        raw_gap=jnp.asarray(data["raw_gap"]),
        gap_mask=jnp.asarray(data["gap_mask"]),
        leak_reversal=jnp.asarray(data["leak_reversal"]),
        synapse_reversal=jnp.asarray(data["synapse_reversal"]),
        raw_time_constant=jnp.asarray(data["raw_time_constant"]),
        threshold=jnp.asarray(data["threshold"]),
        raw_slope=jnp.asarray(data["raw_slope"]),
    )
    gains = SensoryGains(
        weights=jnp.asarray(data["sensory_weights"]),
        bias=jnp.asarray(data["sensory_bias"]),
    )
    return RecurrentConnectomeActorParams(
        neural_params=neural,
        sensory_gains=gains,
        readout_scale=jnp.asarray(data["readout_scale"]),
        readout_bias=jnp.asarray(data["readout_bias"]),
        base_speed_raw=jnp.asarray(data["base_speed_raw"]),
        food_slowing_raw=jnp.asarray(data["food_slowing_raw"]),
        log_std=jnp.asarray(data["log_std"]),
    )


__all__ = [
    "RecurrentConnectomeActorParams",
    "actor_from_arrays",
    "actor_to_arrays",
    "connectome_action_mean_and_next_voltage",
    "init_connectome_actor_params",
    "initial_voltage",
]
