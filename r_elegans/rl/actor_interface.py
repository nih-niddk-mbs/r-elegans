"""A common interface letting ``r_elegans.rl.training`` train either actor.

The analytic seven-parameter controller (:mod:`r_elegans.rl.policy`) is a
pure, memoryless function of the current observation. The recurrent
connectome-subcircuit actor (:mod:`r_elegans.rl.connectome_actor`) instead
carries genuine hidden state (neuron voltage) across environment steps, like
an RNN policy's carry. ``ActorInterface`` lets ``training.py``'s rollout and
loss functions handle both uniformly: for the analytic controller, ``carry``
is a fixed, zero-size array; for the connectome actor, it is the subcircuit's
voltage. PPO and A2C recompute complete time-ordered actor sequences during
optimization, reset voltage at episode boundaries, and backpropagate through
the recurrence rather than treating stored voltages as independent samples.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from .connectome_actor import (
    RecurrentConnectomeActorParams,
    connectome_action_mean_and_next_voltage,
    initial_voltage,
)
from .policy import (
    ACTION_HIGH,
    ACTION_LOW,
    ActorParams,
    action_distribution,
    action_std,
    deterministic_action,
    gaussian_log_prob,
    sample_action,
)

Array = jax.Array


class ActorInterface(NamedTuple):
    """Operations ``training.py`` needs from any actor architecture."""

    step: Callable[[object, Array, Array, Array], tuple[Array, Array, Array, Array]]
    """``(actor_params, carry, obs, key) -> (raw_action, action, log_prob, new_carry)``."""

    distribution: Callable[[object, Array, Array], tuple[Array, Array]]
    """``(actor_params, carry, obs) -> (mean, std)``, differentiable in ``actor_params``."""

    distribution_step: Callable[
        [object, Array, Array], tuple[Array, Array, Array]
    ]
    """``(actor_params, carry, obs) -> (mean, std, new_carry)`` for BPTT."""

    init_carry: Callable[[object], Array]
    """``(actor_params) -> carry``, the value each episode (re)starts from."""

    deterministic_step: Callable[[object, Array, Array], tuple[Array, Array]]
    """``(actor_params, carry, obs) -> (action, new_carry)``, no exploration noise.

    Used for evaluation, deployment, and rendering -- not by the RL update
    loop itself.
    """


def _analytic_step(
    actor_params: ActorParams, carry: Array, obs: Array, key: Array
) -> tuple[Array, Array, Array, Array]:
    raw_action, action, log_prob = sample_action(actor_params, obs, key)
    return raw_action, action, log_prob, carry


def _analytic_distribution(
    actor_params: ActorParams, carry: Array, obs: Array
) -> tuple[Array, Array]:
    del carry
    return action_distribution(actor_params, obs)


def _analytic_distribution_step(
    actor_params: ActorParams, carry: Array, obs: Array
) -> tuple[Array, Array, Array]:
    mean, std = action_distribution(actor_params, obs)
    return mean, std, carry


def _analytic_init_carry(actor_params: ActorParams) -> Array:
    del actor_params
    return jnp.zeros((0,))


def _analytic_deterministic_step(
    actor_params: ActorParams, carry: Array, obs: Array
) -> tuple[Array, Array]:
    return deterministic_action(actor_params.raw_sensory_policy, obs), carry


ANALYTIC_ACTOR_INTERFACE = ActorInterface(
    step=_analytic_step,
    distribution=_analytic_distribution,
    distribution_step=_analytic_distribution_step,
    init_carry=_analytic_init_carry,
    deterministic_step=_analytic_deterministic_step,
)


def make_connectome_actor_interface(
    *, dt: float = 0.02, substeps: int = 4
) -> ActorInterface:
    """Build the connectome actor's interface at a fixed integration ``dt``.

    ``dt`` must match the training/evaluation environment's own ``dt`` (it is
    closed over here rather than threaded through every call, to keep the
    analytic interface's signature equally simple) -- callers are responsible
    for keeping the two in sync.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if substeps <= 0:
        raise ValueError("substeps must be positive")

    def step(
        actor_params: RecurrentConnectomeActorParams,
        carry: Array,
        obs: Array,
        key: Array,
    ) -> tuple[Array, Array, Array, Array]:
        mean, new_carry = connectome_action_mean_and_next_voltage(
            actor_params, carry, obs, dt=dt, substeps=substeps
        )
        std = action_std(actor_params.log_std)
        noise = jax.random.normal(key, shape=mean.shape)
        raw_action = mean + std * noise
        action = jnp.clip(raw_action, ACTION_LOW, ACTION_HIGH)
        log_prob = jnp.sum(gaussian_log_prob(raw_action, mean, std))
        return raw_action, action, log_prob, new_carry

    def distribution(
        actor_params: RecurrentConnectomeActorParams, carry: Array, obs: Array
    ) -> tuple[Array, Array]:
        mean, _ = connectome_action_mean_and_next_voltage(
            actor_params, carry, obs, dt=dt, substeps=substeps
        )
        std = action_std(actor_params.log_std)
        return mean, std

    def distribution_step(
        actor_params: RecurrentConnectomeActorParams, carry: Array, obs: Array
    ) -> tuple[Array, Array, Array]:
        mean, new_carry = connectome_action_mean_and_next_voltage(
            actor_params, carry, obs, dt=dt, substeps=substeps
        )
        std = action_std(actor_params.log_std)
        return mean, std, new_carry

    def deterministic_step(
        actor_params: RecurrentConnectomeActorParams, carry: Array, obs: Array
    ) -> tuple[Array, Array]:
        mean, new_carry = connectome_action_mean_and_next_voltage(
            actor_params, carry, obs, dt=dt, substeps=substeps
        )
        return jnp.clip(mean, ACTION_LOW, ACTION_HIGH), new_carry

    return ActorInterface(
        step=step,
        distribution=distribution,
        distribution_step=distribution_step,
        init_carry=initial_voltage,
        deterministic_step=deterministic_step,
    )


@partial(jax.jit, static_argnames=("env", "actor_interface", "steps"))
def deterministic_rollout(
    actor_params: Any,
    env: Any,
    env_params: Any,
    source_position: Array,
    heading: Array,
    *,
    actor_interface: ActorInterface,
    steps: int,
) -> tuple[Any, Array, Array]:
    """Deterministically roll any actor through the Gymnax env from a fixed start.

    Uses ``env.reset_at`` (not the randomized ``env.reset``) to fix the
    source/heading, and ``env.step_env`` directly (not the auto-resetting
    wrapped ``env.step``) so the same episode runs for exactly ``steps``
    regardless of early success -- matching
    ``r_elegans.envs.petri_dish.simulate_petri_dish``'s fixed-length
    semantics, so results/renderings are directly comparable across actor
    architectures. Returns ``(body_trajectory, distance_to_source)``, where
    ``body_trajectory`` is a stacked ``MuscleBodyState`` (its last segment
    center, via ``world_segment_centers``, is the head path -- see
    ``r_elegans.envs.petri_dish.head_position``) and ``carry_trajectory``,
    the actor's recurrent carry *after* each step (for the connectome actor,
    the subcircuit's per-neuron voltage, shape ``[steps, 14]`` -- see
    ``r_elegans.brain.circuit.SUBCIRCUIT_NEURON_NAMES`` for the neuron order;
    a zero-width ``[steps, 0]`` array for the memoryless analytic actor).

    Jitted with ``env``/``actor_interface``/``steps`` static, mirroring
    ``r_elegans.envs.petri_dish.simulate_petri_dish``'s own
    ``static_argnames=("steps",)`` -- without this, each call in a held-out
    evaluation loop over many sources/headings would retrace and recompile
    from scratch, which is dramatically more expensive than the rollout
    itself for anything but a trivial number of trials.
    """

    obs, state = env.reset_at(source_position, heading, env_params)
    carry = actor_interface.init_carry(actor_params)

    def step_fn(carry_tuple, _):
        obs, state, carry = carry_tuple
        action, new_carry = actor_interface.deterministic_step(actor_params, carry, obs)
        next_obs, next_state, _, _, info = env.step_env(
            jax.random.PRNGKey(0), state, action, env_params
        )
        return (next_obs, next_state, new_carry), (
            next_state.body,
            info["distance_to_source"],
            new_carry,
        )

    _, (body_trajectory, distances, carry_trajectory) = jax.lax.scan(
        step_fn, (obs, state, carry), xs=None, length=steps
    )
    return body_trajectory, distances, carry_trajectory


__all__ = [
    "ActorInterface",
    "ANALYTIC_ACTOR_INTERFACE",
    "deterministic_rollout",
    "make_connectome_actor_interface",
]
