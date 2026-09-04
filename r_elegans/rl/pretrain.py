"""Supervised pretraining of the connectome subcircuit against a teacher policy.

Unlike the RL rollout (:mod:`r_elegans.rl.training`), this stage is not
subject to any stop-gradient boundary: it backpropagates an ordinary MSE loss
through a full within-episode unroll of the subcircuit's own recurrence,
matching :mod:`r_elegans.body.fitting`'s existing ``jax.lax.scan`` +
``jax.value_and_grad`` style. Intended to give RL fine-tuning
(:func:`r_elegans.rl.training.train` with
:func:`r_elegans.rl.actor_interface.make_connectome_actor_interface`) a
competent starting point instead of a randomly initialized network.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax

from .connectome_actor import (
    RecurrentConnectomeActorParams,
    connectome_action_mean_and_next_voltage,
    initial_voltage,
)
from .policy import deterministic_action

Array = jax.Array


def collect_teacher_trajectories(
    env: Any,
    env_params: Any,
    raw_teacher_policy: Array,
    rng: Array,
    num_episodes: int,
    steps: int,
) -> tuple[Array, Array]:
    """Roll a deterministic analytic teacher through the Gymnax env.

    Uses the same environment as RL fine-tuning
    (:class:`r_elegans.envs.gymnax_petri_dish.PetriDishGymnaxEnv`), not
    :func:`r_elegans.envs.petri_dish.simulate_petri_dish`, so pretraining and
    RL fine-tuning see identical observation semantics. Returns
    ``(obs[num_episodes, steps, 5], action[num_episodes, steps, 2])``.
    """

    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, num_episodes)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)

    def step_fn(carry, _):
        obs, state, rng = carry
        rng, step_rng = jax.random.split(rng)
        action = jax.vmap(deterministic_action, in_axes=(None, 0))(
            raw_teacher_policy, obs
        )
        step_rngs = jax.random.split(step_rng, num_episodes)
        next_obs, next_state, _, _, _, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(step_rngs, state, action, env_params)
        return (next_obs, next_state, rng), (obs, action)

    _, (obs_seq, action_seq) = jax.lax.scan(
        step_fn, (obs, state, rng), xs=None, length=steps
    )
    # [steps, episodes, ...] -> [episodes, steps, ...]
    return jnp.swapaxes(obs_seq, 0, 1), jnp.swapaxes(action_seq, 0, 1)


def unroll_student(
    actor: RecurrentConnectomeActorParams,
    obs_sequence: Array,
    *,
    dt: float = 0.02,
    substeps: int = 4,
) -> Array:
    """Roll the subcircuit forward from rest through one episode's observations."""

    def step(voltage: Array, obs: Array) -> tuple[Array, Array]:
        mean, next_voltage = connectome_action_mean_and_next_voltage(
            actor, voltage, obs, dt=dt, substeps=substeps
        )
        return next_voltage, mean

    _, means = jax.lax.scan(step, initial_voltage(actor), obs_sequence)
    return means


def pretrain_loss(
    actor: RecurrentConnectomeActorParams,
    obs_batch: Array,
    teacher_action_batch: Array,
    *,
    weight_l2: float = 1e-4,
) -> tuple[Array, dict[str, Array]]:
    """MSE between the subcircuit's unrolled ``[speed, steering]`` and the teacher's."""

    predicted = jax.vmap(unroll_student, in_axes=(None, 0))(actor, obs_batch)
    mse = jnp.mean((predicted - teacher_action_batch) ** 2)
    weight_penalty = weight_l2 * jnp.mean(actor.neural_params.raw_chemical**2)
    return mse + weight_penalty, {"mse": mse, "weight_penalty": weight_penalty}


def fit(
    actor: RecurrentConnectomeActorParams,
    obs_batch: Array,
    teacher_action_batch: Array,
    *,
    iterations: int,
    learning_rate: float,
    weight_l2: float = 1e-4,
    max_grad_norm: float = 0.5,
    log_every: int = 25,
) -> tuple[RecurrentConnectomeActorParams, list[float]]:
    """Adam-optimize ``pretrain_loss``, printing progress every ``log_every`` steps."""

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate)
    )
    opt_state = optimizer.init(actor)
    grad_fn = jax.jit(
        jax.value_and_grad(
            lambda a: pretrain_loss(a, obs_batch, teacher_action_batch, weight_l2=weight_l2),
            has_aux=True,
        )
    )

    losses: list[float] = []
    for iteration in range(1, iterations + 1):
        (loss, aux), grads = grad_fn(actor)
        updates, opt_state = optimizer.update(grads, opt_state, actor)
        actor = optax.apply_updates(actor, updates)
        losses.append(float(loss))
        if iteration == 1 or iteration % log_every == 0 or iteration == iterations:
            print(
                f"iteration={iteration:4d} loss={float(loss):.6f} "
                f"mse={float(aux['mse']):.6f}"
            )
    return actor, losses


__all__ = [
    "collect_teacher_trajectories",
    "fit",
    "pretrain_loss",
    "unroll_student",
]
