"""Model-free reinforcement learning against the Gymnax Petri dish.

The environment (:mod:`r_elegans.envs.gymnax_petri_dish`) is treated as a
black box that returns rewards, and the seven-parameter sensory controller in
:mod:`r_elegans.rl.policy` is optimized by a policy-gradient update rather
than by differentiating through the muscle/body physics. A small critic
supplies a state-value baseline (via generalized advantage estimation) purely
to reduce gradient variance; it is not part of the deployed controller.

Two update rules share the same rollout collection and GAE:

- **PPO** (``algorithm="ppo"``, the default): the clipped surrogate objective
  of Schulman et al. 2017. Each update collects one on-policy rollout, then
  runs ``update_epochs`` passes over ``num_minibatches`` random shuffles of
  that batch, clipping the probability ratio between the updated and
  rollout-time policy so a single batch can be reused for several gradient
  steps without the policy drifting too far off-policy.
- **A2C** (``algorithm="a2c"``): plain policy gradient with the same GAE
  baseline but no importance-ratio clipping, taking ``update_epochs``
  full-batch gradient steps directly. Simpler, and adequate for this small,
  smooth, low-dimensional controller, but more sensitive to the learning rate
  and epoch count than PPO once epochs exceed one.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from .critic import CriticParams, critic_value, init_critic_params
from .policy import (
    ActorParams,
    action_distribution,
    gaussian_log_prob,
    init_actor_params,
    sample_action,
)

Array = jax.Array


class AgentParams(NamedTuple):
    """Trainable state: the deployed controller plus the training-only critic."""

    actor: ActorParams
    critic: CriticParams


class TrainingConfig(NamedTuple):
    """Rollout size, optimization schedule, and RL hyperparameters."""

    num_envs: int = 32
    num_steps: int = 128
    num_updates: int = 200
    update_epochs: int = 4
    num_minibatches: int = 4
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    learning_rate: float = 3e-3
    max_grad_norm: float = 0.5


class Transition(NamedTuple):
    """One vectorized environment step recorded for advantage estimation."""

    obs: Array
    raw_action: Array
    log_prob: Array
    reward: Array
    done: Array
    value: Array
    distance_to_source: Array
    success: Array


def _rollout(
    agent: AgentParams,
    env: Any,
    env_params: Any,
    rng: Array,
    num_envs: int,
    num_steps: int,
) -> tuple[Transition, Array, Array]:
    """Collect one on-policy batch of parallel episodes."""

    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, num_envs)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)

    def step_fn(carry, _):
        obs, state, rng = carry
        rng, action_rng, step_rng = jax.random.split(rng, 3)
        action_rngs = jax.random.split(action_rng, num_envs)
        raw_action, action, log_prob = jax.vmap(sample_action, in_axes=(None, 0, 0))(
            agent.actor, obs, action_rngs
        )
        value = jax.vmap(critic_value, in_axes=(None, 0))(agent.critic, obs)
        step_rngs = jax.random.split(step_rng, num_envs)
        next_obs, next_state, reward, terminated, truncated, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(step_rngs, state, action, env_params)
        done = jnp.logical_or(terminated, truncated)
        transition = Transition(
            obs=obs,
            raw_action=raw_action,
            log_prob=log_prob,
            reward=reward,
            done=done,
            value=value,
            distance_to_source=info["distance_to_source"],
            success=info["success"],
        )
        return (next_obs, next_state, rng), transition

    (final_obs, _, rng), trajectory = jax.lax.scan(
        step_fn, (obs, state, rng), xs=None, length=num_steps
    )
    last_value = jax.vmap(critic_value, in_axes=(None, 0))(agent.critic, final_obs)
    return trajectory, last_value, rng


def compute_gae(
    trajectory: Transition, last_value: Array, gamma: float, gae_lambda: float
) -> tuple[Array, Array]:
    """Return advantages and bootstrapped returns for the collected batch."""

    def _step(carry, transition):
        gae, next_value = carry
        mask = 1.0 - transition.done.astype(jnp.float32)
        delta = transition.reward + gamma * mask * next_value - transition.value
        gae = delta + gamma * gae_lambda * mask * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _step,
        (jnp.zeros_like(last_value), last_value),
        trajectory,
        reverse=True,
    )
    returns = advantages + trajectory.value
    return advantages, returns


def _normalize(advantage: Array) -> Array:
    return jax.lax.stop_gradient(
        (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)
    )


def _a2c_loss(
    agent: AgentParams,
    obs: Array,
    raw_action: Array,
    advantage: Array,
    returns: Array,
    entropy_coef: float,
    value_coef: float,
) -> tuple[Array, dict[str, Array]]:
    mean, std = jax.vmap(action_distribution, in_axes=(None, 0))(agent.actor, obs)
    log_prob = jnp.sum(gaussian_log_prob(raw_action, mean, std), axis=-1)
    policy_loss = -jnp.mean(log_prob * _normalize(advantage))

    value_pred = jax.vmap(critic_value, in_axes=(None, 0))(agent.critic, obs)
    value_loss = jnp.mean((value_pred - returns) ** 2)

    entropy = jnp.mean(jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * jnp.e * std**2), axis=-1))
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return loss, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
    }


def _ppo_loss(
    agent: AgentParams,
    obs: Array,
    raw_action: Array,
    old_log_prob: Array,
    advantage: Array,
    returns: Array,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
) -> tuple[Array, dict[str, Array]]:
    mean, std = jax.vmap(action_distribution, in_axes=(None, 0))(agent.actor, obs)
    new_log_prob = jnp.sum(gaussian_log_prob(raw_action, mean, std), axis=-1)
    log_ratio = new_log_prob - old_log_prob
    ratio = jnp.exp(log_ratio)

    normalized_advantage = _normalize(advantage)
    surrogate = jnp.minimum(
        ratio * normalized_advantage,
        jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * normalized_advantage,
    )
    policy_loss = -jnp.mean(surrogate)

    value_pred = jax.vmap(critic_value, in_axes=(None, 0))(agent.critic, obs)
    value_loss = jnp.mean((value_pred - returns) ** 2)

    entropy = jnp.mean(jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * jnp.e * std**2), axis=-1))
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return loss, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        # A cheap unbiased KL(old || new) estimate (Schulman's k1); useful to
        # confirm the clip is doing its job (should stay small, e.g. < 0.02).
        "approx_kl": jnp.mean(-log_ratio),
    }


def make_a2c_train_step(env: Any, config: TrainingConfig):
    """Build a jitted function running one rollout-then-update A2C iteration."""

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )
    grad_fn = jax.value_and_grad(
        lambda agent, batch: _a2c_loss(
            agent,
            batch["obs"],
            batch["raw_action"],
            batch["advantage"],
            batch["returns"],
            config.entropy_coef,
            config.value_coef,
        ),
        has_aux=True,
    )

    def train_step(
        agent: AgentParams, opt_state: optax.OptState, env_params: Any, rng: Array
    ):
        rng, rollout_rng = jax.random.split(rng)
        trajectory, last_value, _ = _rollout(
            agent, env, env_params, rollout_rng, config.num_envs, config.num_steps
        )
        advantages, returns = compute_gae(
            trajectory, last_value, config.gamma, config.gae_lambda
        )
        batch = {
            "obs": trajectory.obs.reshape((-1, trajectory.obs.shape[-1])),
            "raw_action": trajectory.raw_action.reshape(
                (-1, trajectory.raw_action.shape[-1])
            ),
            "advantage": advantages.reshape((-1,)),
            "returns": returns.reshape((-1,)),
        }

        def epoch(carry, _):
            agent, opt_state = carry
            (_, metrics), grads = grad_fn(agent, batch)
            updates, opt_state = optimizer.update(grads, opt_state, agent)
            agent = optax.apply_updates(agent, updates)
            return (agent, opt_state), metrics

        (agent, opt_state), epoch_metrics = jax.lax.scan(
            epoch, (agent, opt_state), xs=None, length=config.update_epochs
        )

        metrics = {
            "policy_loss": epoch_metrics["policy_loss"][-1],
            "value_loss": epoch_metrics["value_loss"][-1],
            "entropy": epoch_metrics["entropy"][-1],
            "mean_reward": jnp.mean(trajectory.reward),
            "mean_distance": jnp.mean(trajectory.distance_to_source),
            "success_rate": jnp.mean(trajectory.success.astype(jnp.float32)),
        }
        return agent, opt_state, rng, metrics

    return jax.jit(train_step), optimizer


def make_ppo_train_step(env: Any, config: TrainingConfig):
    """Build a jitted function running one rollout-then-update PPO iteration."""

    batch_size = config.num_envs * config.num_steps
    if batch_size % config.num_minibatches != 0:
        raise ValueError(
            "num_envs * num_steps "
            f"({batch_size}) must be divisible by num_minibatches "
            f"({config.num_minibatches})"
        )
    minibatch_size = batch_size // config.num_minibatches

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )
    grad_fn = jax.value_and_grad(
        lambda agent, batch: _ppo_loss(
            agent,
            batch["obs"],
            batch["raw_action"],
            batch["log_prob"],
            batch["advantage"],
            batch["returns"],
            config.clip_eps,
            config.entropy_coef,
            config.value_coef,
        ),
        has_aux=True,
    )

    def train_step(
        agent: AgentParams, opt_state: optax.OptState, env_params: Any, rng: Array
    ):
        rng, rollout_rng, shuffle_rng = jax.random.split(rng, 3)
        trajectory, last_value, _ = _rollout(
            agent, env, env_params, rollout_rng, config.num_envs, config.num_steps
        )
        advantages, returns = compute_gae(
            trajectory, last_value, config.gamma, config.gae_lambda
        )
        batch = {
            "obs": trajectory.obs.reshape((batch_size, trajectory.obs.shape[-1])),
            "raw_action": trajectory.raw_action.reshape(
                (batch_size, trajectory.raw_action.shape[-1])
            ),
            "log_prob": trajectory.log_prob.reshape((batch_size,)),
            "advantage": advantages.reshape((batch_size,)),
            "returns": returns.reshape((batch_size,)),
        }

        def epoch(carry, epoch_rng):
            agent, opt_state = carry
            permutation = jax.random.permutation(epoch_rng, batch_size)
            shuffled = jax.tree_util.tree_map(lambda x: x[permutation], batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((config.num_minibatches, minibatch_size) + x.shape[1:]),
                shuffled,
            )

            def minibatch_step(carry, minibatch):
                agent, opt_state = carry
                (_, metrics), grads = grad_fn(agent, minibatch)
                updates, opt_state = optimizer.update(grads, opt_state, agent)
                agent = optax.apply_updates(agent, updates)
                return (agent, opt_state), metrics

            (agent, opt_state), minibatch_metrics = jax.lax.scan(
                minibatch_step, (agent, opt_state), minibatches
            )
            return (agent, opt_state), minibatch_metrics

        epoch_rngs = jax.random.split(shuffle_rng, config.update_epochs)
        (agent, opt_state), epoch_metrics = jax.lax.scan(
            epoch, (agent, opt_state), epoch_rngs
        )

        metrics = {
            "policy_loss": epoch_metrics["policy_loss"][-1, -1],
            "value_loss": epoch_metrics["value_loss"][-1, -1],
            "entropy": epoch_metrics["entropy"][-1, -1],
            "approx_kl": epoch_metrics["approx_kl"][-1, -1],
            "mean_reward": jnp.mean(trajectory.reward),
            "mean_distance": jnp.mean(trajectory.distance_to_source),
            "success_rate": jnp.mean(trajectory.success.astype(jnp.float32)),
        }
        return agent, opt_state, rng, metrics

    return jax.jit(train_step), optimizer


_ALGORITHMS = {"ppo": make_ppo_train_step, "a2c": make_a2c_train_step}


def train(
    env: Any,
    env_params: Any,
    config: TrainingConfig = TrainingConfig(),
    *,
    algorithm: str = "ppo",
    seed: int = 0,
    log_every: int = 10,
) -> tuple[AgentParams, list[dict[str, float]]]:
    """Run ``config.num_updates`` reinforcement-learning updates.

    ``algorithm`` selects the update rule: ``"ppo"`` (default, clipped
    surrogate objective) or ``"a2c"`` (plain policy gradient with the same
    GAE baseline, no clipping). Returns the final agent
    (``agent.actor.raw_sensory_policy`` is the trained seven-parameter
    controller) and a per-update metrics history.
    """

    if algorithm not in _ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}; expected 'ppo' or 'a2c'")

    rng = jax.random.PRNGKey(seed)
    critic_key, rng = jax.random.split(rng)
    agent = AgentParams(
        actor=init_actor_params(), critic=init_critic_params(critic_key)
    )
    train_step, optimizer = _ALGORITHMS[algorithm](env, config)
    opt_state = optimizer.init(agent)

    history: list[dict[str, float]] = []
    for update in range(1, config.num_updates + 1):
        agent, opt_state, rng, metrics = train_step(agent, opt_state, env_params, rng)
        numeric_metrics = {key: float(value) for key, value in metrics.items()}
        numeric_metrics["update"] = float(update)
        history.append(numeric_metrics)
        if update == 1 or update % log_every == 0 or update == config.num_updates:
            kl = numeric_metrics.get("approx_kl")
            kl_text = f" kl={kl:.4f}" if kl is not None else ""
            print(
                f"update={update:4d} "
                f"reward={numeric_metrics['mean_reward']:7.4f} "
                f"success={numeric_metrics['success_rate']:.1%} "
                f"mean_dist={numeric_metrics['mean_distance']:.4f} "
                f"entropy={numeric_metrics['entropy']:.3f}"
                f"{kl_text}"
            )
    return agent, history


__all__ = [
    "AgentParams",
    "TrainingConfig",
    "Transition",
    "compute_gae",
    "make_a2c_train_step",
    "make_ppo_train_step",
    "train",
]
