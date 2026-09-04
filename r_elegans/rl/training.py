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
  runs ``update_epochs`` passes over ``num_minibatches`` randomly shuffled
  environment sequences, clipping the probability ratio between the updated
  and rollout-time policy. Timesteps within each sequence remain ordered so a
  recurrent actor is trained with backpropagation through time.
- **A2C** (``algorithm="a2c"``): one full-batch policy-gradient update per
  freshly collected rollout, with the same GAE baseline and no ratio clipping.
  It deliberately does not reuse a rollout for multiple epochs.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax

from .actor_interface import ANALYTIC_ACTOR_INTERFACE, ActorInterface
from .critic import CriticParams, critic_value, init_critic_params
from .policy import ActorParams, gaussian_log_prob, init_actor_params

Array = jax.Array


class AgentParams(NamedTuple):
    """Trainable state: the deployed controller plus the training-only critic."""

    actor: Any
    critic: CriticParams


class TrainingConfig(NamedTuple):
    """Rollout size, optimization schedule, and RL hyperparameters.

    ``update_epochs`` and ``num_minibatches`` apply to PPO. A2C always takes
    one full-batch update from each rollout.
    """

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
    bootstrap_value: Array
    terminated: Array
    truncated: Array
    distance_to_source: Array
    success: Array
    carry_in: Array = jnp.zeros((0,))
    """Diagnostic actor carry before this step (see ``ActorInterface``).

    Zero-size and unused for the memoryless analytic controller; the
    subcircuit's incoming voltage for the recurrent connectome actor. Training
    recomputes the carry from observations rather than treating this as fixed.
    """


def _validate_config(config: TrainingConfig, *, ppo: bool = False) -> None:
    positive_integers = {
        "num_envs": config.num_envs,
        "num_steps": config.num_steps,
        "num_updates": config.num_updates,
        "update_epochs": config.update_epochs,
    }
    if ppo:
        positive_integers["num_minibatches"] = config.num_minibatches
    for name, value in positive_integers.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 <= config.gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= config.gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    if config.learning_rate <= 0.0 or config.max_grad_norm <= 0.0:
        raise ValueError("learning_rate and max_grad_norm must be positive")
    if ppo and config.clip_eps <= 0.0:
        raise ValueError("clip_eps must be positive")


def _rollout(
    agent: AgentParams,
    env: Any,
    env_params: Any,
    rng: Array,
    num_envs: int,
    num_steps: int,
    *,
    actor_interface: ActorInterface = ANALYTIC_ACTOR_INTERFACE,
) -> tuple[Transition, Array, Array]:
    """Collect one on-policy batch of parallel episodes."""

    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, num_envs)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)

    initial_carry = actor_interface.init_carry(agent.actor)
    carry = jnp.broadcast_to(initial_carry, (num_envs,) + initial_carry.shape)

    def step_fn(carry_tuple, _):
        obs, state, carry, rng = carry_tuple
        rng, action_rng, step_rng = jax.random.split(rng, 3)
        action_rngs = jax.random.split(action_rng, num_envs)
        raw_action, action, log_prob, new_carry = jax.vmap(
            actor_interface.step, in_axes=(None, 0, 0, 0)
        )(agent.actor, carry, obs, action_rngs)
        value = jax.vmap(critic_value, in_axes=(None, 0))(agent.critic, obs)
        step_rngs = jax.random.split(step_rng, num_envs)
        next_obs, next_state, reward, terminated, truncated, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(step_rngs, state, action, env_params)
        done = jnp.logical_or(terminated, truncated)
        final_value = jax.vmap(critic_value, in_axes=(None, 0))(
            agent.critic, info["final_observation"]
        )
        # Natural termination has no future value. A time-limit truncation is
        # not terminal and must bootstrap from the pre-reset final observation.
        bootstrap_value = jnp.where(terminated, 0.0, final_value)
        # Reset the actor's recurrent carry (if any) at episode boundaries,
        # exactly as the environment resets obs/state -- otherwise a
        # recurrent actor's hidden state would leak across episodes.
        done_broadcast = done.reshape((num_envs,) + (1,) * initial_carry.ndim)
        reset_carry = jnp.where(done_broadcast, initial_carry, new_carry)
        transition = Transition(
            obs=obs,
            raw_action=raw_action,
            log_prob=log_prob,
            carry_in=carry,
            reward=reward,
            done=done,
            value=value,
            bootstrap_value=bootstrap_value,
            terminated=terminated,
            truncated=truncated,
            distance_to_source=info["distance_to_source"],
            success=info["success"],
        )
        return (next_obs, next_state, reset_carry, rng), transition

    (final_obs, _, _, rng), trajectory = jax.lax.scan(
        step_fn, (obs, state, carry, rng), xs=None, length=num_steps
    )
    last_value = jax.vmap(critic_value, in_axes=(None, 0))(agent.critic, final_obs)
    return trajectory, last_value, rng


def compute_gae(
    trajectory: Transition, last_value: Array, gamma: float, gae_lambda: float
) -> tuple[Array, Array]:
    """Return advantages and returns with correct time-limit bootstrapping.

    ``last_value`` is retained for API compatibility; each transition now
    carries the value of its own pre-reset successor, which is required when
    Gymnax auto-resets an environment after truncation.
    """

    del last_value

    def _step(carry, transition):
        gae = carry
        continuation = 1.0 - transition.done.astype(jnp.float32)
        delta = (
            transition.reward
            + gamma * transition.bootstrap_value
            - transition.value
        )
        gae = delta + gamma * gae_lambda * continuation * gae
        return gae, gae

    _, advantages = jax.lax.scan(
        _step,
        jnp.zeros_like(trajectory.value[-1]),
        trajectory,
        reverse=True,
    )
    returns = advantages + trajectory.value
    return advantages, returns


def _normalize(advantage: Array) -> Array:
    return jax.lax.stop_gradient(
        (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)
    )


def _sequence_distributions(
    actor: Any,
    obs: Array,
    episode_start: Array,
    actor_interface: ActorInterface,
) -> tuple[Array, Array]:
    """Recompute a time-major policy batch while preserving recurrent state."""

    num_envs = obs.shape[1]
    initial = actor_interface.init_carry(actor)
    carries = jnp.broadcast_to(initial, (num_envs,) + initial.shape)

    def step(carry: Array, inputs: tuple[Array, Array]):
        observation, starts = inputs
        start_mask = starts.reshape((num_envs,) + (1,) * initial.ndim)
        carry = jnp.where(start_mask, initial, carry)
        mean, std, next_carry = jax.vmap(
            actor_interface.distribution_step, in_axes=(None, 0, 0)
        )(actor, carry, observation)
        return next_carry, (mean, std)

    _, (means, stds) = jax.lax.scan(step, carries, (obs, episode_start))
    return means, stds


def _critic_sequence_values(critic: CriticParams, obs: Array) -> Array:
    flat_obs = obs.reshape((-1, obs.shape[-1]))
    values = jax.vmap(critic_value, in_axes=(None, 0))(critic, flat_obs)
    return values.reshape(obs.shape[:-1])


def _a2c_loss(
    agent: AgentParams,
    obs: Array,
    raw_action: Array,
    episode_start: Array,
    advantage: Array,
    returns: Array,
    entropy_coef: float,
    value_coef: float,
    *,
    actor_interface: ActorInterface,
) -> tuple[Array, dict[str, Array]]:
    mean, std = _sequence_distributions(
        agent.actor, obs, episode_start, actor_interface
    )
    log_prob = jnp.sum(gaussian_log_prob(raw_action, mean, std), axis=-1)
    policy_loss = -jnp.mean(log_prob * _normalize(advantage))

    value_pred = _critic_sequence_values(agent.critic, obs)
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
    episode_start: Array,
    old_log_prob: Array,
    advantage: Array,
    returns: Array,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    *,
    actor_interface: ActorInterface,
) -> tuple[Array, dict[str, Array]]:
    mean, std = _sequence_distributions(
        agent.actor, obs, episode_start, actor_interface
    )
    new_log_prob = jnp.sum(gaussian_log_prob(raw_action, mean, std), axis=-1)
    log_ratio = new_log_prob - old_log_prob
    ratio = jnp.exp(jnp.clip(log_ratio, -20.0, 20.0))

    normalized_advantage = _normalize(advantage)
    surrogate = jnp.minimum(
        ratio * normalized_advantage,
        jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * normalized_advantage,
    )
    policy_loss = -jnp.mean(surrogate)

    value_pred = _critic_sequence_values(agent.critic, obs)
    value_loss = jnp.mean((value_pred - returns) ** 2)

    entropy = jnp.mean(jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * jnp.e * std**2), axis=-1))
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return loss, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        # Schulman's nonnegative k3 approximation is less noisy than -log(r).
        "approx_kl": jnp.mean((ratio - 1.0) - log_ratio),
        "clip_fraction": jnp.mean(
            (jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32)
        ),
    }


def make_a2c_train_step(
    env: Any,
    config: TrainingConfig,
    *,
    actor_interface: ActorInterface = ANALYTIC_ACTOR_INTERFACE,
):
    """Build a jitted function running one rollout-then-update A2C iteration."""

    _validate_config(config)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )
    grad_fn = jax.value_and_grad(
        lambda agent, batch: _a2c_loss(
            agent,
            batch["obs"],
            batch["raw_action"],
            batch["episode_start"],
            batch["advantage"],
            batch["returns"],
            config.entropy_coef,
            config.value_coef,
            actor_interface=actor_interface,
        ),
        has_aux=True,
    )

    def train_step(
        agent: AgentParams, opt_state: optax.OptState, env_params: Any, rng: Array
    ):
        rng, rollout_rng = jax.random.split(rng)
        trajectory, last_value, _ = _rollout(
            agent,
            env,
            env_params,
            rollout_rng,
            config.num_envs,
            config.num_steps,
            actor_interface=actor_interface,
        )
        advantages, returns = compute_gae(
            trajectory, last_value, config.gamma, config.gae_lambda
        )
        episode_start = jnp.concatenate(
            (
                jnp.ones((1, config.num_envs), dtype=bool),
                trajectory.done[:-1],
            ),
            axis=0,
        )
        batch = {
            "obs": trajectory.obs,
            "raw_action": trajectory.raw_action,
            "episode_start": episode_start,
            "advantage": advantages,
            "returns": returns,
        }

        (_, loss_metrics), grads = grad_fn(agent, batch)
        updates, opt_state = optimizer.update(grads, opt_state, agent)
        agent = optax.apply_updates(agent, updates)

        metrics = {
            "policy_loss": loss_metrics["policy_loss"],
            "value_loss": loss_metrics["value_loss"],
            "entropy": loss_metrics["entropy"],
            "mean_reward": jnp.mean(trajectory.reward),
            "mean_distance": jnp.mean(trajectory.distance_to_source),
            "success_rate": jnp.sum(trajectory.success) / jnp.maximum(
                jnp.sum(trajectory.done), 1
            ),
            "completed_episodes": jnp.sum(trajectory.done),
        }
        return agent, opt_state, rng, metrics

    return jax.jit(train_step), optimizer


def make_ppo_train_step(
    env: Any,
    config: TrainingConfig,
    *,
    actor_interface: ActorInterface = ANALYTIC_ACTOR_INTERFACE,
):
    """Build a jitted function running one rollout-then-update PPO iteration."""

    _validate_config(config, ppo=True)
    if config.num_envs % config.num_minibatches != 0:
        raise ValueError(
            "num_envs "
            f"({config.num_envs}) must be divisible by num_minibatches "
            f"({config.num_minibatches})"
        )
    minibatch_envs = config.num_envs // config.num_minibatches

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )
    grad_fn = jax.value_and_grad(
        lambda agent, batch: _ppo_loss(
            agent,
            batch["obs"],
            batch["raw_action"],
            batch["episode_start"],
            batch["log_prob"],
            batch["advantage"],
            batch["returns"],
            config.clip_eps,
            config.entropy_coef,
            config.value_coef,
            actor_interface=actor_interface,
        ),
        has_aux=True,
    )

    def train_step(
        agent: AgentParams, opt_state: optax.OptState, env_params: Any, rng: Array
    ):
        rng, rollout_rng, shuffle_rng = jax.random.split(rng, 3)
        trajectory, last_value, _ = _rollout(
            agent,
            env,
            env_params,
            rollout_rng,
            config.num_envs,
            config.num_steps,
            actor_interface=actor_interface,
        )
        advantages, returns = compute_gae(
            trajectory, last_value, config.gamma, config.gae_lambda
        )
        episode_start = jnp.concatenate(
            (
                jnp.ones((1, config.num_envs), dtype=bool),
                trajectory.done[:-1],
            ),
            axis=0,
        )
        batch = {
            "obs": trajectory.obs,
            "raw_action": trajectory.raw_action,
            "episode_start": episode_start,
            "log_prob": trajectory.log_prob,
            "advantage": advantages,
            "returns": returns,
        }

        def epoch(carry, epoch_rng):
            agent, opt_state = carry
            # Shuffle complete environment trajectories, never individual
            # timesteps: recurrent actors must be unrolled in temporal order.
            permutation = jax.random.permutation(epoch_rng, config.num_envs)
            shuffled = jax.tree_util.tree_map(lambda x: x[:, permutation], batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    x.reshape(
                        (config.num_steps, config.num_minibatches, minibatch_envs)
                        + x.shape[2:]
                    ),
                    0,
                    1,
                ),
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
        del epoch_metrics
        _, final_loss_metrics = _ppo_loss(
            agent,
            batch["obs"],
            batch["raw_action"],
            batch["episode_start"],
            batch["log_prob"],
            batch["advantage"],
            batch["returns"],
            config.clip_eps,
            config.entropy_coef,
            config.value_coef,
            actor_interface=actor_interface,
        )

        metrics = {
            "policy_loss": final_loss_metrics["policy_loss"],
            "value_loss": final_loss_metrics["value_loss"],
            "entropy": final_loss_metrics["entropy"],
            "approx_kl": final_loss_metrics["approx_kl"],
            "clip_fraction": final_loss_metrics["clip_fraction"],
            "mean_reward": jnp.mean(trajectory.reward),
            "mean_distance": jnp.mean(trajectory.distance_to_source),
            "success_rate": jnp.sum(trajectory.success) / jnp.maximum(
                jnp.sum(trajectory.done), 1
            ),
            "completed_episodes": jnp.sum(trajectory.done),
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
    actor_interface: ActorInterface = ANALYTIC_ACTOR_INTERFACE,
    init_actor: Callable[[], Any] = init_actor_params,
    initial_actor_params: Any = None,
    seed: int = 0,
    log_every: int = 10,
) -> tuple[AgentParams, list[dict[str, float]]]:
    """Run ``config.num_updates`` reinforcement-learning updates.

    ``algorithm`` selects the update rule: ``"ppo"`` (default, clipped
    surrogate objective) or ``"a2c"`` (plain policy gradient with the same
    GAE baseline, no clipping). ``actor_interface`` selects the actor
    architecture (default: the memoryless analytic seven-parameter
    controller; pass
    ``r_elegans.rl.actor_interface.make_connectome_actor_interface()`` to
    train the recurrent connectome subcircuit instead). ``initial_actor_params``
    seeds the actor from an existing (e.g. supervised-pretrained) checkpoint
    when given, otherwise ``init_actor()`` constructs a fresh one. Returns the
    final agent and a per-update metrics history.
    """

    if algorithm not in _ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}; expected 'ppo' or 'a2c'")

    rng = jax.random.PRNGKey(seed)
    critic_key, rng = jax.random.split(rng)
    actor_params = initial_actor_params if initial_actor_params is not None else init_actor()
    observation_shape = env.observation_space(env_params).shape
    if len(observation_shape) != 1:
        raise ValueError("The critic currently requires a one-dimensional observation")
    agent = AgentParams(
        actor=actor_params,
        critic=init_critic_params(critic_key, observation_dim=observation_shape[0]),
    )
    train_step, optimizer = _ALGORITHMS[algorithm](
        env, config, actor_interface=actor_interface
    )
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
            clip_fraction = numeric_metrics.get("clip_fraction")
            clip_text = (
                f" clipped={clip_fraction:.1%}"
                if clip_fraction is not None
                else ""
            )
            print(
                f"update={update:4d} "
                f"reward={numeric_metrics['mean_reward']:7.4f} "
                f"success={numeric_metrics['success_rate']:.1%} "
                f"mean_dist={numeric_metrics['mean_distance']:.4f} "
                f"entropy={numeric_metrics['entropy']:.3f}"
                f"{kl_text}"
                f"{clip_text}"
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
