"""A stochastic actor built around the seven-parameter sensory controller.

The mean action reproduces :func:`r_elegans.envs.petri_dish.decode_sensory_policy`
and its steering/speed formula exactly, but as a function of an externally
supplied observation rather than of environment internals. This keeps the
*deployed* controller identical in form to the differentiable-fit baseline --
seven physically bounded numbers mapping sensed concentration dynamics and
gait phase to ``[speed, steering]`` -- while allowing it to be optimized by a
model-free reinforcement-learning loop against the Gymnax environment in
:mod:`r_elegans.envs.gymnax_petri_dish`.

The per-channel log standard deviation used for exploration is a training-only
addition. It is not part of the seven fitted parameters and is discarded (or
driven to its floor) when the controller is deployed deterministically.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from r_elegans.envs.petri_dish import decode_sensory_policy

Array = jax.Array

ACTION_LOW = jnp.asarray([0.0, -1.0])
ACTION_HIGH = jnp.asarray([1.0, 1.0])


class ActorParams(NamedTuple):
    """Seven sensory-controller weights plus a training-only exploration scale."""

    raw_sensory_policy: Array
    log_std: Array


def init_actor_params(*, log_std: tuple[float, float] = (-1.0, -1.0)) -> ActorParams:
    """Return a near-neutral controller with a modest exploration scale."""

    raw = jnp.asarray([0.0, 0.05, -0.05, 0.05, -0.05, 0.0, -2.0])
    return ActorParams(raw_sensory_policy=raw, log_std=jnp.asarray(log_std))


def action_mean(raw_sensory_policy: Array, observation: Array) -> Array:
    """Reproduce the fitted controller's ``[speed, steering]`` mean action.

    ``observation`` is ``[response, derivative, sin(phase), cos(phase),
    relative_concentration]``, matching
    :func:`r_elegans.envs.gymnax_petri_dish.PetriDishGymnaxEnv.get_obs`.
    """

    policy = decode_sensory_policy(raw_sensory_policy)
    response, derivative, sine, cosine, relative = observation
    steering = jnp.tanh(
        policy.response_sine_gain * response * sine
        + policy.response_cosine_gain * response * cosine
        + 0.1 * policy.derivative_sine_gain * derivative * sine
        + 0.1 * policy.derivative_cosine_gain * derivative * cosine
        + policy.steering_bias
    )
    speed = policy.base_speed * (1.0 - policy.food_slowing * relative)
    return jnp.stack((speed, steering))


def action_distribution(actor: ActorParams, observation: Array) -> tuple[Array, Array]:
    """Return the Gaussian mean and standard deviation for ``observation``."""

    mean = action_mean(actor.raw_sensory_policy, observation)
    std = jnp.exp(actor.log_std)
    return mean, std


def gaussian_log_prob(x: Array, mean: Array, std: Array) -> Array:
    """Per-channel Gaussian log-density (channels are summed by the caller)."""

    variance = std**2
    return -0.5 * ((x - mean) ** 2 / variance + jnp.log(2.0 * jnp.pi * variance))


def sample_action(
    actor: ActorParams, observation: Array, key: Array
) -> tuple[Array, Array, Array]:
    """Sample a pre-clip action, the dish-bounded action, and its log-prob."""

    mean, std = action_distribution(actor, observation)
    noise = jax.random.normal(key, shape=mean.shape)
    raw_action = mean + std * noise
    action = jnp.clip(raw_action, ACTION_LOW, ACTION_HIGH)
    log_prob = jnp.sum(gaussian_log_prob(raw_action, mean, std))
    return raw_action, action, log_prob


def action_log_prob(actor: ActorParams, observation: Array, raw_action: Array) -> Array:
    """Log-probability of a previously sampled ``raw_action`` under ``actor``."""

    mean, std = action_distribution(actor, observation)
    return jnp.sum(gaussian_log_prob(raw_action, mean, std))


def deterministic_action(raw_sensory_policy: Array, observation: Array) -> Array:
    """The dish-bounded action a deployed (noise-free) controller would take."""

    mean = action_mean(raw_sensory_policy, observation)
    return jnp.clip(mean, ACTION_LOW, ACTION_HIGH)


__all__ = [
    "ACTION_HIGH",
    "ACTION_LOW",
    "ActorParams",
    "action_distribution",
    "action_log_prob",
    "action_mean",
    "deterministic_action",
    "gaussian_log_prob",
    "init_actor_params",
    "sample_action",
]
