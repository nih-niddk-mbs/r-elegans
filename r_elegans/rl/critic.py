"""A small state-value baseline used only to reduce policy-gradient variance.

The critic is not part of the deployed sensory controller -- it exists solely
to compute an advantage estimate during training and is discarded afterwards.
Its architecture is therefore unconstrained; a compact two-hidden-layer MLP
over the five-dimensional observation is enough for this task.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class CriticParams(NamedTuple):
    """Weights and biases of a two-hidden-layer tanh MLP."""

    w1: Array
    b1: Array
    w2: Array
    b2: Array
    w3: Array
    b3: Array


def init_critic_params(
    key: Array, *, observation_dim: int = 5, hidden: int = 64
) -> CriticParams:
    """Initialize with small random weights and zero biases."""

    key1, key2, key3 = jax.random.split(key, 3)

    def layer(layer_key: Array, fan_in: int, fan_out: int) -> Array:
        return jax.random.normal(layer_key, (fan_in, fan_out)) / jnp.sqrt(fan_in)

    return CriticParams(
        w1=layer(key1, observation_dim, hidden),
        b1=jnp.zeros((hidden,)),
        w2=layer(key2, hidden, hidden),
        b2=jnp.zeros((hidden,)),
        w3=layer(key3, hidden, 1),
        b3=jnp.zeros((1,)),
    )


def critic_value(critic: CriticParams, observation: Array) -> Array:
    """Return a scalar state-value estimate for ``observation``."""

    hidden = jnp.tanh(observation @ critic.w1 + critic.b1)
    hidden = jnp.tanh(hidden @ critic.w2 + critic.b2)
    return (hidden @ critic.w3 + critic.b3)[..., 0]


__all__ = ["CriticParams", "critic_value", "init_critic_params"]
