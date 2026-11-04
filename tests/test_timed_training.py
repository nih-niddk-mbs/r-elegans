import importlib.util

import pytest

if importlib.util.find_spec("gymnax") is None:
    pytest.skip("gymnax extra not installed", allow_module_level=True)
import gymnax

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from r_elegans.rl import AgentParams, init_actor_params, init_critic_params
from scripts.train_timed import checkpoint, validation_rank


def test_training_checkpoint_roundtrips_optimizer_and_rng(tmp_path):
    agent = AgentParams(init_actor_params(), init_critic_params(jax.random.PRNGKey(0), hidden=4))
    opt_state = optax.adam(1e-3).init(agent)
    rng = jax.random.PRNGKey(8)
    template = dict(agent=agent, opt_state=opt_state, rng=rng,
                    updates=np.asarray(7, dtype=np.int64))
    path = tmp_path / "latest.msgpack"

    checkpoint(path, agent, opt_state, rng, updates=7)
    restored = serialization.from_bytes(template, path.read_bytes())
    assert int(restored["updates"]) == 7
    for expected, actual in zip(jax.tree_util.tree_leaves(template),
                                jax.tree_util.tree_leaves(restored)):
        np.testing.assert_array_equal(actual, expected)


def test_nonfinite_state_cannot_overwrite_a_good_checkpoint(tmp_path):
    path = tmp_path / "latest.msgpack"
    rng = jax.random.PRNGKey(0)
    checkpoint(path, {"weight": jnp.asarray(1.0)}, {}, rng, updates=2)
    saved = path.read_bytes()

    with pytest.raises(RuntimeError, match="nonfinite"):
        checkpoint(path, {"weight": jnp.asarray(float("nan"))}, {}, rng, updates=3)
    assert path.read_bytes() == saved


def test_validation_selection_prioritizes_success_then_distance():
    better_success = dict(success_rate=0.5, mean_minimum_distance=0.3)
    better_distance = dict(success_rate=0.25, mean_minimum_distance=0.1)
    tie_success = dict(success_rate=0.5, mean_minimum_distance=0.2)

    assert validation_rank(better_success) > validation_rank(better_distance)
    assert validation_rank(tie_success) > validation_rank(better_success)
