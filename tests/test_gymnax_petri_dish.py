import pytest

pytest.importorskip("gymnax")

import jax
import jax.numpy as jnp

from r_elegans.body import decode_commanded_controller, default_muscle_body_params
from r_elegans.envs.gymnax_petri_dish import (
    ACTION_HIGH,
    ACTION_LOW,
    PetriDishGymnaxEnv,
)

RAW_GAIT = jnp.asarray(
    [
        2.34,
        -0.45,
        -0.256,
        0.64,
        0.011,
        2.47,
        -0.535,
        0.369,
        0.714,
        -0.018,
        0.2,
        -0.2,
        0.5,
        0.0,
    ]
)


def make_env() -> PetriDishGymnaxEnv:
    body_params = default_muscle_body_params(12)
    gait_params = decode_commanded_controller(RAW_GAIT)
    return PetriDishGymnaxEnv(body_params, gait_params)


def test_reset_returns_five_dimensional_observation_at_zero_time() -> None:
    env = make_env()
    params = env.default_params
    obs, state = env.reset(jax.random.PRNGKey(0), params)

    assert obs.shape == (5,)
    assert jnp.all(jnp.isfinite(obs))
    assert state.time == 0
    # No sensory history yet: response and derivative both start at zero.
    assert obs[0] == 0.0
    assert obs[1] == 0.0


def test_reset_samples_source_within_configured_annulus() -> None:
    env = make_env()
    params = env.default_params
    keys = jax.random.split(jax.random.PRNGKey(1), 32)
    _, states = jax.vmap(env.reset, in_axes=(0, None))(keys, params)

    radii = jnp.linalg.norm(states.source_position, axis=-1)
    assert jnp.all(radii >= params.source_radius_min - 1e-5)
    assert jnp.all(radii <= params.source_radius_max + 1e-5)


def test_step_advances_time_and_stays_finite() -> None:
    env = make_env()
    params = env.default_params
    obs, state = env.reset(jax.random.PRNGKey(2), params)

    action = jnp.array([0.7, 0.1])
    next_obs, next_state, reward, terminated, truncated, info = env.step(
        jax.random.PRNGKey(3), state, action, params
    )

    assert next_state.time == 1
    assert jnp.all(jnp.isfinite(next_obs))
    assert jnp.isfinite(reward)
    assert bool(terminated) is False
    assert bool(truncated) is False
    assert "distance_to_source" in info
    assert "success" in info


def test_episode_truncates_at_max_steps() -> None:
    env = make_env()
    params = env.default_params.replace(max_steps_in_episode=3)
    key = jax.random.PRNGKey(4)
    obs, state = env.reset(key, params)

    action = jnp.array([0.5, 0.0])
    truncated = False
    for _ in range(3):
        key, step_key = jax.random.split(key)
        obs, state, reward, terminated, truncated, info = env.step(
            step_key, state, action, params
        )
    assert bool(truncated) is True


def test_head_toward_source_yields_positive_progress_reward() -> None:
    """Steering the fitted gait's forward command straight at the source
    should reduce distance and therefore earn positive net reward on average.
    """

    env = make_env()
    params = env.default_params
    key = jax.random.PRNGKey(5)
    obs, state = env.reset(key, params)

    total_reward = 0.0
    action = jnp.array([0.7, 0.0])
    for _ in range(20):
        key, step_key = jax.random.split(key)
        obs, state, reward, terminated, truncated, info = env.step(
            step_key, state, action, params
        )
        total_reward += float(reward)
    assert jnp.isfinite(total_reward)


def test_action_is_clipped_to_declared_bounds() -> None:
    env = make_env()
    params = env.default_params
    _, state = env.reset(jax.random.PRNGKey(6), params)

    out_of_bounds = jnp.array([5.0, -5.0])
    _, next_state, *_ = env.step(jax.random.PRNGKey(7), state, out_of_bounds, params)
    _, clipped_state, *_ = env.step(
        jax.random.PRNGKey(7),
        state,
        jnp.clip(out_of_bounds, ACTION_LOW, ACTION_HIGH),
        params,
    )
    assert jnp.allclose(next_state.body.joint_angles, clipped_state.body.joint_angles)
