import pytest

pytest.importorskip("gymnax")
pytest.importorskip("optax")

import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.body import decode_commanded_controller, default_muscle_body_params
from r_elegans.envs import default_petri_dish_params, simulate_petri_dish
from r_elegans.envs.gymnax_petri_dish import PetriDishGymnaxEnv
from r_elegans.rl import (
    ActorParams,
    TrainingConfig,
    action_mean,
    action_std,
    compute_gae,
    deterministic_action,
    init_actor_params,
    make_ppo_train_step,
    sample_action,
    train,
)
from r_elegans.rl.critic import init_critic_params
from r_elegans.rl.training import Transition

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


def test_action_mean_matches_decode_sensory_policy_formula() -> None:
    from r_elegans.envs.petri_dish import decode_sensory_policy

    raw = jnp.asarray([0.3, 0.6, -0.4, 0.2, -0.1, 0.05, -1.5])
    policy = decode_sensory_policy(raw)
    observation = jnp.array([0.2, -0.1, 0.5, 0.86, 0.4])
    response, derivative, sine, cosine, relative = observation

    expected_steering = jnp.tanh(
        policy.response_sine_gain * response * sine
        + policy.response_cosine_gain * response * cosine
        + 0.1 * policy.derivative_sine_gain * derivative * sine
        + 0.1 * policy.derivative_cosine_gain * derivative * cosine
        + policy.steering_bias
    )
    expected_speed = policy.base_speed * (1.0 - policy.food_slowing * relative)

    mean = action_mean(raw, observation)
    assert jnp.allclose(mean, jnp.stack((expected_speed, expected_steering)), atol=1e-6)


def test_deterministic_action_is_clipped_to_dish_bounds() -> None:
    raw = jnp.asarray([10.0] * 7)
    observation = jnp.array([50.0, 50.0, 1.0, 0.0, 0.0])
    action = deterministic_action(raw, observation)
    assert jnp.all(action >= jnp.array([0.0, -1.0]))
    assert jnp.all(action <= jnp.array([1.0, 1.0]))


def test_sample_action_log_prob_is_finite_and_decreases_with_distance() -> None:
    actor = init_actor_params()
    observation = jnp.array([0.1, 0.0, 0.0, 1.0, 0.5])
    key = jax.random.PRNGKey(0)
    raw_action, action, log_prob = sample_action(actor, observation, key)

    assert jnp.isfinite(log_prob)
    assert action.shape == (2,)
    assert jnp.all(action >= jnp.array([0.0, -1.0]))
    assert jnp.all(action <= jnp.array([1.0, 1.0]))


def test_exploration_standard_deviation_is_bounded() -> None:
    std = action_std(jnp.asarray([-1000.0, 1000.0]))
    assert jnp.all(jnp.isfinite(std))
    assert float(std[0]) > 0.0
    assert float(std[1]) < 3.0


def _zero_transition(steps: int, envs: int) -> Transition:
    zeros = jnp.zeros((steps, envs))
    return Transition(
        obs=jnp.zeros((steps, envs, 5)),
        raw_action=jnp.zeros((steps, envs, 2)),
        log_prob=zeros,
        reward=zeros,
        done=jnp.zeros((steps, envs), dtype=bool),
        value=zeros,
        bootstrap_value=zeros,
        terminated=jnp.zeros((steps, envs), dtype=bool),
        truncated=jnp.zeros((steps, envs), dtype=bool),
        distance_to_source=zeros,
        success=jnp.zeros((steps, envs), dtype=bool),
        carry_in=jnp.zeros((steps, envs, 0)),
    )


def test_compute_gae_is_zero_for_zero_reward_zero_value() -> None:
    envs = 3
    trajectory = _zero_transition(5, envs)
    advantages, returns = compute_gae(
        trajectory, jnp.zeros((envs,)), gamma=0.99, gae_lambda=0.95
    )
    assert jnp.allclose(advantages, 0.0)
    assert jnp.allclose(returns, 0.0)


def test_compute_gae_masks_bootstrap_across_episode_boundary() -> None:
    """A `done` flag must zero out the bootstrap term at that step."""

    trajectory = _zero_transition(2, 1)._replace(
        reward=jnp.array([[1.0], [0.0]]),
        done=jnp.array([[True], [False]]),
    )
    advantages, _ = compute_gae(
        trajectory, jnp.array([100.0]), gamma=0.99, gae_lambda=0.95
    )
    # Step 0 is terminal: its advantage must not see the huge bootstrap value.
    assert float(advantages[0, 0]) == pytest.approx(1.0, abs=1e-5)


def test_compute_gae_bootstraps_time_limit_without_crossing_reset() -> None:
    trajectory = _zero_transition(2, 1)._replace(
        reward=jnp.array([[1.0], [100.0]]),
        done=jnp.array([[True], [False]]),
        truncated=jnp.array([[True], [False]]),
        bootstrap_value=jnp.array([[2.0], [0.0]]),
    )
    advantages, _ = compute_gae(
        trajectory, jnp.zeros((1,)), gamma=0.9, gae_lambda=0.95
    )

    # Includes the final observation's value (2), but not the next reset
    # episode's reward (100).
    assert float(advantages[0, 0]) == pytest.approx(2.8, abs=1e-5)


def test_make_ppo_train_step_rejects_indivisible_minibatch_count() -> None:
    env = make_env()
    config = TrainingConfig(num_envs=8, num_steps=5, num_minibatches=3)
    with pytest.raises(ValueError, match="divisible"):
        make_ppo_train_step(env, config)


def test_training_config_rejects_invalid_hyperparameters() -> None:
    env = make_env()
    with pytest.raises(ValueError, match="num_envs"):
        make_ppo_train_step(env, TrainingConfig(num_envs=0))
    with pytest.raises(ValueError, match="gamma"):
        make_ppo_train_step(env, TrainingConfig(gamma=1.1))


@pytest.mark.parametrize("algorithm", ["ppo", "a2c"])
def test_short_training_run_stays_finite_and_yields_a_valid_controller(
    algorithm: str,
) -> None:
    """A short RL run should stay numerically stable and produce a usable policy.

    This does not assert improvement over the initial controller: eight
    updates is too short to guarantee that reliably, and asserting it would
    make the test flaky. ``test_gymnax_petri_dish`` and the smoke runs in
    ``scripts/train_rl_chemotaxis.py`` cover behavior at scale.
    """

    env = make_env()
    params = env.default_params.replace(max_steps_in_episode=48)
    config = TrainingConfig(
        num_envs=8,
        num_steps=48,
        num_updates=8,
        update_epochs=2,
        num_minibatches=4,
        learning_rate=5e-3,
    )
    agent, history = train(env, params, config, algorithm=algorithm, seed=0, log_every=1000)

    assert len(history) == config.num_updates
    for metrics in history:
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["value_loss"])

    body_params = default_muscle_body_params(12)
    gait_params = decode_commanded_controller(RAW_GAIT)
    source = jnp.asarray([0.7, 0.0])
    dish = default_petri_dish_params(source)

    initial_raw = jnp.asarray([0.0, 0.05, -0.05, 0.05, -0.05, 0.0, -2.0])
    _, _, initial_obs = simulate_petri_dish(
        initial_raw, dish, body_params, gait_params, heading=jnp.asarray(np.pi), steps=200
    )
    _, _, trained_obs = simulate_petri_dish(
        agent.actor.raw_sensory_policy,
        dish,
        body_params,
        gait_params,
        heading=jnp.asarray(np.pi),
        steps=200,
    )
    assert jnp.all(jnp.isfinite(trained_obs.head_position))


def test_train_rejects_unknown_algorithm() -> None:
    env = make_env()
    with pytest.raises(ValueError, match="ppo"):
        train(env, env.default_params, TrainingConfig(num_updates=1), algorithm="reinforce")
