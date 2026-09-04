import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("gymnax")
pytest.importorskip("optax")

from r_elegans.body import decode_commanded_controller, default_muscle_body_params
from r_elegans.data.connectome import load_connectome
from r_elegans.envs.gymnax_petri_dish import PetriDishGymnaxEnv
from r_elegans.rl import (
    ACTION_HIGH,
    ACTION_LOW,
    ANALYTIC_ACTOR_INTERFACE,
    TrainingConfig,
    connectome_action_mean_and_next_voltage,
    deterministic_rollout,
    init_actor_params,
    init_connectome_actor_params,
    initial_voltage,
    make_connectome_actor_interface,
    train,
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


def test_init_connectome_actor_params_slices_real_connectome() -> None:
    connectome = load_connectome()
    actor = init_connectome_actor_params(connectome, jax.random.PRNGKey(0))

    assert actor.neural_params.chemical_mask.shape == (14, 14)
    assert jnp.sum(actor.neural_params.chemical_mask) > 0
    assert actor.sensory_gains.weights.shape == (4, 4)
    assert actor.log_std.shape == (2,)


def test_initial_voltage_matches_leak_reversal() -> None:
    connectome = load_connectome()
    actor = init_connectome_actor_params(connectome, jax.random.PRNGKey(0))
    voltage = initial_voltage(actor)

    assert voltage.shape == (14,)
    np.testing.assert_allclose(
        np.asarray(voltage), np.full((14,), -0.35), atol=1e-6
    )


def test_forward_step_returns_bounded_action_and_finite_next_voltage() -> None:
    connectome = load_connectome()
    actor = init_connectome_actor_params(connectome, jax.random.PRNGKey(1))
    voltage = initial_voltage(actor)
    observation = jnp.array([0.2, -0.1, 0.5, 0.86, 0.4])

    mean, next_voltage = connectome_action_mean_and_next_voltage(
        actor, voltage, observation
    )

    assert mean.shape == (2,)
    assert next_voltage.shape == (14,)
    assert jnp.all(jnp.isfinite(next_voltage))
    assert 0.0 <= float(mean[0]) <= 1.0  # speed
    assert -1.0 <= float(mean[1]) <= 1.0  # steering


def test_forward_step_is_differentiable_and_respects_masks() -> None:
    connectome = load_connectome()
    actor = init_connectome_actor_params(connectome, jax.random.PRNGKey(2))
    voltage = initial_voltage(actor)
    observation = jnp.array([0.2, -0.1, 0.5, 0.86, 0.4])

    def loss(actor_params):
        mean, _ = connectome_action_mean_and_next_voltage(
            actor_params, voltage, observation
        )
        return jnp.sum(mean**2)

    gradient = jax.grad(loss)(actor)
    assert jnp.all(jnp.isfinite(gradient.neural_params.raw_chemical))
    assert jnp.all(gradient.neural_params.chemical_mask == 0.0)
    assert jnp.all(gradient.neural_params.gap_mask == 0.0)


def test_connectome_actor_interface_step_and_distribution_agree_on_mean() -> None:
    connectome = load_connectome()
    actor = init_connectome_actor_params(connectome, jax.random.PRNGKey(3))
    voltage = initial_voltage(actor)
    observation = jnp.array([0.1, 0.0, 0.0, 1.0, 0.5])
    interface = make_connectome_actor_interface()

    mean, std = interface.distribution(actor, voltage, observation)
    raw_action, action, log_prob, new_carry = interface.step(
        actor, voltage, observation, jax.random.PRNGKey(4)
    )

    assert mean.shape == (2,) and std.shape == (2,)
    assert jnp.all(action >= ACTION_LOW) and jnp.all(action <= ACTION_HIGH)
    assert jnp.isfinite(log_prob)
    assert new_carry.shape == (14,)


def test_short_connectome_training_run_stays_finite() -> None:
    """Smoke test: PPO through the recurrent connectome actor, end to end.

    Not a convergence test (too short/small to be non-flaky) -- just confirms
    the recurrent carry threads through rollout, GAE, and the loss without
    shape errors or non-finite values.
    """

    env = make_env()
    connectome = load_connectome()
    params = env.default_params.replace(max_steps_in_episode=24)
    config = TrainingConfig(
        num_envs=4,
        num_steps=24,
        num_updates=3,
        update_epochs=1,
        num_minibatches=2,
        learning_rate=1e-3,
    )
    agent, history = train(
        env,
        params,
        config,
        algorithm="ppo",
        actor_interface=make_connectome_actor_interface(dt=params.dt),
        init_actor=lambda: init_connectome_actor_params(connectome, jax.random.PRNGKey(0)),
        seed=0,
        log_every=1000,
    )

    assert len(history) == config.num_updates
    for metrics in history:
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["value_loss"])
    assert jnp.all(jnp.isfinite(agent.actor.neural_params.raw_chemical))


def test_deterministic_rollout_returns_voltage_trajectory_for_connectome_actor() -> None:
    env = make_env()
    params = env.default_params
    connectome = load_connectome()
    actor = init_connectome_actor_params(connectome, jax.random.PRNGKey(5))
    interface = make_connectome_actor_interface(dt=params.dt)

    body_trajectory, distances, voltage_trajectory = deterministic_rollout(
        actor, env, params, jnp.asarray([0.7, 0.0]), jnp.asarray(0.0),
        actor_interface=interface, steps=15,
    )

    assert body_trajectory.position.shape == (15, 2)
    assert distances.shape == (15,)
    assert voltage_trajectory.shape == (15, 14)
    assert jnp.all(jnp.isfinite(voltage_trajectory))
    # The very first step's carry-in is the resting potential; its readout
    # after one integration step should already have moved away from rest.
    assert not jnp.allclose(voltage_trajectory[0], initial_voltage(actor))


def test_deterministic_rollout_returns_zero_width_voltage_for_analytic_actor() -> None:
    env = make_env()
    params = env.default_params
    actor = init_actor_params()

    _, distances, carry_trajectory = deterministic_rollout(
        actor, env, params, jnp.asarray([0.7, 0.0]), jnp.asarray(0.0),
        actor_interface=ANALYTIC_ACTOR_INTERFACE, steps=10,
    )

    assert distances.shape == (10,)
    assert carry_trajectory.shape == (10, 0)
