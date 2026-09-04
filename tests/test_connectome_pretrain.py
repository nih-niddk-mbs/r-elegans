import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("gymnax")
pytest.importorskip("optax")

from r_elegans.body import decode_commanded_controller, default_muscle_body_params
from r_elegans.data.connectome import load_connectome
from r_elegans.envs.gymnax_petri_dish import PetriDishGymnaxEnv
from r_elegans.rl import (
    collect_teacher_trajectories,
    init_connectome_actor_params,
    pretrain_fit,
    pretrain_loss,
    unroll_student,
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
TEACHER_RAW_POLICY = jnp.asarray([0.0, 0.05, -0.05, 0.05, -0.05, 0.0, -2.0])


def make_env() -> PetriDishGymnaxEnv:
    body_params = default_muscle_body_params(12)
    gait_params = decode_commanded_controller(RAW_GAIT)
    return PetriDishGymnaxEnv(body_params, gait_params)


def test_collect_teacher_trajectories_has_expected_shapes() -> None:
    env = make_env()
    params = env.default_params.replace(max_steps_in_episode=10)
    obs, action = collect_teacher_trajectories(
        env, params, TEACHER_RAW_POLICY, jax.random.PRNGKey(0), num_episodes=3, steps=10
    )
    assert obs.shape == (3, 10, 5)
    assert action.shape == (3, 10, 2)
    assert jnp.all(jnp.isfinite(obs))
    assert jnp.all(action[..., 0] >= 0.0) and jnp.all(action[..., 0] <= 1.0)
    assert jnp.all(action[..., 1] >= -1.0) and jnp.all(action[..., 1] <= 1.0)


def test_unroll_student_matches_scan_length() -> None:
    connectome = load_connectome()
    actor = init_connectome_actor_params(connectome, jax.random.PRNGKey(1))
    obs_sequence = jnp.zeros((10, 5))

    predicted = unroll_student(actor, obs_sequence)
    assert predicted.shape == (10, 2)
    assert jnp.all(jnp.isfinite(predicted))


def test_pretrain_loss_is_finite_and_differentiable() -> None:
    connectome = load_connectome()
    actor = init_connectome_actor_params(connectome, jax.random.PRNGKey(2))
    obs_batch = jnp.zeros((2, 5, 5))
    teacher_action_batch = jnp.zeros((2, 5, 2))

    (loss, aux), gradient = jax.value_and_grad(
        lambda a: pretrain_loss(a, obs_batch, teacher_action_batch), has_aux=True
    )(actor)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(aux["mse"])
    assert jnp.all(gradient.neural_params.chemical_mask == 0.0)


def test_pretraining_reduces_loss_on_a_synthetic_teacher() -> None:
    env = make_env()
    params = env.default_params.replace(max_steps_in_episode=16)
    obs_batch, action_batch = collect_teacher_trajectories(
        env, params, TEACHER_RAW_POLICY, jax.random.PRNGKey(3), num_episodes=4, steps=16
    )

    connectome = load_connectome()
    actor = init_connectome_actor_params(connectome, jax.random.PRNGKey(4))

    _, losses = pretrain_fit(
        actor, obs_batch, action_batch, iterations=20, learning_rate=1e-2, log_every=1000
    )

    assert len(losses) == 20
    assert all(loss == loss and loss < float("inf") for loss in losses)  # finite, no NaN
    assert losses[-1] < losses[0]
