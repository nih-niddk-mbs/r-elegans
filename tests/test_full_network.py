"""Full topology, temporal gradients, direct muscle control and subset training."""
import importlib.util

import pytest
if importlib.util.find_spec("gymnax") is None:
    pytest.skip("gymnax extra not installed", allow_module_level=True)

import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.model import load_builtin_model
from r_elegans.body import default_muscle_body_params
from r_elegans.brain.circuit import decode_subcircuit_params
from r_elegans.brain.dynamics import integrate_neural_fixed_step
from r_elegans.envs.full_network_petri_dish import FullNetworkPetriDishEnv
from r_elegans.rl.full_network_actor import (
    init_full_network_actor, make_full_network_actor_interface,
    integrate_frozen_conductance, actor_to_arrays, actor_from_arrays,
)
from r_elegans.rl.training import AgentParams, TrainingConfig, make_ppo_train_step
from r_elegans.rl.critic import init_critic_params


@pytest.fixture
def setup():
    model = load_builtin_model()
    actor = init_full_network_actor(model, jax.random.PRNGKey(2))
    env = FullNetworkPetriDishEnv(default_muscle_body_params(12))
    return model, actor, env


def test_full_topology_and_roundtrip(setup):
    model, actor, env = setup
    assert actor.neural.raw_chemical.shape == (302, 302)
    np.testing.assert_array_equal(actor.neural.chemical_mask, model.connectome.chemical_counts > 0)
    np.testing.assert_array_equal(actor.neural.gap_mask, model.connectome.gap_counts > 0)
    restored = actor_from_arrays(actor_to_arrays(actor))
    for a, b in zip(jax.tree_util.tree_leaves(actor), jax.tree_util.tree_leaves(restored)):
        np.testing.assert_array_equal(a, b)
    arrays = actor_to_arrays(actor)
    arrays["sensory_weights"] = np.zeros((4, 5))
    with pytest.raises(ValueError, match="shapes"):
        actor_from_arrays(arrays)


def test_integrator_convergence_and_high_conductance_stability(setup):
    model, actor, env = setup
    params = decode_subcircuit_params(actor.neural)
    v = params.leak_reversal
    external = jnp.zeros(302).at[0].set(0.3)
    reference = integrate_neural_fixed_step(v, params, external, 0.02, substeps=128)
    coarse = integrate_frozen_conductance(v, params, external, 0.02, 4)
    fine = integrate_frozen_conductance(v, params, external, 0.02, 64)
    assert jnp.linalg.norm(fine - reference) < jnp.linalg.norm(coarse - reference)
    np.testing.assert_allclose(fine, reference, atol=1e-4, rtol=1e-3)
    strong = params._replace(raw_chemical=jnp.full((302, 302), 100.0),
                             raw_gap=jnp.full((302, 302), 100.0))
    final = integrate_frozen_conductance(v, strong, external, 10.0, 4)
    assert np.all(np.isfinite(final))
    assert np.max(np.abs(final)) < 1.0


def sequence_loss(actor, interface):
    def step(v, obs):
        mean, _, next_v = interface.distribution_step(actor, v, obs)
        return next_v, mean
    _, means = jax.lax.scan(step, interface.init_carry(actor),
                            jnp.tile(jnp.array([0.3, -0.2, 0.4]), (8, 1)))
    # Nonuniform targets avoid cancellation between dorsal/ventral muscles.
    return jnp.mean((means - jnp.linspace(-1, 1, 95)) ** 2)


def test_temporal_gradients_and_anatomical_masks(setup):
    model, actor, env = setup
    interface = make_full_network_actor_interface(model)
    grad = jax.jit(jax.grad(lambda a: sequence_loss(a, interface)))(actor)
    for leaf in jax.tree_util.tree_leaves(grad):
        assert np.all(np.isfinite(leaf))
    for field in (grad.neural.raw_chemical, grad.neural.raw_gap, grad.sensory_weights,
                  grad.raw_nmj, grad.unknown_polarity):
        assert np.linalg.norm(field) > 0
    np.testing.assert_array_equal(grad.neural.raw_chemical * (1 - actor.neural.chemical_mask), 0)
    np.testing.assert_array_equal(grad.raw_nmj * (model.neuromuscular_params.synapse_weights == 0), 0)
    np.testing.assert_array_equal(grad.neural.chemical_mask, 0)
    np.testing.assert_array_equal(grad.neural.gap_mask, 0)


def test_subset_freezes_parameters_without_changing_forward_model(setup):
    model, actor, env = setup
    all_interface = make_full_network_actor_interface(model)
    restricted = make_full_network_actor_interface(model, train_group="neural", train_neurons=["AWCL"])
    v = all_interface.init_carry(actor)
    obs = jnp.array([0.3, -0.2, 0.4])
    for a, b in zip(all_interface.distribution_step(actor, v, obs), restricted.distribution_step(actor, v, obs)):
        np.testing.assert_array_equal(a, b)
    grad = jax.jit(jax.grad(lambda a: sequence_loss(a, restricted)))(actor)
    np.testing.assert_array_equal(grad.raw_nmj, 0)
    np.testing.assert_array_equal(grad.sensory_weights, 0)
    outside = np.ones(302, dtype=bool)
    outside[model.neuron_ids.index("AWCL")] = False
    np.testing.assert_array_equal(np.asarray(grad.neural.raw_chemical)[outside], 0)
    assert np.linalg.norm(grad.neural.raw_chemical) > 0


def test_recurrent_brain_drives_body_without_gait(setup):
    model, actor, env = setup
    assert env.gait_params is None
    interface = make_full_network_actor_interface(model)
    obs, state = env.reset_at(jnp.array([0.7, 0.0]), jnp.asarray(0.0))
    assert obs.shape == (3,)
    def rollout(a):
        def step(carry, _):
            obs, s, voltage = carry
            muscles, voltage = interface.deterministic_step(a, voltage, obs)
            obs, s, _, _, _ = env.step_env(jax.random.PRNGKey(0), s, muscles, env.default_params)
            return (obs, s, voltage), muscles
        return jax.lax.scan(step, (obs, state, interface.init_carry(a)), None, length=40)
    final, muscles = jax.jit(rollout)(actor)
    assert final[2].shape == (302,)
    assert muscles.shape == (40, 95)
    assert np.max(np.abs(final[1].body.joint_angles)) > 1e-5
    assert np.all(np.isfinite(final[1].body.position))
    silence = actor._replace(raw_nmj=jnp.full_like(actor.raw_nmj, -100.0))
    silent_final, _ = jax.jit(rollout)(silence)
    assert np.linalg.norm(final[1].body.joint_angles - silent_final[1].body.joint_angles) > 1e-5


def test_full_network_ppo_update_and_frozen_subset(setup):
    model, actor, env = setup
    interface = make_full_network_actor_interface(model, train_group="nmj")
    config = TrainingConfig(num_envs=2, num_steps=4, num_updates=1,
                            num_minibatches=1, update_epochs=2)
    agent = AgentParams(actor, init_critic_params(jax.random.PRNGKey(3), observation_dim=3))
    train, optimizer = make_ppo_train_step(env, config, actor_interface=interface)
    trained, opt, rng, metrics = jax.block_until_ready(train(agent, optimizer.init(agent), env.default_params,
                                                          jax.random.PRNGKey(4)))
    assert all(np.all(np.isfinite(v)) for v in jax.tree_util.tree_leaves((trained, opt, metrics)))
    for a, b in zip(actor.neural, trained.actor.neural):
        np.testing.assert_array_equal(a, b)
    assert np.linalg.norm(trained.actor.raw_nmj - actor.raw_nmj) > 0
