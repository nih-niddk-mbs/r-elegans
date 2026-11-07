"""Dynamical controls for the reduced motor-cell hypothesis (not animal fits)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from r_elegans.brain.intrinsic_motor import (
    IntrinsicMotorConfig, motor_class_masks,
)
from r_elegans.model import load_builtin_model
from r_elegans.rl.full_network_actor import (
    init_full_network_actor, make_full_network_actor_interface, actor_to_arrays,
    actor_from_arrays, integrate_frozen_conductance,
)
from r_elegans.brain.circuit import decode_subcircuit_params
from scripts.assay_intrinsic_motor import isolated_trace, cellular_metrics


def test_cell_identity_masks_do_not_select_premotor_or_other_classes():
    a, b = motor_class_masks(("DA1", "VA12", "DB7", "VB11", "AVA", "AVB", "DVA", "RMDL"))
    np.testing.assert_array_equal(a, [1, 1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(b, [0, 0, 1, 1, 0, 0, 0, 0])
    model = load_builtin_model()
    a, b = motor_class_masks(model.neuron_ids)
    assert int(a.sum()) == 21  # DA1-9, VA1-12
    assert int(b.sum()) == 18  # DB1-7, VB1-11


def test_sustained_cellular_rhythms_require_regenerative_current_and_b_drive():
    config = IntrinsicMotorConfig()
    trace = np.asarray(jax.jit(lambda: isolated_trace(config))())
    fine = np.asarray(jax.jit(lambda: isolated_trace(config, substeps=8))())
    metrics = cellular_metrics(trace, .02)
    assert np.isfinite(trace).all()
    for name in ("A_autonomous", "B_tonic"):
        assert metrics[name]["upward_crossings"] >= 3
        assert metrics[name]["voltage_p2p"] > .2
    for name in ("B_quiet", "B_withdrawn"):
        assert metrics[name]["voltage_p2p"] < 1e-4
    assert np.max(np.abs(trace-fine)) < .002
    blocked = jax.jit(lambda: isolated_trace(IntrinsicMotorConfig(calcium_conductance=0.)))()
    assert cellular_metrics(blocked, .02)["A_autonomous"]["voltage_p2p"] < 1e-4


def test_intrinsic_full_network_carry_temporal_gradients_and_legacy_compatibility():
    model = load_builtin_model()
    actor = init_full_network_actor(model, jax.random.PRNGKey(0))
    original = actor_to_arrays(actor)
    intrinsic = make_full_network_actor_interface(
        model, train_group="neural", input_mode="current", intrinsic_motor=IntrinsicMotorConfig())
    legacy = make_full_network_actor_interface(model, input_mode="current")
    assert intrinsic.init_carry(actor).shape == (2, 302)
    assert legacy.init_carry(actor).shape == (302,)

    def rollout(a):
        def step(state, _):
            muscle, state = intrinsic.deterministic_step(a, state, jnp.zeros(302))
            return state, muscle
        return jax.lax.scan(step, intrinsic.init_carry(a), None, length=20)

    state, muscles = jax.jit(rollout)(actor)
    assert np.isfinite(state).all() and np.isfinite(muscles).all()
    assert np.min(muscles) >= 0 and np.max(muscles) <= 1
    grad = jax.jit(jax.grad(lambda a: jnp.mean(rollout(a)[1]**2)))(actor)
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(grad))
    assert np.linalg.norm(grad.neural.raw_chemical) > 0
    assert np.linalg.norm(grad.raw_nmj) == 0  # frozen by neural-only interface

    _, legacy_state = legacy.deterministic_step(actor, legacy.init_carry(actor), jnp.zeros(302))
    expected = integrate_frozen_conductance(actor.neural.leak_reversal,
        decode_subcircuit_params(actor.neural), jnp.zeros(302), .02, 4)
    np.testing.assert_array_equal(legacy_state, expected)
    for key, value in actor_to_arrays(actor_from_arrays(original)).items():
        np.testing.assert_array_equal(value, original[key])
    with pytest.raises(ValueError, match="carry"):
        intrinsic.deterministic_step(actor, legacy.init_carry(actor), jnp.zeros(302))


def test_invalid_intrinsic_coefficients():
    with pytest.raises(ValueError):
        IntrinsicMotorConfig(clock_scale=0)
    with pytest.raises(ValueError):
        IntrinsicMotorConfig(calcium_conductance=float("nan"))
