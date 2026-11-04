"""A small supervised milestone must improve neural control, not muscle biases."""
import importlib.util
import pytest
if importlib.util.find_spec("gymnax") is None:
    pytest.skip("gymnax extra not installed", allow_module_level=True)

import jax
import jax.numpy as jnp
import numpy as np
import optax

from r_elegans.model import load_builtin_model
from r_elegans.rl.full_network_actor import init_full_network_actor, make_full_network_actor_interface
from r_elegans.rl.motor_curriculum import (
    bend_target, neural_muscle_sequence, make_bend_update, bend_metrics, bend_passes,
    motor_stimulation_vector, commanded_bend_target, commanded_muscle_sequence,
    alternating_metrics, alternating_passes,
)


def test_supervision_improves_full_brain_with_fixed_muscle_mapping():
    model = load_builtin_model()
    actor = init_full_network_actor(model, jax.random.PRNGKey(0))
    original = actor
    interface = make_full_network_actor_interface(model, train_group="neural")
    target = bend_target(24)
    optimizer = optax.adam(0.003)
    state = optimizer.init(actor)
    update = make_bend_update(interface, target, optimizer)
    initial_loss = float(jnp.mean((neural_muscle_sequence(actor, interface, 24) - target) ** 2))
    for _ in range(30):
        actor, state, loss = update(actor, state)
    final_loss = float(jnp.mean((neural_muscle_sequence(actor, interface, 24) - target) ** 2))
    assert np.isfinite(final_loss)
    assert final_loss < initial_loss * 0.5
    assert actor.neural.leak_reversal.shape == (302,)
    assert np.linalg.norm(actor.neural.raw_chemical - original.neural.raw_chemical) > 0
    for field in ("raw_nmj", "unknown_polarity", "muscle_threshold", "raw_muscle_slope", "sensory_weights", "log_std"):
        np.testing.assert_array_equal(getattr(actor, field), getattr(original, field))
    np.testing.assert_array_equal(actor.neural.chemical_mask, original.neural.chemical_mask)
    np.testing.assert_array_equal(actor.neural.gap_mask, original.neural.gap_mask)


def test_bend_gate_checks_actual_body_and_rejects_no_bend():
    target = bend_target(32)
    assert target.shape == (32, 95)
    perfect = jax.jit(bend_metrics)(target, target)
    assert bend_passes(perfect)
    straight = jax.jit(bend_metrics)(jnp.full_like(target, 0.4), target)
    assert not bend_passes(straight)
    wrong_direction = jax.jit(bend_metrics)(0.8 - target, target)
    assert not bend_passes(wrong_direction)


def test_motor_stimulation_keeps_default_senses_and_anatomical_outputs_separate():
    model = load_builtin_model()
    actor = init_full_network_actor(model, jax.random.PRNGKey(0))
    current = make_full_network_actor_interface(model, input_mode="current", train_group="neural")
    chemical = make_full_network_actor_interface(model, train_group="neural")
    voltage = chemical.init_carry(actor)
    for a, b in zip(current.distribution_step(actor, voltage, jnp.zeros(302)),
                    chemical.distribution_step(actor, voltage, jnp.zeros(3))):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="shape"):
        current.deterministic_step(actor, voltage, jnp.zeros(3))
    stimulation = motor_stimulation_vector(model)
    outside = np.asarray(model.neuromuscular_params.synapse_weights.sum(axis=0)) == 0
    np.testing.assert_array_equal(np.asarray(stimulation)[outside], 0)
    assert np.linalg.norm(stimulation) > 0
    plus = commanded_muscle_sequence(actor, current, jnp.ones(8), stimulation)
    minus = commanded_muscle_sequence(actor, current, -jnp.ones(8), stimulation)
    assert plus.shape == (8, 95)
    assert np.linalg.norm(plus - minus) > 0


def test_alternation_gate_rejects_static_or_reversed_output():
    commands = jnp.sin(jnp.arange(192) * .02 * 2 * jnp.pi / 2.4)
    target = commanded_bend_target(commands)
    metrics = jax.jit(alternating_metrics)
    assert alternating_passes(metrics(target, target))
    assert not alternating_passes(metrics(jnp.full_like(target, .4), target))
    assert not alternating_passes(metrics(.8 - target, target))


def test_curriculum_requires_completed_run_as_well_as_a_passing_checkpoint():
    from scripts.run_motor_curriculum import stage_passed
    assert stage_passed(dict(passed=True, exit_code=0, status="completed"))
    assert stage_passed(dict(passed=True, exit_code=0))  # static-bend summary
    assert not stage_passed(dict(passed=True, exit_code=2, status="failed"))
    assert not stage_passed(dict(passed=True, exit_code=0, status="interrupted"))
    assert not stage_passed(dict(passed=False, exit_code=0, status="completed"))
    assert not stage_passed({})
