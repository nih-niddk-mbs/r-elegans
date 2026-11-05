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
    traveling_bend_target, traveling_metrics, traveling_passes,
    reverse_context_vector, directional_muscle_sequence,
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


def test_wave_gate_requires_real_forward_propagation_in_muscles_and_body():
    from scripts.train_motor_alternation import command_sequences
    commands = command_sequences(384, held_out=True)
    metrics = jax.jit(traveling_metrics)
    for command in commands:
        target = traveling_bend_target(command)
        perfect = metrics(target, target)
        assert traveling_passes(perfect)
        assert float(perfect["target_muscle_lag"]) >= .08
        assert float(perfect["target_body_lag"]) >= .08
        uniform = commanded_bend_target(command)
        assert not traveling_passes(metrics(uniform, target))
        reverse = traveling_bend_target(command[::-1])[::-1]
        assert not traveling_passes(metrics(reverse, target))


def test_wave_targets_are_causal_and_resume_matches_seed_and_gate(tmp_path):
    import json
    from scripts.run_motor_curriculum import resume_checkpoints, find_resume_run
    command = jnp.zeros(192).at[90:].set(1.)
    target = traveling_bend_target(command)
    np.testing.assert_allclose(target[:90], .4, atol=1e-7)
    run = tmp_path / "motor-0001"
    folder = run / "seed-3/alternation"
    folder.mkdir(parents=True)
    (folder / "actor_best.npz").write_bytes(b"checkpoint")
    summary = dict(workers=[dict(seed=3, stages=dict(alternation=dict(passed=True, exit_code=0, status="completed")))])
    (run / "summary.json").write_text(json.dumps(summary))
    assert resume_checkpoints(run, [3])[3] == folder / "actor_best.npz"
    assert find_resume_run(tmp_path, [3])[0] == run
    with pytest.raises(ValueError):
        resume_checkpoints(run, [0])
    summary["workers"][0]["stages"]["alternation"]["exit_code"] = 2
    (run / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        resume_checkpoints(run, [3])


def test_reverse_target_is_causal_and_requires_opposite_propagation():
    from scripts.train_motor_alternation import command_sequences
    commands = command_sequences(384, held_out=True)
    metrics = jax.jit(traveling_metrics)
    for command in commands:
        reverse = traveling_bend_target(command, reverse=True)
        perfect = metrics(reverse, reverse)
        assert traveling_passes(perfect, reverse=True)
        assert not traveling_passes(perfect)
        assert float(perfect["target_muscle_lag"]) <= -.08
        forward = traveling_bend_target(command)
        assert not traveling_passes(metrics(forward, reverse), reverse=True)
    step = jnp.zeros(192).at[90:].set(1.)
    np.testing.assert_allclose(traveling_bend_target(step, reverse=True)[:90], .4, atol=1e-7)


def test_direction_selector_preserves_forward_input_and_trains_neural_fields_only():
    model = load_builtin_model()
    actor = init_full_network_actor(model, jax.random.PRNGKey(0))
    interface = make_full_network_actor_interface(model, train_group="neural", input_mode="current")
    stimulation = motor_stimulation_vector(model)
    context = reverse_context_vector(model)
    command = jnp.sin(jnp.arange(12) * .1)
    original = commanded_muscle_sequence(actor, interface, command, stimulation)
    forward = directional_muscle_sequence(actor, interface, command, stimulation, context=context)
    np.testing.assert_array_equal(forward, original)
    backward = directional_muscle_sequence(actor, interface, command, stimulation, context=context, reverse=True)
    assert np.linalg.norm(backward - forward) > 0
    outside = np.asarray(model.neuromuscular_params.synapse_weights.sum(axis=0)) == 0
    np.testing.assert_array_equal(np.asarray(context)[outside], 0)
    def loss(a):
        return jnp.mean(directional_muscle_sequence(a, interface, command, stimulation, context=context, reverse=True)**2)
    gradient = jax.jit(jax.grad(loss))(actor)
    assert np.linalg.norm(gradient.neural.leak_reversal) > 0
    for field in ("raw_nmj", "unknown_polarity", "muscle_threshold", "raw_muscle_slope", "sensory_weights"):
        np.testing.assert_array_equal(getattr(gradient, field), 0)
    np.testing.assert_array_equal(gradient.neural.chemical_mask, 0)


def test_reversing_target_wave_reverses_actual_body_translation():
    from r_elegans.rl.motor_curriculum import muscle_sequence_body
    # After settling, measure complete equal-phase cycles rather than a head tip.
    steps_per_cycle = 120
    commands = .85 * jnp.sin(2*jnp.pi*jnp.arange(steps_per_cycle * 8)/steps_per_cycle)
    displacement = []
    for reverse in (False, True):
        muscles = traveling_bend_target(commands, amplitude=.07, reverse=reverse)
        body = jax.jit(muscle_sequence_body)(muscles)
        displacement.append(float(body.position[-1, 0] - body.position[4*steps_per_cycle-1, 0]))
    assert displacement[0] > .001
    assert displacement[1] < -.001
