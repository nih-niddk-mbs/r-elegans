from dataclasses import replace
from importlib.resources import files

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from r_elegans.brain.conductance_network import load_network, advance_network, MembraneConductanceInput
from r_elegans.body import physical_muscle_body_params
from r_elegans.body.channel_bridge import channel_body_chunk
from r_elegans.body.closed_loop import initial_embodied_state, embodied_channel_body_chunk
from r_elegans.body.embodied_sensory import (
    EmbodiedSensoryConfig, build_sensory_maps, default_chemical_world,
    initial_embodied_sensory_state, sensory_head_position, sample_chemical_world,
    transduce_sensory_signals, validate_chemical_world,
)
from r_elegans.full_body_assessment import rod_nodes
from r_elegans.model import load_builtin_model
from r_elegans.rl.full_network_actor import actor_from_arrays


@pytest.fixture(scope='module')
def setup():
    root = files('r_elegans.assets').joinpath('mac_rhythm_v1')
    params, fixed, config = load_network(root/'network.npz')
    with np.load(root/'parent_actor.npz', allow_pickle=False) as z:
        parent = actor_from_arrays(z)
    model = load_builtin_model()
    body = physical_muscle_body_params(12, medium='liquid')
    world = default_chemical_world()
    maps = build_sensory_maps(model)
    state = initial_embodied_state(fixed, config, body, world)
    return params, fixed, config, parent, model, body, world, maps, state


def test_true_head_endpoint_and_translation_invariance(setup):
    _, _, _, _, _, bp, world, _, state = setup
    body = state.brain_body.body._replace(joint_angles=.1*jnp.sin(jnp.arange(11)),
        position=jnp.array([.002, -.003]), heading=jnp.array(.7))
    endpoint = sensory_head_position(body, bp)
    expected = rod_nodes(np.asarray(body.joint_angles)[None, :], np.asarray(body.position)[None, :],
        np.asarray(body.heading)[None], float(bp.mechanics.segment_length)*12)[0, -1]
    np.testing.assert_allclose(endpoint, expected, atol=5e-10)
    shift = jnp.array([.01, -.02])
    translated = body._replace(position=body.position+shift)
    moved_world = world._replace(source_position_m=world.source_position_m+shift)
    a = sample_chemical_world(endpoint, 1.5, world)
    b = sample_chemical_world(sensory_head_position(translated, bp), 1.5, moved_world)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-10)


def test_odor_and_salt_response_directions_and_locality(setup):
    _, _, _, _, model, bp, world, maps, state = setup
    sc = EmbodiedSensoryConfig(b_bend_gain_nS=0., dva_gain_nS=0.)
    body = state.brain_body.body
    head = sensory_head_position(body, bp)
    world = world._replace(source_position_m=jnp.broadcast_to(head, (3, 2)),
        onset_s=jnp.ones(3), washout_s=jnp.full(3, 10.))
    initial = initial_embodied_sensory_state(body, bp, world, sc)
    body = body._replace(time=jnp.array(2.))
    memory, membrane, obs = transduce_sensory_signals(initial, body, bp, world, maps, .1, sc)
    ids = model.neuron_ids
    assert obs.chemical_excitatory_nS[ids.index('AWAL')] > 0
    assert obs.chemical_inhibitory_nS[ids.index('AWCL')] > 0
    assert obs.chemical_excitatory_nS[ids.index('ASEL')] > 0
    assert obs.chemical_excitatory_nS[ids.index('ASER')] == 0
    assert obs.chemical_excitatory_nS[ids.index('AWCL')] == 0
    for _ in range(30):
        memory, _, _ = transduce_sensory_signals(memory, body, bp, world, maps, .1, sc)
    washed = world._replace(peak_um=jnp.zeros(3))
    _, _, down = transduce_sensory_signals(memory, body, bp, washed, maps, .1, sc)
    assert down.chemical_excitatory_nS[ids.index('AWCL')] > 0
    assert down.chemical_excitatory_nS[ids.index('ASER')] > 0
    assert down.chemical_excitatory_nS[ids.index('ASEL')] == 0
    outside = np.ones(302, dtype=bool)
    outside[np.asarray(maps.chemical_indices)] = False
    np.testing.assert_array_equal(np.asarray(membrane.conductance_nS)[outside], 0.)
    assert np.all(np.asarray(membrane.conductance_nS) >= 0)


def test_actual_bending_drives_dorsal_ventral_and_dva_targets(setup):
    _, _, _, _, _, bp, world, maps, state = setup
    sc = EmbodiedSensoryConfig(chemical_gain_nS=0., smdd_gain_nS=.002)
    body = state.brain_body.body._replace(joint_angles=jnp.full(11, .02))
    memory = initial_embodied_sensory_state(body, bp, world, sc)
    _, _, positive = transduce_sensory_signals(memory, body, bp, world, maps, .002, sc)
    body = body._replace(joint_angles=-body.joint_angles)
    memory = initial_embodied_sensory_state(body, bp, world, sc)
    _, _, negative = transduce_sensory_signals(memory, body, bp, world, maps, .002, sc)
    b = np.asarray(maps.b_indices)
    assert np.all(np.asarray(positive.proprioceptive_nS)[b[:7]] > 0)
    np.testing.assert_array_equal(np.asarray(positive.proprioceptive_nS)[b[7:]], 0.)
    assert np.all(np.asarray(negative.proprioceptive_nS)[b[7:]] > 0)
    np.testing.assert_array_equal(np.asarray(negative.proprioceptive_nS)[b[:7]], 0.)
    np.testing.assert_allclose(positive.proprioceptive_nS[maps.dva_index], negative.proprioceptive_nS[maps.dva_index])
    allowed = np.r_[b, np.asarray(maps.dva_index).reshape(1), np.asarray(maps.smdd_indices)]
    outside = np.ones(302, dtype=bool)
    outside[allowed] = False
    np.testing.assert_array_equal(np.asarray(positive.proprioceptive_nS)[outside], 0.)
    assert np.all(np.asarray(positive.proprioceptive_nS) >= 0)


def test_membrane_input_matches_passive_conductance_solution(setup):
    params, fixed, config, _, _, _, _, _, state = setup
    fixed = fixed._replace(cells=fixed.cells._replace(conductances_nS=jnp.zeros((302, 17)),
        capacitance_pF=jnp.ones(302)), chemical_mask=jnp.zeros((302, 302)),
        gap_mask=jnp.zeros((302, 302)), bias_pA=jnp.zeros(302))
    initial = state.brain_body.brain._replace(voltage_mV=jnp.full(302, -60.))
    membrane = MembraneConductanceInput(jnp.zeros(302).at[0].set(.1), jnp.zeros(302))
    result = jax.jit(lambda s:advance_network(s, params, fixed, .002, 1, config, membrane_input=membrane))(initial)
    np.testing.assert_allclose(result.voltage_mV[0], -60*np.exp(-2*.1), rtol=1e-6)
    np.testing.assert_array_equal(result.voltage_mV[1:], initial.voltage_mV[1:])


def test_disabled_gains_recover_legacy_and_streaming_preserves_memory(setup):
    params, fixed, config, parent, model, bp, world, maps, state = setup
    sc = EmbodiedSensoryConfig(chemical_gain_nS=0., b_bend_gain_nS=0., dva_gain_nS=0.)
    run = jax.jit(lambda s, n:embodied_channel_body_chunk(s, params, fixed, config, parent,
        model, bp, world, maps, steps=n, sensory_config=sc), static_argnums=1)
    full, _ = run(state, 20)
    first, _ = run(state, 10)
    split, _ = run(first, 10)
    legacy, _ = jax.jit(lambda s:channel_body_chunk(s, params, fixed, config, parent, model, bp, steps=20))(state.brain_body)
    for a, b in zip(jax.tree_util.tree_leaves(full), jax.tree_util.tree_leaves(split)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    for a, b in zip(jax.tree_util.tree_leaves(full.brain_body), jax.tree_util.tree_leaves(legacy)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    # Also test nonzero sensory memory across a stimulus onset, not just zeros.
    world = world._replace(onset_s=jnp.full(3, .01), washout_s=jnp.full(3, 10.))
    active = replace(sc, chemical_gain_nS=.05, b_bend_gain_nS=.005, dva_gain_nS=.002)
    run_active = jax.jit(lambda s, n:embodied_channel_body_chunk(s, params, fixed, config, parent,
        model, bp, world, maps, steps=n, sensory_config=active), static_argnums=1)
    full_active, trace = run_active(state, 40)
    first, _ = run_active(state, 20)
    split_active, _ = run_active(first, 20)
    for a, b in zip(jax.tree_util.tree_leaves(full_active), jax.tree_util.tree_leaves(split_active)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    blocked, blocked_trace = jax.jit(lambda s:embodied_channel_body_chunk(s, params, fixed, config,
        parent, model, bp, world, maps, steps=40, sensory_config=sc))(state)
    outside = np.ones(302, dtype=bool)
    outside[np.asarray(maps.chemical_indices)] = False
    assert np.max(np.abs(np.asarray(full_active.brain_body.brain.voltage_mV-blocked.brain_body.brain.voltage_mV)[outside])) > 1e-6
    assert np.max(np.abs(np.asarray(trace.muscles-blocked_trace.muscles))) > 1e-8
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(full_active))


def test_invalid_configuration_and_world_rejected():
    import json
    from dataclasses import asdict
    original = EmbodiedSensoryConfig()
    assert EmbodiedSensoryConfig(**json.loads(json.dumps(asdict(original)))) == original
    with pytest.raises(ValueError):
        EmbodiedSensoryConfig(chemical_gain_nS=-.1)
    with pytest.raises(ValueError):
        EmbodiedSensoryConfig(curvature_reference_per_m=0.)
    with pytest.raises(ValueError):
        validate_chemical_world(default_chemical_world()._replace(width_m=jnp.zeros(3)))


def test_trainable_gain_multipliers_preserve_routing_and_block_selectively(setup):
    _, _, _, _, _, bp, world, maps, state = setup
    body=state.brain_body.body._replace(joint_angles=jnp.full(11,.02))
    memory=initial_embodied_sensory_state(body,bp,world)
    body=body._replace(time=jnp.array(1.5))
    def run(gains):
        return transduce_sensory_signals(memory,body,bp,world,maps,.002,
            gain_multipliers=jnp.asarray(gains))[2]
    normal=run([1.,1.,1.]);scaled=run([2.,.5,1.5]);blocked=run([0.,1.,1.])
    np.testing.assert_allclose(scaled.chemical_excitatory_nS,2*normal.chemical_excitatory_nS)
    np.testing.assert_array_equal(blocked.chemical_excitatory_nS,0.)
    np.testing.assert_array_equal(blocked.proprioceptive_nS,normal.proprioceptive_nS)
    np.testing.assert_allclose(scaled.proprioceptive_nS[maps.b_indices],.5*normal.proprioceptive_nS[maps.b_indices])
    np.testing.assert_allclose(scaled.proprioceptive_nS[maps.dva_index],1.5*normal.proprioceptive_nS[maps.dva_index])
