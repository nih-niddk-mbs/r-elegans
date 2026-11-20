import jax
import jax.numpy as jnp
import numpy as np
import pytest

from r_elegans.brain.conductance_network import (ConductanceNetworkConfig, advance_network, initial_network_state, initialize_conductance_network,
                                                 load_network, save_network)
from r_elegans.data import synapse_polarity
from r_elegans.model import load_builtin_model
from r_elegans.rl.full_network_actor import init_full_network_actor
from scripts.apply_synapse_polarity import polarized_reversal


@pytest.fixture(scope='module')
def network():
    model = load_builtin_model()
    params, fixed = initialize_conductance_network(init_full_network_actor(model, jax.random.PRNGKey(0)), model.neuron_ids)
    return params, fixed, ConductanceNetworkConfig()


def step(network, fixed):
    params, _, config = network
    state = initial_network_state(fixed, config)
    return advance_network(state, params, fixed, .1, 1, config)  # one substep: a change cannot propagate beyond its target


def test_edge_matrix_equal_to_the_broadcast_presynaptic_vector_gives_identical_dynamics(network):
    _, fixed, _ = network
    matrix = jnp.broadcast_to(fixed.synapse_reversal_mV[None, :], (302, 302))
    a, b = step(network, fixed), step(network, fixed._replace(synapse_reversal_mV=matrix))
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_allclose(x, y, atol=1e-12)


def test_flipping_one_edge_changes_only_its_postsynaptic_neuron(network):
    params, fixed, config = network
    state = initial_network_state(fixed, config)
    release = np.asarray(jax.nn.sigmoid((state.voltage_mV - fixed.release_threshold_mV) / fixed.release_slope_mV))
    has_target = np.asarray(fixed.chemical_mask).sum(axis=0) > 0
    pre = int(np.argmax(np.where(has_target, release, -1.)))
    assert release[pre] > .01  # an edge from a silent presynaptic cell would carry no drive
    post = int(np.nonzero(np.asarray(fixed.chemical_mask)[:, pre])[0][0])
    matrix = np.broadcast_to(np.asarray(fixed.synapse_reversal_mV)[None, :], (302, 302)).copy()
    base = step(network, fixed._replace(synapse_reversal_mV=jnp.array(matrix)))
    matrix[post, pre] = matrix[post, pre] - 80.
    flipped = step(network, fixed._replace(synapse_reversal_mV=jnp.array(matrix)))
    difference = np.abs(np.asarray(base.voltage_mV) - np.asarray(flipped.voltage_mV))
    assert difference[post] > 0 and np.all(np.delete(difference, post) < 1e-12)


def test_checkpoint_round_trip_keeps_an_edge_reversal_matrix(network, tmp_path):
    params, fixed, config = network
    matrix = jnp.broadcast_to(fixed.synapse_reversal_mV[None, :], (302, 302)) - 1.
    save_network(tmp_path / 'n.npz', params, fixed._replace(synapse_reversal_mV=matrix), config)
    assert load_network(tmp_path / 'n.npz')[1].synapse_reversal_mV.shape == (302, 302)


def test_polarized_reversal_sets_signed_edges_and_keeps_or_neutralizes_the_rest():
    sign = np.array([[1, -1], [0, 0]], np.int8)
    old = np.array([-30., -10.])
    keep = polarized_reversal(old, sign, excitatory_mV=0., inhibitory_mV=-85., undetermined='keep', neutral_mV=-40.)
    assert keep.tolist() == [[0., -85.], [-30., -10.]]
    neutral = polarized_reversal(old, sign, excitatory_mV=0., inhibitory_mV=-85., undetermined='neutral', neutral_mV=-40.)
    assert neutral.tolist() == [[0., -85.], [-40., -40.]]


def test_edge_polarity_prefers_receptor_prediction_and_leaves_unpredicted_edges_undetermined(monkeypatch):
    nt_r = {('A', 'B'): -1, ('A', 'C'): 0}
    nt_only = {('A', 'B'): 1, ('A', 'C'): 1, ('C', 'B'): 0}
    monkeypatch.setattr(synapse_polarity, 'parse_polarity_tables', lambda root: (nt_r, nt_only))
    counts = np.zeros((3, 3))
    counts[1, 0], counts[2, 0], counts[1, 2], counts[0, 1] = 5, 2, 1, 1  # A->B, A->C, C->B, B->A (absent from tables)
    sign, source, conflict, unknown = synapse_polarity.edge_polarity('unused', ['A', 'B', 'C'], counts)
    assert sign[1, 0] == -1 and source[1, 0] == 1 and conflict[1, 0]
    assert sign[2, 0] == 1 and source[2, 0] == 2 and not conflict[2, 0]
    assert sign[1, 2] == 0 and sign[0, 1] == 0
    assert unknown == []
