import hashlib
import json
from importlib.resources import files

import jax
import numpy as np

from r_elegans.brain.conductance_network import ConductanceNetworkConfig, initialize_conductance_network
from r_elegans.brain.edge_polarity import apply_edge_polarity, load_edge_polarity, polarized_reversal
from r_elegans.data.connectome import load_connectome
from r_elegans.model import load_builtin_model
from r_elegans.rl.full_network_actor import init_full_network_actor


def test_bundled_polarity_asset_matches_the_runtime_topology_and_its_recorded_counts():
    polarity = load_edge_polarity()
    meta = polarity['metadata']
    chemical = np.asarray(load_connectome().chemical_counts) > 0
    sign = polarity['sign']
    assert sign.shape == (302, 302) and list(polarity['neuron_ids']) == list(load_builtin_model().neuron_ids)
    assert not np.any((sign != 0) & ~chemical)
    assert meta['runtime_model_sha256'] == hashlib.sha256(files('r_elegans.assets').joinpath('runtime_model_v1.json').read_bytes()).hexdigest()
    assert meta['model_chemical_edges'] == int(chemical.sum())
    assert (int((sign > 0).sum()), int((sign < 0).sum())) == (meta['excitatory'], meta['inhibitory'])
    assert meta['edges_nt_receptor'] + meta['edges_nt_only'] + meta['edges_undetermined'] == meta['model_chemical_edges']
    assert int(((polarity['source'] == 0) & chemical).sum()) == meta['edges_undetermined']
    assert json.loads(json.dumps(meta))['reversal_priors_mV']['status'].startswith('project priors')


def test_initializing_with_edge_polarity_equals_polarizing_the_legacy_initialization():
    model = load_builtin_model()
    parent = init_full_network_actor(model, jax.random.PRNGKey(0))
    config = ConductanceNetworkConfig()
    legacy_params, legacy = initialize_conductance_network(parent, model.neuron_ids, config)
    params, polar = initialize_conductance_network(parent, model.neuron_ids, config, edge_polarity=True)
    expected = apply_edge_polarity(legacy)
    assert polar.synapse_reversal_mV.shape == (302, 302)
    np.testing.assert_array_equal(polar.synapse_reversal_mV, expected.synapse_reversal_mV)
    for name in ('release_threshold_mV', 'release_slope_mV', 'initial_voltage_mV', 'bias_pA'):
        np.testing.assert_array_equal(getattr(polar, name), getattr(legacy, name))
    sign = load_edge_polarity()['sign']
    matrix = np.asarray(polar.synapse_reversal_mV)
    assert np.all(matrix[sign > 0] == 0.) and np.all(matrix[sign < 0] == -85.)
    undetermined = (sign == 0) & (np.asarray(legacy.chemical_mask) > 0)
    np.testing.assert_array_equal(matrix[undetermined], np.broadcast_to(np.asarray(legacy.synapse_reversal_mV)[None, :], matrix.shape)[undetermined])
    np.testing.assert_array_equal(params.raw_chemical, legacy_params.raw_chemical)


def test_polarized_reversal_rejects_unknown_undetermined_policy():
    try:
        polarized_reversal(np.zeros(2), np.zeros((2, 2), np.int8), undetermined='drop')
    except ValueError:
        return
    raise AssertionError('expected ValueError')
