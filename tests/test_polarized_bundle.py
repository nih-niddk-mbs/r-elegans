import hashlib
import json
from importlib.resources import files

import numpy as np

from r_elegans.brain.conductance_network import load_network
from r_elegans.brain.edge_polarity import apply_edge_polarity, load_edge_polarity


def test_bundled_polarized_network_is_the_pinned_parent_with_only_the_reversal_matrix_changed():
    base = files('r_elegans.assets').joinpath('mac_rhythm_v1')
    polar = files('r_elegans.assets').joinpath('mac_rhythm_v1_polarized')
    info = json.loads(polar.joinpath('inputs.json').read_text())
    assert hashlib.sha256(polar.joinpath('network.npz').read_bytes()).hexdigest() == info['network_sha256']
    assert hashlib.sha256(base.joinpath('network.npz').read_bytes()).hexdigest() in info['derived_from']
    parent_params, parent_fixed, _ = load_network(base.joinpath('network.npz'))
    params, fixed, _ = load_network(polar.joinpath('network.npz'))
    expected = apply_edge_polarity(parent_fixed)
    np.testing.assert_array_equal(fixed.synapse_reversal_mV, expected.synapse_reversal_mV)
    for name in fixed._fields:
        if name not in ('cells', 'synapse_reversal_mV'):
            np.testing.assert_array_equal(getattr(fixed, name), getattr(parent_fixed, name))
    for a, b in zip(params, parent_params):
        np.testing.assert_array_equal(a, b)
    sign = load_edge_polarity()['sign']
    matrix = np.asarray(fixed.synapse_reversal_mV)
    assert np.all(matrix[sign > 0] == 0.) and np.all(matrix[sign < 0] == -85.)
