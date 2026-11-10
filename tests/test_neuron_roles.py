"""Canonical ID/order and overlapping-function annotation invariants."""
import pytest

from r_elegans.assets import load_asset_document
from r_elegans.neuron_roles import ROLE_GROUPS, annotate_neuron_roles


def test_curated_members_exist_and_unknown_functions_are_not_invented():
    ids = load_asset_document('runtime_model_v1.json')['neuron_ids']
    rows = annotate_neuron_roles(ids)
    assert [row['neuron'] for row in rows] == ids
    assert [row['index'] for row in rows] == list(range(302))
    assert len({g.key for g in ROLE_GROUPS}) == len(ROLE_GROUPS)
    by_name = {r['neuron']:r for r in rows}
    assert by_name['CANL']['annotation_status'] == 'unreviewed'
    assert by_name['CANL']['roles'] == []  # unreviewed does not mean no function
    assert set(by_name['AVBL']['roles']) == {'interneuron','premotor'}
    assert set(by_name['DB01']['roles']) == {'motor','proprioceptive'}
    assert set(by_name['SMDDL']['roles']) == {'motor','proprioceptive'}
    assert set(by_name['DVA']['roles']) == {'interneuron','proprioceptive'}
    assert 'sensory' in by_name['AWCL']['roles']
    assert 'state_control' in by_name['RIS']['roles']
    assert all(g.sources and len(g.neurons) == len(set(g.neurons)) for g in ROLE_GROUPS)


def test_malformed_inventory_fails_instead_of_silently_losing_members():
    ids = load_asset_document('runtime_model_v1.json')['neuron_ids']
    with pytest.raises(ValueError, match='unique'):
        annotate_neuron_roles(ids+[ids[0]])
    with pytest.raises(ValueError, match='AWCL'):
        annotate_neuron_roles([n for n in ids if n != 'AWCL'])
