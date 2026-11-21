import jax.numpy as jnp
import numpy as np

from r_elegans.assets import load_asset_document
from r_elegans.brain.conductance_network import load_network
from r_elegans.brain.extrasynaptic import (
    E_EXCITATORY_MV, E_INHIBITORY_MV, PeptideParams, clip_peptide_params, coupling_matrices, initial_peptide_params,
    load_extrasynaptic_asset, peptide_level_step, peptide_membrane_input)

BUNDLE = 'r_elegans/assets/mac_rhythm_v1_polarized/network.npz'


def test_asset_matches_model_neurons_and_is_binary():
    asset = load_extrasynaptic_asset()
    assert list(asset['neuron_ids']) == list(load_asset_document('runtime_model_v1.json')['neuron_ids'])
    assert asset['sender'].shape == asset['receiver'].shape == (269, 302)
    assert set(np.unique(asset['sender'])) <= {0, 1} and set(np.unique(asset['receiver'])) <= {0, 1}
    assert asset['sender'].sum(axis=1).min() > 0 and asset['receiver'].sum(axis=1).min() > 0
    assert len(asset['metadata']['pair_labels']) == 269


def test_coupling_equals_the_explicit_pair_sum():
    rng = np.random.default_rng(0)
    sender, receiver = (rng.random((7, 9)) < .4).astype(float), (rng.random((7, 9)) < .5).astype(float)
    p = PeptideParams(jnp.asarray(rng.normal(size=7)), jnp.asarray(rng.normal(size=7)), jnp.asarray(np.log(5.)))
    c_exc, c_inh = coupling_matrices(p, sender, receiver)
    gains = np.log1p(np.exp(np.asarray(p.raw_exc)))
    expected = sum(gains[k] * np.outer(receiver[k], sender[k]) for k in range(7))
    assert np.allclose(np.asarray(c_exc), expected) and np.all(np.asarray(c_exc) >= 0) and np.all(np.asarray(c_inh) >= 0)


def test_membrane_input_reversals_and_level_dynamics():
    level = jnp.ones(5)
    c = jnp.eye(5)
    membrane = peptide_membrane_input(level, c, 2. * c, unit_nS=.03)
    assert np.allclose(np.asarray(membrane.conductance_nS), .03 * 3.)
    assert np.allclose(np.asarray(membrane.weighted_reversal_pA), .03 * 2. * E_INHIBITORY_MV + .03 * E_EXCITATORY_MV)
    _, fixed, _ = load_network(BUNDLE)
    voltage = jnp.full(302, 100.)  # far above every release threshold
    level = jnp.zeros(302)
    for _ in range(300):
        level = peptide_level_step(level, voltage, fixed, .1, jnp.log(5.))
    assert np.all(np.asarray(level) > .99) and np.all(np.asarray(level) <= 1.)


def test_initial_parameters_are_small_and_clipping_is_bounded():
    p = initial_peptide_params(269, 5.)
    assert np.isclose(float(np.exp(np.asarray(p.log_tau_s))), 5.)
    assert float(np.max(np.log1p(np.exp(np.asarray(p.raw_exc))))) < 1e-3
    wild = clip_peptide_params(PeptideParams(jnp.full(3, 50.), jnp.full(3, -50.), jnp.asarray(50.)))
    assert float(wild.raw_exc.max()) <= 2.95 and float(wild.raw_inh.min()) >= -12. and float(np.exp(wild.log_tau_s)) <= 30. + 1e-9
