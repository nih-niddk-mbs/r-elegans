import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.brain.conductance_network import load_network,network_synapses
from r_elegans.brain.joint_autonomous_fit import (
    joint_fit_reference,apply_joint_offsets,joint_proposal,frozen_joint_fields_unchanged,
)
from r_elegans.brain.single_compartment import CHANNEL_INDEX
from scripts.run_embodied_diagnostics import select_input_bundle


def test_joint_fit_counts_identity_topology_bounds_and_electrical_symmetry():
    initial,fixed,config=load_network(select_input_bundle().parent/'autonomous_wiggle_v1/network.npz')
    fixed,spec=joint_fit_reference(fixed)
    assert spec.counts==dict(channel_densities=4814,chemical_strengths=3709,electrical_pairs=1093)
    assert spec.size==9616 and np.all(spec.channels.any(axis=1))
    assert not spec.channels[:,CHANNEL_INDEX['LEAK']].any()
    np.testing.assert_array_equal(fixed.bias_pA,0.)
    zero=apply_joint_offsets(initial,jnp.zeros(spec.size),spec)
    for a,b in zip(jax.tree_util.tree_leaves(initial),jax.tree_util.tree_leaves(zero)):
        np.testing.assert_array_equal(a,b)
    offsets=jnp.linspace(-np.log(4.),np.log(4.),spec.size)
    changed=apply_joint_offsets(initial,offsets,spec)
    assert frozen_joint_fields_unchanged(changed,initial,spec)
    n,m,_=spec.counts.values()
    np.testing.assert_allclose(np.asarray(changed.log_channel_scale-initial.log_channel_scale)[spec.channels],offsets[:n],atol=1e-7)
    chem,gap=network_synapses(changed,fixed,config);old_chem,old_gap=network_synapses(initial,fixed,config)
    np.testing.assert_allclose(np.asarray(chem/ jnp.where(old_chem>0,old_chem,1.))[spec.chemical],np.exp(offsets[n:n+m]),rtol=2e-6)
    upper=np.triu(spec.gap,1)
    np.testing.assert_allclose(np.asarray(gap/jnp.where(old_gap>0,old_gap,1.))[upper],np.exp(offsets[n+m:]),rtol=2e-6)
    np.testing.assert_array_equal(gap,gap.T)
    assert np.count_nonzero(chem)==3709 and np.count_nonzero(gap)==2*1093
    assert not frozen_joint_fields_unchanged(changed._replace(raw_chemical=changed.raw_chemical+1),initial,spec)


def test_block_search_exposes_all_cells_and_all_parameter_types():
    _,fixed,_=load_network(select_input_bundle().parent/'autonomous_wiggle_v1/network.npz')
    _,spec=joint_fit_reference(fixed);rng=np.random.default_rng(0)
    covered=np.zeros(spec.size,dtype=bool)
    for _ in range(256):
        proposal=joint_proposal(np.zeros(spec.size),spec,rng,.48)
        assert np.max(np.abs(proposal))<=np.log(4.)
        covered|=proposal!=0
    assert covered.mean()>.9
    n,m,_=spec.counts.values()
    assert covered[:n].any() and covered[n:n+m].any() and covered[n+m:].any()


def test_anatomy_focus_remains_a_proposal_prior_without_freezing_other_parameters():
    _,fixed,_=load_network(select_input_bundle().parent/'autonomous_wiggle_v1/network.npz')
    _,spec=joint_fit_reference(fixed);rng=np.random.default_rng(19);covered=np.zeros(spec.size,dtype=bool)
    for _ in range(128):
        proposal=joint_proposal(np.zeros(spec.size),spec,rng,.3,16.,focus_cells=np.arange(20))
        assert np.max(np.abs(proposal))<=np.log(16.)
        covered|=proposal!=0
    n,m,_=spec.counts.values()
    # All neuron rows remain reachable, and edges outside the focused cell set too.
    assert set(np.nonzero(spec.channels)[0][covered[:n]])==set(range(302))
    postsyn=np.nonzero(spec.chemical)[0]
    assert covered[n:n+m][postsyn>=20].any()
