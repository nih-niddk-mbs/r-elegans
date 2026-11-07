import jax
import numpy as np

from r_elegans.brain.conductance_network import network_recording,initialize_conductance_network
from r_elegans.brain.rhythm_qualification import isolated_rmd_grid,rhythm_metrics,apply_rmd_rhythm_prior,compare_cycles
from r_elegans.model import load_builtin_model
from r_elegans.rl.full_network_actor import init_full_network_actor


def test_persistence_gate_rejects_plateaus_transients_and_blowups():
    time=np.arange(0.,10.,.002)
    sine=-30.+5.*np.sin(2*np.pi*time/.2)
    assert rhythm_metrics(sine,.002)['passed']
    assert not rhythm_metrics(np.full(len(time),-30.),.002)['passed']
    assert not rhythm_metrics(-30.+5*np.exp(-time)*np.sin(2*np.pi*time/.2),.002)['passed']
    assert not rhythm_metrics(sine+150.,.002)['passed']
    shifted=-30.+5*np.sin(2*np.pi*time/.2+1.)
    assert compare_cycles(sine,shifted,.002)['passed']
    assert not compare_cycles(sine,-30.+5*np.sin(2*np.pi*time/.3),.002)['passed']


def test_conductance_cycle_persists_without_input_and_requires_calcium():
    p,f,_=isolated_rmd_grid([.045],[30.,0.])
    coarse=jax.jit(lambda:network_recording(p,f,.002,3000,20))()
    fine=jax.jit(lambda:network_recording(p,f,.002,3000,40))()
    a=rhythm_metrics(coarse.voltage_mV[:,0],.002)
    b=rhythm_metrics(fine.voltage_mV[:,0],.002)
    assert a['passed'] and b['passed']
    assert not rhythm_metrics(fine.voltage_mV[:,1],.002)['passed']
    assert abs(a['median_period_s']-b['median_period_s'])<.005
    assert np.isfinite(fine.calcium_uM).all()
    np.testing.assert_array_equal(f.bias_pA,0.)


def test_localized_profile_preserves_anatomy_and_other_cells():
    m=load_builtin_model()
    parent=init_full_network_actor(m,jax.random.PRNGKey(0))
    p,f=initialize_conductance_network(parent,m.neuron_ids)
    q,g,indices=apply_rmd_rhythm_prior(p,f,m.neuron_ids)
    other=np.ones(302,dtype=bool);other[np.asarray(indices)]=False
    assert len(indices)==6
    np.testing.assert_array_equal(q.raw_chemical,p.raw_chemical)
    np.testing.assert_array_equal(q.raw_gap,p.raw_gap)
    np.testing.assert_array_equal(g.chemical_mask,f.chemical_mask)
    np.testing.assert_array_equal(g.gap_mask,f.gap_mask)
    for before,after in zip(jax.tree_util.tree_leaves(f.cells),jax.tree_util.tree_leaves(g.cells)):
        np.testing.assert_array_equal(before[other],after[other])
    np.testing.assert_array_equal(g.bias_pA[indices],0.)
