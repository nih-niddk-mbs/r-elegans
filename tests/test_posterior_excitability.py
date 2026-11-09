import jax
import jax.numpy as jnp
import numpy as np
from r_elegans.brain.posterior_excitability import isolated_recording,density_profiles
from r_elegans.brain.conductance_network import load_network,network_recording
from r_elegans.brain.single_compartment import CHANNEL_INDEX


def test_isolation_matches_zero_edge_network_with_same_cells_and_initial_state():
    p,f,c=load_network('r_elegans/assets/autonomous_wiggle_v1/network.npz')
    ids=jnp.array([0,30,200]);cells=jax.tree.map(lambda x:x[ids],f.cells)
    scale=p.log_channel_scale[ids]
    f=f._replace(cells=cells,initial_voltage_mV=f.initial_voltage_mV[ids],bias_pA=jnp.zeros(3),
        release_threshold_mV=f.release_threshold_mV[ids],release_slope_mV=f.release_slope_mV[ids],
        synapse_reversal_mV=f.synapse_reversal_mV[ids],chemical_mask=jnp.zeros((3,3)),gap_mask=jnp.zeros((3,3)))
    p=p._replace(log_channel_scale=scale,raw_chemical=jnp.zeros((3,3)),raw_gap=jnp.zeros((3,3)))
    effective=cells._replace(conductances_nS=cells.conductances_nS*jnp.exp(scale))
    v,ca=isolated_recording(effective,f.initial_voltage_mV,.002,30,20)
    trace=network_recording(p,f,.002,30,20,c)
    np.testing.assert_allclose(v,trace.voltage_mV,atol=2e-5,rtol=1e-6)
    np.testing.assert_allclose(ca,trace.calcium_uM,atol=1e-7,rtol=1e-6)


def test_profiles_keep_original_bounds_frozen_leak_and_determinism():
    mask=np.ones((4,17),bool);mask[:,-1]=False;mask[0,2]=False
    current=np.zeros_like(mask,dtype=float);current[:,CHANNEL_INDEX['CCA1']]=np.log(16)
    a,ra=density_profiles(current,mask,16,8,4000);b,rb=density_profiles(current,mask,16,8,4000)
    assert ra==rb and len(a)==19
    for (name,x),(_,y) in zip(a,b):
        assert np.array_equal(x,y) and np.all(x[~mask]==0)
        assert abs(x).max()<=np.log(16)
    assert np.array_equal(a[0][1],current)
    assert not a[1][1].any()
