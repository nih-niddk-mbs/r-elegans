import json
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from r_elegans.brain.conductance_network import (
    ConductanceNetworkConfig,initialize_conductance_network,network_synapses,
    network_recording,initial_network_state,channel_open_fractions,save_network,load_network,
)
from r_elegans.brain.single_compartment import ionic_currents,SingleCompartmentState
from r_elegans.model import load_builtin_model
from r_elegans.rl.full_network_actor import init_full_network_actor
from r_elegans.data.calcium import load_identified_calcium


@pytest.fixture
def network():
    model=load_builtin_model()
    parent=init_full_network_actor(model,jax.random.PRNGKey(0))
    params,fixed=initialize_conductance_network(parent,model.neuron_ids)
    return params,fixed,parent


def test_all_cells_have_channel_states_and_original_edges(network,tmp_path):
    params,fixed,parent=network
    config=ConductanceNetworkConfig()
    state=initial_network_state(fixed,config)
    assert state.gates.shape == (302,35)
    assert fixed.cells.conductances_nS.shape == (302,17)
    assert np.all(fixed.cells.conductances_nS > 0)
    chemical,gap=network_synapses(params,fixed,config)
    np.testing.assert_array_equal(chemical>0,parent.neural.chemical_mask>0)
    np.testing.assert_array_equal(gap>0,parent.neural.gap_mask>0)
    np.testing.assert_allclose(gap,gap.T,atol=1e-9)
    np.testing.assert_array_equal(params.raw_chemical,parent.neural.raw_chemical)
    file=tmp_path/'network.npz';save_network(file,params,fixed,config)
    restored=load_network(file)
    for a,b in zip(jax.tree_util.tree_leaves((params,fixed)),jax.tree_util.tree_leaves(restored[:2])):
        np.testing.assert_array_equal(a,b)
    assert restored[2] == config


def test_open_fractions_reproduce_existing_ionic_currents(network):
    _,fixed,_=network
    state=initial_network_state(fixed,ConductanceNetworkConfig())
    current=jax.vmap(ionic_currents)(SingleCompartmentState(state.voltage_mV,state.gates,state.calcium_uM),fixed.cells)
    reversals=jnp.stack([fixed.cells.calcium_reversal_mV if i in (6,7,8)
        else fixed.cells.sodium_reversal_mV if i==14 else fixed.cells.leak_reversal_mV if i==16
        else fixed.cells.potassium_reversal_mV for i in range(17)],axis=1)
    expected=fixed.cells.conductances_nS*channel_open_fractions(state.gates)*(state.voltage_mV[:,None]-reversals)
    np.testing.assert_allclose(current,expected,rtol=2e-6,atol=1e-6)


def test_calcium_and_synaptic_gradients_and_convergence(network):
    params,fixed,_=network
    run=jax.jit(lambda p:network_recording(p,fixed,.02,5,160))
    coarse=run(params)
    fine=jax.jit(lambda:network_recording(params,fixed,.02,5,320))()
    assert all(np.isfinite(v).all() for v in coarse)
    assert np.min(coarse.calcium_uM)>=0
    assert np.min(coarse.indicator)>=0 and np.max(coarse.indicator)<=1
    assert np.max(np.abs(coarse.voltage_mV-fine.voltage_mV))<.2
    gradient=jax.jit(jax.grad(lambda p:jnp.mean(run(p).indicator[-1])))(params)
    assert all(np.isfinite(v).all() for v in gradient)
    assert np.linalg.norm(gradient.log_channel_scale)>0
    assert np.linalg.norm(gradient.raw_chemical)>0
    assert np.linalg.norm(gradient.raw_gap)>0
    assert np.max(np.abs(np.asarray(gradient.raw_chemical)[np.asarray(fixed.chemical_mask)==0])) == 0


def test_channels_do_not_require_spikes(network):
    params,fixed,_=network
    g=jnp.zeros_like(fixed.cells.conductances_nS).at[:,16].set(.03)
    fixed=fixed._replace(cells=fixed.cells._replace(conductances_nS=g,leak_reversal_mV=jnp.full(302,-60.)),
        chemical_mask=jnp.zeros((302,302)),gap_mask=jnp.zeros((302,302)),
        bias_pA=jnp.zeros(302),initial_voltage_mV=jnp.full(302,-70.))
    state=jax.jit(lambda:network_recording(params,fixed,.02,10,40))()
    assert np.all(np.diff(state.voltage_mV,axis=0)>0)
    assert np.max(state.voltage_mV)<-60.
    np.testing.assert_allclose(state.calcium_uM,.05,atol=1e-7)


def test_calcium_import_excludes_ambiguous_labels_and_uses_train_only_scale(tmp_path):
    h5py=pytest.importorskip('h5py')
    file=tmp_path/'record.h5';labels=tmp_path/'labels.json'
    values=np.stack((np.arange(13),np.arange(13)**2,np.arange(13)),axis=1).astype(float)
    values[10:,0]+=1000
    with h5py.File(file,'w') as f:
        f['timing/timestamp_confocal']=np.arange(13)*.6
        f['gcamp/trace_array_original']=values
    labels.write_text(json.dumps({'data':{'worm':{'idx_neuron-label':{
        '1':{'label':'AVAL','confidence':5},'2':{'label':'AVA?','confidence':5},
        '3':{'label':'AVBR','confidence':2}}}}}))
    d=load_identified_calcium(file,labels,load_builtin_model().neuron_ids,'worm',frames=12)
    assert d['names'] == ['AVAL']
    assert d['rows'] == [0] and d['train_frames']==9
    np.testing.assert_allclose(d['training_mean'],np.mean(np.arange(1,10)))
    assert np.min(d['normalized'][9:])>100
