import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from r_elegans.brain.conductance_network import initialize_conductance_network
from r_elegans.brain.fit_constraints import conservative_fit_constraints,project_unknowns,frozen_parameter_changes
from r_elegans.brain.single_compartment import CHANNEL_INDEX
from r_elegans.model import load_builtin_model
from r_elegans.rl.full_network_actor import init_full_network_actor


def fixture():
    m=load_builtin_model();p,f=initialize_conductance_network(init_full_network_actor(m,jax.random.PRNGKey(0)),m.neuron_ids)
    return m,p,f


def test_stereotyped_anchors_and_unobserved_channels_stay_fixed():
    m,p,f=fixture();masks,policy=conservative_fit_constraints(f,m.neuron_ids,['ASEL','RMDL'])
    assert not np.any(masks.channels[:,CHANNEL_INDEX['LEAK']])
    assert not np.any(masks.channels[m.neuron_ids.index('RMDL')])
    assert not np.any(masks.channels[m.neuron_ids.index('ASER')])
    assert int(masks.channels[m.neuron_ids.index('ASEL')].sum())==16
    assert policy['trainable']==dict(channel_conductances=16,chemical_strengths=3709,independent_gap_strengths=1093)
    np.testing.assert_array_equal(masks.gap,masks.gap.T)


def test_projection_blocks_gradients_and_optimizer_momentum_changes():
    m,p,f=fixture();masks,_=conservative_fit_constraints(f,m.neuron_ids,['ASEL'])
    loss=lambda q:sum(jnp.sum(a**2) for a in project_unknowns(q,p,masks))
    q=jax.tree.map(lambda a:a+.1,p)
    gradient=jax.grad(loss)(q)
    for a,mask in zip(gradient,masks):assert np.all(np.asarray(a)[~np.asarray(mask)]==0)
    optimizer=optax.adam(.01);state=optimizer.init(p)
    # Deliberately build momentum on every field; projection must still freeze.
    updates,state=optimizer.update(jax.tree.map(jnp.ones_like,p),state,p)
    new=project_unknowns(optax.apply_updates(p,updates),p,masks)
    assert all(n==0 for n in frozen_parameter_changes(new,p,masks).values())
    assert np.max(np.abs(new.log_channel_scale-p.log_channel_scale))>0


def test_additional_known_edges_freeze_both_gap_directions():
    m,p,f=fixture();post,pre=np.argwhere(np.triu(np.asarray(f.gap_mask),1))[0]
    extra={'frozen_edges':[dict(kind='gap',post=m.neuron_ids[post],pre=m.neuron_ids[pre],
                               source='test known-pair constraint',reason='hold conductance fixed')]}
    masks,_=conservative_fit_constraints(f,m.neuron_ids,['ASEL'],extra)
    assert not masks.gap[post,pre] and not masks.gap[pre,post]
    with pytest.raises(ValueError,match='source'):
        conservative_fit_constraints(f,m.neuron_ids,['ASEL'],{'frozen_channels':[dict(neuron='ASEL',channel='UNC2')]})
    with pytest.raises(ValueError,match='Unrecognized'):
        conservative_fit_constraints(f,m.neuron_ids,['ASEL'],{'train_capacitance':True})
