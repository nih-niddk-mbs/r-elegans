import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.brain.autonomous_wiggle import (
    autonomous_fit_reference,apply_density_offsets,density_proposal,
    amplitude_metrics,frozen_fit_fields_unchanged,
)
from r_elegans.brain.conductance_network import load_network
from r_elegans.brain.single_compartment import CHANNEL_INDEX
from r_elegans.model import load_builtin_model
from scripts.run_embodied_diagnostics import select_input_bundle,verify_input_bundle


def test_intrinsic_fit_has_zero_injection_and_exact_fixed_fields():
    initial,fixed,_=load_network(select_input_bundle()/'network.npz')
    ids=load_builtin_model().neuron_ids
    autonomous,mask=autonomous_fit_reference(initial,fixed,ids)
    np.testing.assert_array_equal(autonomous.bias_pA,0.)
    for field in fixed._fields:
        if field=='bias_pA':continue
        for a,b in zip(jax.tree_util.tree_leaves(getattr(autonomous,field)),
                       jax.tree_util.tree_leaves(getattr(fixed,field))):
            np.testing.assert_array_equal(a,b)
    assert not mask[:,CHANNEL_INDEX['LEAK']].any()
    assert not mask[[not n.startswith('RMD') for n in ids]].any()
    rng=np.random.default_rng(4)
    offsets=density_proposal(np.zeros(mask.sum()),mask,rng,10.)
    assert np.max(np.abs(offsets))<=np.log(4.)
    changed=apply_density_offsets(initial,jnp.asarray(offsets),mask)
    assert frozen_fit_fields_unchanged(changed,initial,mask)
    assert not frozen_fit_fields_unchanged(changed._replace(raw_chemical=changed.raw_chemical+1),initial,mask)


def test_amplitude_rewards_repeated_bends_and_rejects_static_or_clipped_posture():
    dt=.002;t=np.arange(0,12,dt);wave=np.sin(2*np.pi*t/.22)
    voltage=-50+10*np.tile(wave[:,None],(1,3))
    muscle=np.full((len(t),95),.5)
    q=.005*wave[:,None]
    def score(joints):return amplitude_metrics(voltage,muscle,joints,dt,('RMDL','RMDR','RMDVL'),.75)
    metrics=score(q)
    assert metrics['valid'] and metrics['rhythmic_rmd']==3
    assert .009<metrics['sustained_joint_p2p_rad']<.011
    assert not score(np.full_like(q,.3))['valid']
    assert not score(q*np.exp(-t[:,None]))['valid']
    assert not score(q+.75)['valid']


def test_bundled_amplitude_fit_has_no_injection_or_hidden_readout_change():
    root=select_input_bundle().parent/'autonomous_wiggle_v1'
    manifest=verify_input_bundle(root)
    assert manifest['sha256']['network.npz']=='482995e03f28cfd62704b20630c28c2fb1706aec87dbd4e2cc6eecd9315a2ff6'
    assert manifest['sha256']['parent_actor.npz']==verify_input_bundle(select_input_bundle())['sha256']['parent_actor.npz']
    initial,original,config=load_network(select_input_bundle()/'network.npz')
    expected,mask=autonomous_fit_reference(initial,original,load_builtin_model().neuron_ids)
    fitted,fixed,other_config=load_network(root/'network.npz')
    assert config==other_config
    assert frozen_fit_fields_unchanged(fitted,initial,mask)
    for a,b in zip(jax.tree_util.tree_leaves(fixed),jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(a,b)
    assert np.max(np.abs(np.asarray(fitted.log_channel_scale-initial.log_channel_scale)))<=np.log(4.)
