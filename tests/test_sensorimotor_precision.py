import json
import numpy as np
import pytest


def test_operational_reference_preserves_legacy_values_at_zero_delta(tmp_path):
    import jax
    from r_elegans.brain.conductance_network import load_network,network_synapses,save_network
    from r_elegans.brain.joint_autonomous_fit import joint_fit_reference,apply_joint_offsets,joint_proposal,frozen_joint_fields_unchanged
    from scripts.run_embodied_diagnostics import select_input_bundle
    from scripts.assess_sensorimotor_resolution import promote_floating_values
    initial,fixed,config=load_network(select_input_bundle().parent/'autonomous_wiggle_v1/network.npz')
    _,spec=joint_fit_reference(fixed)
    origin=np.linspace(-.6,.6,spec.size)
    stored=apply_joint_offsets(initial,origin,spec)
    with jax.experimental.enable_x64(True):
        base=promote_floating_values(stored)
        decoded=apply_joint_offsets(base,origin-origin,spec)
        for a,b in zip(jax.tree_util.tree_leaves(stored),jax.tree_util.tree_leaves(decoded)):
            np.testing.assert_array_equal(np.asarray(a).astype(np.float64),b)
        x=joint_proposal(origin,spec,np.random.default_rng(6200),.03,2.)
        assert np.max(np.abs(x))<=np.log(2.)
        changed=apply_joint_offsets(base,x-origin,spec)
        assert frozen_joint_fields_unchanged(changed,base,spec)
        _,gap=network_synapses(changed,fixed,config)
        np.testing.assert_array_equal(gap,gap.T)
        save_network(tmp_path/'parameter_reference.npz',base,promote_floating_values(fixed),config)
        save_network(tmp_path/'network_best.npz',changed,promote_floating_values(fixed),config)
        np.savez(tmp_path/'parameter_origin.npz',offsets=origin)
        (tmp_path/'configuration.json').write_text(json.dumps(dict(arguments=dict(precision='float64'))))
    from scripts.sensorimotor_parameters import load_saved_parameters
    restored,_,_=load_saved_parameters(tmp_path,initial,fixed,config,spec,x)
    for expected,actual in zip(jax.tree_util.tree_leaves(changed),jax.tree_util.tree_leaves(restored)):
        assert actual.dtype==np.float64
        np.testing.assert_array_equal(actual,expected)


def test_precision_reference_checks_model_and_migrated_response(tmp_path):
    from scripts.sensorimotor_parameters import verify_precision_baseline
    data=dict(precision='float64',checkpoint_sha256='source',cases=[dict(case='coarse',metrics=dict(cue_specific_peak_rad=2.8e-5))])
    path=tmp_path/'summary.json';path.write_text(json.dumps(data))
    assert verify_precision_baseline(tmp_path,'source',2.8e-5)==2.8e-5
    with pytest.raises(ValueError,match='model mismatch'):verify_precision_baseline(tmp_path,'different',2.8e-5)
    with pytest.raises(ValueError,match='changed low-dose'):verify_precision_baseline(tmp_path,'source',3e-5)
