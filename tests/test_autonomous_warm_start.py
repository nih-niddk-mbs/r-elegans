import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.autonomous_warm_start import read_warm_start,copy_warm_start
from scripts.experiment_records import file_sha256
from scripts.run_embodied_diagnostics import select_input_bundle
from r_elegans.brain.conductance_network import load_network,save_network
from r_elegans.brain.joint_autonomous_fit import joint_fit_reference,apply_joint_offsets,joint_proposal


def parent_fixture(root):
    root.mkdir();shutil.copytree(select_input_bundle().parent/'autonomous_wiggle_v1',root/'source_inputs')
    initial,fixed,config=load_network(root/'source_inputs/network.npz');fixed,spec=joint_fit_reference(fixed)
    offsets=np.zeros(spec.size);offsets[0]=np.log(16.);offsets[spec.counts['channel_densities']]=.2
    chosen=apply_joint_offsets(initial,jnp.asarray(offsets),spec)
    save_network(root/'network_best.npz',chosen,fixed,config)
    np.savez_compressed(root/'allowed_parameters.npz',channel_mask=spec.channels,chemical_mask=spec.chemical,gap_mask=spec.gap)
    (root/'summary.json').write_text(json.dumps(dict(status='completed',objective='body-wave',fit_scope='neurons-connections',
        total_trainable=spec.size,iterations=3,selected_iteration=1,network_best_sha256=file_sha256(root/'network_best.npz'))))
    (root/'fit_policy.json').write_text(json.dumps(dict(fit_scope='neurons-connections',maximum_density_factor=16.)))
    (root/'search_state.json').write_text(json.dumps(dict(best=offsets.tolist(),completed_iterations=3,selected_iteration=1)))
    (root/'best.json').write_text(json.dumps(dict(iteration=1,offsets=offsets.tolist())))
    (root/'history.jsonl').write_text('test fixture history\n')
    return initial,fixed,spec,offsets


def test_warm_fork_keeps_original_anchor_and_copies_parent_history(tmp_path):
    root=tmp_path/'parent';initial,fixed,spec,offsets=parent_fixture(root)
    data=copy_warm_start(root,tmp_path/'copy')
    assert data['maximum_factor']==16. and data['state']['best']==offsets.tolist()
    assert data['parameters_at_bound']==1
    assert (tmp_path/'copy/history.jsonl').read_text()=='test fixture history\n'
    restored=apply_joint_offsets(initial,jnp.asarray(data['state']['best']),spec)
    saved,_,_=load_network(tmp_path/'copy/network_best.npz')
    for a,b in zip(jax.tree_util.tree_leaves(restored),jax.tree_util.tree_leaves(saved)):np.testing.assert_array_equal(a,b)
    proposal=joint_proposal(offsets,spec,np.random.default_rng(4),1.,16.)
    assert np.max(np.abs(proposal))<=np.log(16.)  # Not a new factor16 around the selected model.
    assert data['anchor_manifest']['sha256']['network.npz']==file_sha256(root/'source_inputs/network.npz')
    assert file_sha256(root/'source_inputs/network.npz')!=file_sha256(root/'network_best.npz')


def test_warm_source_rejects_mutation_live_runs_and_out_of_bounds_offsets(tmp_path):
    root=tmp_path/'parent';parent_fixture(root)
    summary=json.loads((root/'summary.json').read_text());summary['status']='running'
    (root/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='completed'):read_warm_start(root)
    summary['status']='completed';(root/'summary.json').write_text(json.dumps(summary))
    state=json.loads((root/'search_state.json').read_text());state['best'][0]=np.log(256.)
    (root/'search_state.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='bounds'):read_warm_start(root)
