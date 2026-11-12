"""Validate legacy or operational-reference parameter encodings without downcast."""
import json
from pathlib import Path
import numpy as np


def verify_precision_baseline(reference,checkpoint_sha,peak):
    """Check migration against the previously measured64 source, before fitting."""
    import math
    reference=Path(reference)
    summary=json.loads((reference/'summary.json').read_text())
    if summary.get('precision')!='float64' or summary['checkpoint_sha256']!=checkpoint_sha:
        raise ValueError('Precision baseline reference model mismatch')
    expected=next(r['metrics']['cue_specific_peak_rad'] for r in summary['cases'] if r['case']=='coarse')
    if not math.isclose(peak,expected,rel_tol=5e-8,abs_tol=1e-12):
        raise ValueError('Operational float64 migration changed low-dose response')
    return expected


def load_saved_parameters(root,initial,fixed,config,spec,offsets):
    import jax
    from r_elegans.brain.conductance_network import load_network
    from r_elegans.brain.joint_autonomous_fit import apply_joint_offsets
    from scripts.assess_sensorimotor_resolution import promote_floating_values
    root=Path(root);cfg=json.loads((root/'configuration.json').read_text())
    precision=cfg['arguments'].get('precision','float32')
    with jax.experimental.enable_x64(precision=='float64'):
        actual,saved_fixed,saved_config=load_network(root/'network_best.npz')
        if precision=='float64':
            base,base_fixed,base_config=load_network(root/'parameter_reference.npz')
            with np.load(root/'parameter_origin.npz') as z:origin=z['offsets'].copy()
            expected=apply_joint_offsets(base,offsets-origin,spec)
            expected_fixed=promote_floating_values(fixed)
            if base_config!=config or any(not np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(
                jax.tree_util.tree_leaves(base_fixed),jax.tree_util.tree_leaves(expected_fixed))):
                raise ValueError('Operational reference changed fixed fields')
        else:
            base=None;origin=None;expected=apply_joint_offsets(initial,offsets,spec);expected_fixed=fixed
        if saved_config!=config or any(not np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(
            jax.tree_util.tree_leaves((expected,expected_fixed)),jax.tree_util.tree_leaves((actual,saved_fixed)))):
            raise ValueError('Saved parameters, encoding and fixed fields disagree')
    return actual,base,origin
