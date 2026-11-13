"""Extend the exact E067 parent setup with unknown nonleak densities."""
import numpy as np


def setup(inputs):
    from scripts.connection_amplitude_setup import setup as parent_setup
    from r_elegans.brain.joint_autonomous_fit import joint_fit_reference,apply_joint_offsets,frozen_joint_fields_unchanged
    d=parent_setup(inputs)
    source_factory=d['factory']
    def factory(*args,**kwargs):return source_factory(*args,**kwargs,medium='liquid')
    _,spec=joint_fit_reference(d['fixed'])
    count=int(spec.channels.sum())
    origin=np.r_[np.zeros(count),d['origin']]
    lower=np.r_[np.full(count,-np.log(2.)),d['lower']]
    upper=np.r_[np.full(count,np.log(2.)),d['upper']]
    original_base=d['base']
    def decode(x):
        if x.shape!=origin.shape or not np.isfinite(x).all() or np.any(x<lower-1e-12) or np.any(x>upper+1e-12):
            raise ValueError('Neural timing coordinates exceed original-anchor bounds')
        params=apply_joint_offsets(original_base,np.asarray(x)-origin,spec)
        if not frozen_joint_fields_unchanged(params,original_base,spec):raise ValueError('Frozen fields changed')
        return params
    d.update(spec=spec,origin=origin,lower=lower,upper=upper,decode=decode,
             source_factory=source_factory,factory=factory,training_medium='liquid')
    if 'continuation_parent.npz' in d['manifest']['sha256']:
        from pathlib import Path
        from r_elegans.brain.conductance_network import load_network
        from r_elegans.rhythm_comparison import assert_fixed_equal
        root=Path(inputs)
        parent,fixed,config=load_network(root/'continuation_parent.npz')
        assert_fixed_equal(d['fixed'],fixed,d['config'],config)
        with np.load(root/'parent_best_state.npz',allow_pickle=False) as z:warm=z['offsets']
        rebuilt=decode(warm)
        if any(not np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(rebuilt,parent)):
            raise ValueError('Selected E070 coordinates do not reconstruct the exact parent')
        d.update(base=parent,origin=warm,continuation=True)
    return d
