"""Unknown densities and existing recurrent strengths for autonomous behavior.

No anatomy, intrinsic kinetic constants, release properties, injected currents
or muscle/body parameters are optimized. Electrical pairs share one multiplier.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .single_compartment import CHANNEL_INDEX


@dataclass
class JointFitSpec:
    channels: np.ndarray
    chemical: np.ndarray
    gap: np.ndarray

    @property
    def counts(self):
        return dict(channel_densities=int(self.channels.sum()),chemical_strengths=int(self.chemical.sum()),
                    electrical_pairs=int(np.triu(self.gap,1).sum()))

    @property
    def size(self):return sum(self.counts.values())


def joint_fit_reference(fixed):
    channels=np.asarray(fixed.cells.conductances_nS)>0
    channels[:,CHANNEL_INDEX['LEAK']]=False
    chemical=np.asarray(fixed.chemical_mask,dtype=bool)
    gap=np.asarray(fixed.gap_mask,dtype=bool)
    if not np.array_equal(gap,gap.T) or np.diag(gap).any():raise ValueError('Symmetric nonself electrical topology required')
    return fixed._replace(bias_pA=jnp.zeros_like(fixed.bias_pA)),JointFitSpec(channels,chemical,gap)


def _scaled_raw(raw, offset):
    # Stable inverse softplus. Multipliers apply to effective conductance, not
    # arbitrary raw coordinates. Preserve exact identity on untouched values.
    value=jax.nn.softplus(raw)*jnp.exp(offset)
    transformed=value+jnp.log(-jnp.expm1(-value))
    return jnp.where(offset==0,raw,transformed)


def apply_joint_offsets(initial, offsets, spec):
    a,b,c=spec.counts.values()
    if offsets.shape!=(a+b+c,):raise ValueError('Wrong joint-fit vector shape')
    ci=np.nonzero(spec.channels);ei=np.nonzero(spec.chemical);gi=np.nonzero(np.triu(spec.gap,1))
    channels=initial.log_channel_scale.at[ci].add(offsets[:a])
    chemical=initial.raw_chemical.at[ei].set(_scaled_raw(initial.raw_chemical[ei],offsets[a:a+b]))
    reverse=(gi[1],gi[0]);raw=.5*(initial.raw_gap[gi]+initial.raw_gap[reverse])
    delta=_scaled_raw(raw,offsets[a+b:])-raw
    # Preserve any inert antisymmetric raw component; the actual equations use
    # softplus((raw_ij+raw_ji)/2), so conductance stays symmetric exactly.
    gap=initial.raw_gap.at[gi].add(delta).at[reverse].add(delta)
    return initial._replace(log_channel_scale=channels,raw_chemical=chemical,raw_gap=gap)


def joint_proposal(best,spec,rng,sigma,maximum_factor=4.,focus_cells=None):
    if not np.isfinite(maximum_factor) or maximum_factor<=1:raise ValueError('Factor must exceed one')
    if sigma<=0 or not np.isfinite(sigma):raise ValueError('Positive finite sigma required')
    a,b,c=spec.counts.values();noise=np.zeros(spec.size)
    ci=np.nonzero(spec.channels);ei=np.nonzero(spec.chemical);gi=np.nonzero(np.triu(spec.gap,1))
    mode=int(rng.integers(4))
    if mode==0:
        # Densities and incoming/outgoing edges of a few cells, individually free.
        pool=np.arange(spec.channels.shape[0])
        if focus_cells is not None and len(focus_cells) and rng.random()<.5:pool=np.asarray(focus_cells)
        cells=rng.choice(pool,size=min(8,len(pool)),replace=False)
        selected=np.concatenate((np.isin(ci[0],cells),np.isin(ei[0],cells)|np.isin(ei[1],cells),
            np.isin(gi[0],cells)|np.isin(gi[1],cells)))
        noise[selected]=rng.normal(size=int(selected.sum()))
    elif mode==1:
        channel=int(rng.choice(np.unique(ci[1])))
        selected=ci[1]==channel
        noise[np.flatnonzero(selected)]=rng.normal()  # Coherent direction; values remain free.
    elif mode==2:
        kind=int(rng.integers(2));start,end=(a,a+b) if kind==0 else (a+b,a+b+c)
        pool=np.arange(start,end)
        if focus_cells is not None and len(focus_cells) and rng.random()<.5:
            relevant=np.isin(ei[0],focus_cells) if kind==0 else (np.isin(gi[0],focus_cells)|np.isin(gi[1],focus_cells))
            if relevant.any():pool=pool[relevant]
        selected=rng.choice(pool,size=min(128,len(pool)),replace=False)
        noise[selected]=rng.normal(size=len(selected))
    else:
        selected=rng.choice(spec.size,size=min(128,spec.size),replace=False)
        noise[selected]=rng.normal(size=len(selected))
    return np.clip(best+sigma*noise,-np.log(maximum_factor),np.log(maximum_factor))


def frozen_joint_fields_unchanged(candidate,initial,spec):
    return (np.array_equal(np.asarray(candidate.log_channel_scale)[~spec.channels],np.asarray(initial.log_channel_scale)[~spec.channels])
        and np.array_equal(np.asarray(candidate.raw_chemical)[~spec.chemical],np.asarray(initial.raw_chemical)[~spec.chemical])
        and np.array_equal(np.asarray(candidate.raw_gap)[~spec.gap],np.asarray(initial.raw_gap)[~spec.gap]))
