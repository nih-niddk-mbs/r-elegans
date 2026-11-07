"""Conservative fitting policy: shared physiology fixed, explicit unknowns fit.

Fixed model estimates are not promoted to experimental facts. LEAK is the
conductance-scale anchor. RMD profiles (including the exploratory oscillating
profile) are held as hypotheses. Unobserved cells retain their channel priors;
unmeasured strengths on existing anatomical edges remain eligible to fit.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

from .conductance_network import ConductanceNetworkParams
from .single_compartment import CHANNEL_INDEX,CHANNEL_NAMES


class UnknownMasks(NamedTuple):
    channels: jax.Array
    chemical: jax.Array
    gap: jax.Array


def conservative_fit_constraints(fixed,neuron_ids,observed_names,extra=None):
    ids=tuple(neuron_ids)
    if set(observed_names)-set(ids):raise ValueError('Unknown observed neuron')
    masks=np.zeros((len(ids),len(CHANNEL_NAMES)),dtype=bool)
    reasons=[]
    for i,name in enumerate(ids):
        if name.startswith('RMD'):
            reason='Existing class profile held fixed; active high-CCA1 profile is a hypothesis, not measured truth'
        elif name not in observed_names:
            reason='Unobserved channel prior held fixed in this recording'
        else:
            reason='Unknown channel densities eligible; LEAK fixed as model conductance-scale anchor'
            masks[i]=True
            masks[i,CHANNEL_INDEX['LEAK']]=False
        reasons.append(dict(neuron=name,reason=reason))
    chemical=np.asarray(fixed.chemical_mask,dtype=bool).copy()
    gap=np.asarray(fixed.gap_mask,dtype=bool).copy()
    extra=extra or {}
    if set(extra)-{'frozen_channels','frozen_edges'}:raise ValueError('Unrecognized constraint section')
    for item in extra.get('frozen_channels',[]):
        if not item.get('source') or not item.get('reason'):raise ValueError('Frozen channel requires source and reason')
        if item.get('neuron') not in ids or item.get('channel') not in CHANNEL_INDEX:
            raise ValueError('Unknown constrained cell/channel')
        masks[ids.index(item['neuron']),CHANNEL_INDEX[item['channel']]]=False
    for item in extra.get('frozen_edges',[]):
        if not item.get('source') or not item.get('reason'):raise ValueError('Frozen edge requires source and reason')
        if item.get('post') not in ids or item.get('pre') not in ids or item.get('kind') not in ('chemical','gap'):
            raise ValueError('Unknown constrained edge')
        post,pre=ids.index(item['post']),ids.index(item['pre'])
        if item['kind']=='chemical':
            if not fixed.chemical_mask[post,pre]:raise ValueError('Constrained edge absent from anatomy')
            chemical[post,pre]=False
        else:
            if not fixed.gap_mask[post,pre]:raise ValueError('Constrained edge absent from anatomy')
            gap[post,pre]=gap[pre,post]=False
    policy=dict(version='unknown-only-v1',
        fixed_shared_properties=['channel gating equations/rate constants','ion reversal potentials',
            'calcium handling constants','capacitance','leak reversal','LEAK reference conductance',
            'anatomical topology','release thresholds/slopes/reversal priors','NMJ/muscle/body readout','indicator constants/readout'],
        fixed_is_not_always_measured='Transferred estimates and exploratory profiles stay labeled model assumptions',
        fixed_categories=dict(topology='anatomical data',kinetics='published-model shared laws',
            capacitance_reversals_calcium='current physiological estimates held fixed',
            leak_conductance='model scale anchor held fixed',
            RMD_profile='class prior or exploratory hypothesis held fixed',
            unobserved_channels='prior held due to recording scope'),
        trainable=dict(channel_conductances=int(masks.sum()),chemical_strengths=int(chemical.sum()),
            independent_gap_strengths=int(np.triu(gap,1).sum())),
        cell_policy=reasons,additional_constraints=extra,
        no_universal_cell_assumption='Capacitance/leak may differ across cells; holding current estimates fixed is a modeling choice',
        anatomy_counts_are_not_conductance_measurements=True)
    return UnknownMasks(jnp.asarray(masks),jnp.asarray(chemical),jnp.asarray(gap)),policy


def project_unknowns(params,initial,masks):
    return ConductanceNetworkParams(
        jnp.where(masks.channels,params.log_channel_scale,initial.log_channel_scale),
        jnp.where(masks.chemical,params.raw_chemical,initial.raw_chemical),
        jnp.where(masks.gap,params.raw_gap,initial.raw_gap))


def frozen_parameter_changes(params,initial,masks):
    fields=('log_channel_scale','raw_chemical','raw_gap')
    return {name:int(np.count_nonzero(np.asarray(a)[~np.asarray(m)]!=np.asarray(b)[~np.asarray(m)]))
            for name,a,b,m in zip(fields,params,initial,masks)}
