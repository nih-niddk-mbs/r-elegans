"""Zero-input diagnostic isolation using the channel network's integration rule."""
import jax
import jax.numpy as jnp
import numpy as np
from .conductance_network import channel_open_fractions
from .single_compartment import SingleCompartmentState, initial_single_compartment_state, _gate_targets_and_taus, CHANNEL_INDEX


def isolated_recording(cells, initial_voltage, dt_s, frames, substeps):
    """Independent cells, no synapses/bias; same frozen-conductance update.

    ``cells.conductances_nS`` includes the candidate's density multipliers.
    This graph ablation is diagnostic, not a replacement runtime model.
    """
    state = jax.vmap(initial_single_compartment_state)(initial_voltage,cells)
    h = dt_s*1000/substeps
    reversal = jnp.stack([cells.calcium_reversal_mV if i in (6,7,8) else
        cells.sodium_reversal_mV if i==14 else cells.leak_reversal_mV if i==16 else
        cells.potassium_reversal_mV for i in range(17)],axis=1)
    def inner(s,_):
        conductance = cells.conductances_nS*channel_open_fractions(s.gates)
        total = conductance.sum(axis=1)
        equilibrium = (conductance*reversal).sum(axis=1)/jnp.maximum(total,1e-12)
        voltage = s.voltage_mV+(equilibrium-s.voltage_mV)*(-jnp.expm1(-h*total/cells.capacitance_pF))
        target,tau = jax.vmap(_gate_targets_and_taus)(s.voltage_mV,s.calcium_uM,s.gates)
        gates = s.gates+(target-s.gates)*(-jnp.expm1(-h/tau))
        current = (conductance[:,6:9]*(s.voltage_mV[:,None]-cells.calcium_reversal_mV[:,None])).sum(axis=1)
        influx = jnp.where(s.voltage_mV<=cells.calcium_reversal_mV,
            -cells.free_calcium_fraction*current*1e6/(2*96485*cells.cell_volume_um3),0.)
        target_ca = cells.calcium_equilibrium_uM+influx*cells.calcium_removal_ms
        calcium = s.calcium_uM+(target_ca-s.calcium_uM)*(-jnp.expm1(-h/cells.calcium_removal_ms))
        return SingleCompartmentState(voltage,gates,calcium),None
    def frame(s,_):
        s,_ = jax.lax.scan(inner,s,None,length=substeps)
        return s,(s.voltage_mV,s.calcium_uM)
    return jax.lax.scan(frame,state,None,length=frames)[1]


def density_profiles(current, eligible, maximum_factor, random_profiles, seed):
    """Absolute original-anchor offsets, not additional factors on the parent."""
    current=np.asarray(current,dtype=float);eligible=np.asarray(eligible,dtype=bool)
    if current.shape!=eligible.shape or current.ndim!=2 or current.shape[1]!=17:
        raise ValueError('Expected cell/channel offsets and masks')
    if (not np.isfinite(maximum_factor) or maximum_factor<=1 or random_profiles<0 or seed<0
            or not np.isfinite(current).all()):raise ValueError('Invalid bounds/count/seed')
    bound=np.log(maximum_factor)
    if np.any(np.abs(current[eligible])>bound+1e-8) or np.any(current[~eligible]!=0):
        raise ValueError('Incoming offsets violate original bounds/frozen terms')
    if eligible[:,CHANNEL_INDEX['LEAK']].any():raise ValueError('Leak must stay frozen')
    profiles=[('current',current.copy()),('anchor',np.zeros_like(current))]
    def add(name, value):
        profiles.append((name,np.where(eligible,np.clip(value,-bound,bound),0.)))
    add('all_high',np.full_like(current,bound));add('all_low',np.full_like(current,-bound))
    inward=[CHANNEL_INDEX[n] for n in ('EGL19','UNC2','CCA1','NCA')]
    inward_mask=np.zeros_like(eligible);inward_mask[:,inward]=True
    add('inward_high_outward_low',np.where(inward_mask,bound,-bound))
    add('inward_high_outward_current',np.where(inward_mask,bound,current))
    add('inward_low_outward_high',np.where(inward_mask,-bound,bound))
    for name in ('EGL19','UNC2','CCA1','NCA'):
        value=current.copy();value[:,CHANNEL_INDEX[name]]=bound
        add(name+'_high',value)
    rng=np.random.default_rng(seed)
    for i in range(random_profiles):add(f'random-{i}',rng.uniform(-bound,bound,size=current.shape))
    return profiles,rng.bit_generator.state
