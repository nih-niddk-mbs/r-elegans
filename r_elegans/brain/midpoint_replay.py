"""Opt-in solver replay and multi-cycle full-state sampling; old engines unchanged."""
import numpy as np
from .response_clock_probe import clock_counts


def make_replay(config, observed, dt, step_ms, *, seconds=12., stimulus_seconds=.5, window=(1.5,11.5), method='midpoint', trace_mode='full'):
    import jax
    import jax.numpy as jnp
    from .conductance_network import advance_network,MembraneConductanceInput
    from .extrasynaptic import peptide_membrane_input,peptide_level_step
    from .peptide_state_replay import FullState
    from .midpoint_network import advance_midpoint
    if method not in ('legacy','midpoint'):raise ValueError('Unknown integration method')
    if trace_mode not in ('full','shams'):raise ValueError('Unknown trace mode')
    advance = advance_network if method=='legacy' else advance_midpoint
    substeps,block,frames=clock_counts(dt,step_ms,seconds)
    ticks=[round(t/dt) for t in (stimulus_seconds,*window)]
    if not all(np.isclose(k*dt,t) for k,t in zip(ticks,(stimulus_seconds,*window))) or not 0<=ticks[1]<ticks[2]<=round(seconds/dt):
        raise ValueError('Stimulus/observation boundaries must align with clock')
    index=jnp.asarray(observed)
    @jax.jit
    def replay(p,f,pp,exc,inh,starts,stimuli):
        def trial(start,g):
            def export(carry,frame):
                state,total=carry
                def inner(j,carry):
                    state,total,lo,hi=carry;k=frame*block+j
                    extra=peptide_membrane_input(state.peptide,exc,inh,config.synaptic_unit_nS)
                    neural=advance(state.neural,p,f,dt,substeps,config,
                        membrane_input=MembraneConductanceInput(extra.conductance_nS+g*(k<ticks[0]),extra.weighted_reversal_pA))
                    state=FullState(neural,peptide_level_step(state.peptide,neural.voltage_mV,f,dt,pp.log_tau_s))
                    total=total+jnp.where((k>=ticks[1])&(k<ticks[2]),neural.indicator,0.)
                    return state,total,jnp.minimum(lo,jnp.min(neural.voltage_mV)),jnp.maximum(hi,jnp.max(neural.voltage_mV))
                state,total,lo,hi=jax.lax.fori_loop(0,block,inner,(state,total,jnp.array(jnp.inf),jnp.array(-jnp.inf)))
                trace=(state.neural.voltage_mV[index],state.neural.calcium_uM[index],state.neural.indicator[index],state.peptide[index],lo,hi)
                return (state,total),trace
            (final,total),traces=jax.lax.scan(export,(start,jnp.zeros_like(g)),jnp.arange(frames))
            return total/(ticks[2]-ticks[1]),final,traces
        means,final,traces=jax.vmap(lambda s:jax.vmap(lambda g:trial(s,g))(stimuli))(starts)
        if trace_mode=='shams':traces=tuple(a[:,-2:] for a in traces[:4])+traces[4:]
        return means,final,traces
    return replay



def cycle_phases(voltage,dt,cycles=4):
    from scipy.signal import find_peaks
    v=np.asarray(voltage)
    peaks=find_peaks(v,prominence=1.,distance=max(1,round(.02/dt)))[0]
    if cycles<2 or len(peaks)-1<cycles:raise ValueError('Insufficient complete cycles for prospective sampling')
    selected=np.rint(np.linspace(0,len(peaks)-2,cycles)).astype(int)
    indices=np.concatenate([np.rint(peaks[i]+np.arange(4)*(peaks[i+1]-peaks[i])/4).astype(int) for i in selected])
    if len(set(indices.tolist()))!=4*cycles:raise ValueError('Unresolved sampled phases')
    return indices,dict(record_dt_s=dt,complete_cycles=len(peaks)-1,selected_cycles=selected.tolist(),indices=indices.tolist(),cycle_peak_indices=[[int(peaks[i]),int(peaks[i+1])] for i in selected])


def make_anchors(config,rmd_index,dt,step_ms,*,warmup=60.,record=4.,cycles=4):
    import jax
    import jax.numpy as jnp
    from .conductance_network import initial_network_state,MembraneConductanceInput
    from .extrasynaptic import peptide_membrane_input,peptide_level_step
    from .peptide_state_replay import FullState
    from .midpoint_network import advance_midpoint
    substeps,_,_=clock_counts(dt,step_ms,record)
    @jax.jit
    def settle(p,f,pp,exc,inh):
        def step(state,_):
            extra=peptide_membrane_input(state.peptide,exc,inh,config.synaptic_unit_nS)
            neural=advance_midpoint(state.neural,p,f,dt,substeps,config,membrane_input=MembraneConductanceInput(extra.conductance_nS,extra.weighted_reversal_pA))
            state=FullState(neural,peptide_level_step(state.peptide,neural.voltage_mV,f,dt,pp.log_tau_s))
            return state,state
        start=FullState(initial_network_state(f,config),jnp.zeros_like(f.initial_voltage_mV))
        start=jax.lax.scan(lambda s,k:(step(s,k)[0],None),start,None,length=round(warmup/dt))[0]
        return jax.lax.scan(step,start,None,length=round(record/dt))[1]
    def anchors(p,f,pp,exc,inh):
        states=settle(p,f,pp,exc,inh)
        reference=np.asarray(states.neural.voltage_mV[:,rmd_index])
        indices,metadata=cycle_phases(reference,dt,cycles)
        metadata['reference_voltage_mV']=reference.tolist()
        return jax.tree.map(lambda a:a[indices],states),metadata
    return anchors
