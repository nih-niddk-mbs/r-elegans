"""Forward-only replay of E104/E105 selected models; no optimization.

Peptide levels update on the original outer clock, including during dense
rhythm recording. Refining neural steps does not silently change that clock.
"""
import jax
import jax.numpy as jnp
import numpy as np
from .conductance_network import MembraneConductanceInput,advance_network,initial_network_state
from .extrasynaptic import peptide_membrane_input,peptide_level_step


def atlas_forward(params,fixed,config,conductances,substeps,*,peptide=None,dt=.1,settle_seconds=60.,frames=120,stim_seconds=.5,window=(1.5,11.5)):
    lo,hi=(int(round(x/dt)) for x in window);stim_frames=int(round(stim_seconds/dt));n=params.raw_chemical.shape[0]
    if substeps<1 or not 0<=lo<hi<=frames or settle_seconds<0:raise ValueError('Invalid replay protocol')
    if peptide is None:
        @jax.jit
        def settle(p,f):
            def step(state,_):return advance_network(state,p,f,dt,substeps,config),None
            return jax.lax.scan(step,initial_network_state(f,config),None,length=int(round(settle_seconds/dt)))[0]
        def run(p,f,state0,g):
            def step(state,k):
                on=(k<stim_frames).astype(g.dtype)
                state=advance_network(state,p,f,dt,substeps,config,membrane_input=MembraneConductanceInput(g*on,g*on*0.))
                return state,state.indicator
            return jax.lax.scan(jax.checkpoint(step),state0,jnp.arange(frames))[1]
        @jax.jit
        def responses(p,f,state,gs):
            control=run(p,f,state,jnp.zeros(n,conductances.dtype))
            stimulated=jax.vmap(lambda g:run(p,f,state,g))(gs)
            return (stimulated-control[None])[:,lo:hi].mean(axis=1)
        state=settle(params,fixed)
        return np.asarray(responses(params,fixed,state,conductances))
    exc,inh,tau=peptide;unit=config.synaptic_unit_nS
    @jax.jit
    def settle(p,exc,inh,tau):
        def step(carry,_):
            state,level=carry;membrane=peptide_membrane_input(level,exc,inh,unit)
            state=advance_network(state,p,fixed,dt,substeps,config,membrane_input=membrane)
            return (state,peptide_level_step(level,state.voltage_mV,fixed,dt,tau)),None
        return jax.lax.scan(step,(initial_network_state(fixed,config),jnp.zeros(n,conductances.dtype)),None,length=int(round(settle_seconds/dt)))[0]
    def run(p,exc,inh,tau,state0,level0,g):
        def step(carry,k):
            state,level=carry;on=(k<stim_frames).astype(g.dtype);pep=peptide_membrane_input(level,exc,inh,unit)
            membrane=MembraneConductanceInput(g*on+pep.conductance_nS,g*on*0.+pep.weighted_reversal_pA)
            state=advance_network(state,p,fixed,dt,substeps,config,membrane_input=membrane)
            return (state,peptide_level_step(level,state.voltage_mV,fixed,dt,tau)),state.indicator
        return jax.lax.scan(jax.checkpoint(step),(state0,level0),jnp.arange(frames))[1]
    @jax.jit
    def responses(p,exc,inh,tau,state,level,gs):
        control=run(p,exc,inh,tau,state,level,jnp.zeros(n,conductances.dtype))
        stimulated=jax.vmap(lambda g:run(p,exc,inh,tau,state,level,g))(gs)
        return (stimulated-control[None])[:,lo:hi].mean(axis=1)
    state,level=settle(params,exc,inh,tau)
    return np.asarray(responses(params,exc,inh,tau,state,level,conductances))


def rhythm_forward(params,fixed,config,*,peptide=None,seconds=40.,sample_dt=.002,max_step_ms=1.,peptide_dt=.1,warmup_seconds=0.):
    outer=peptide_dt if peptide is not None else sample_dt
    inner=int(round(outer/sample_dt));frames=int(round(seconds/outer));substeps=int(np.ceil(sample_dt*1000/max_step_ms))
    if warmup_seconds<0 or min(inner,frames,substeps)<1 or not np.isclose(inner*sample_dt,outer) or not np.isclose(frames*outer,seconds) or not np.isclose(round(warmup_seconds/outer)*outer,warmup_seconds):raise ValueError('Rhythm clocks must divide exactly')
    n=params.raw_chemical.shape[0]
    @jax.jit
    def run(p):
        def step(carry,_):
            state,level=carry
            membrane=None if peptide is None else peptide_membrane_input(level,peptide[0],peptide[1],config.synaptic_unit_nS)
            def neural(s,_):
                s=advance_network(s,p,fixed,sample_dt,substeps,config,membrane_input=membrane)
                return s,s.voltage_mV
            state,v=jax.lax.scan(neural,state,None,length=inner)
            if peptide is not None:level=peptide_level_step(level,state.voltage_mV,fixed,outer,peptide[2])
            return (state,level),v
        start=(initial_network_state(fixed,config),jnp.zeros(n,params.log_channel_scale.dtype))
        start=jax.lax.scan(lambda carry,k:(step(carry,k)[0],None),start,None,length=int(round(warmup_seconds/outer)))[0]
        return jax.lax.scan(step,start,None,length=frames)[1]
    return np.asarray(run(params)).reshape(-1,n)
