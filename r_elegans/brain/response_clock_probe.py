"""Fixed-state clock isolation with compact, matched-time diagnostic traces.

Uses the existing neural and peptide update laws, without changing model weights.
"""
import numpy as np

CLOCKS = {
    'coarse': (.001, .03125),
    'neural-half': (.001, .015625),
    'outer-half': (.0005, .03125),
    'joint-half': (.0005, .015625),
    'joint-quarter': (.00025, .0078125),
    'joint-eighth': (.000125, .00390625),
}
CASES = [(name, 'common') for name in CLOCKS] + [(name, 'own') for name in ('coarse','joint-half','joint-quarter')]


def clock_counts(dt, step_ms, seconds=12., export_dt=.001):
    substeps=round(dt*1000/step_ms);block=round(export_dt/dt);frames=round(seconds/export_dt)
    if min(substeps,block,frames)<1 or not all((np.isclose(substeps*step_ms,dt*1000),np.isclose(block*dt,export_dt),np.isclose(frames*export_dt,seconds))):
        raise ValueError('Clocks must divide update, export and duration intervals')
    return substeps,block,frames


def make_replay(config, observed, dt, step_ms, *, seconds=12., stimulus_seconds=.5, window=(1.5,11.5)):
    import jax
    import jax.numpy as jnp
    from .conductance_network import advance_network,MembraneConductanceInput
    from .extrasynaptic import peptide_membrane_input,peptide_level_step
    from .peptide_state_replay import FullState
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
                    neural=advance_network(state.neural,p,f,dt,substeps,config,
                        membrane_input=MembraneConductanceInput(extra.conductance_nS+g*(k<ticks[0]),extra.weighted_reversal_pA))
                    state=FullState(neural,peptide_level_step(state.peptide,neural.voltage_mV,f,dt,pp.log_tau_s))
                    total=total+jnp.where((k>=ticks[1])&(k<ticks[2]),neural.indicator,0.)
                    return state,total,jnp.minimum(lo,jnp.min(neural.voltage_mV)),jnp.maximum(hi,jnp.max(neural.voltage_mV))
                state,total,lo,hi=jax.lax.fori_loop(0,block,inner,(state,total,jnp.array(jnp.inf),jnp.array(-jnp.inf)))
                trace=(state.neural.voltage_mV[index],state.neural.calcium_uM[index],state.neural.indicator[index],state.peptide[index],lo,hi)
                return (state,total),trace
            (final,total),traces=jax.lax.scan(export,(start,jnp.zeros_like(g)),jnp.arange(frames))
            return total/(ticks[2]-ticks[1]),final,traces
        return jax.vmap(lambda s:jax.vmap(lambda g:trial(s,g))(stimuli))(starts)
    return replay


def response_differences(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape or a.ndim!=3 or not np.isfinite([a,b]).all():raise ValueError('Matching finite phase/target/receiver arrays required')
    return dict(phasewise_max=float(abs(a-b).max()),phase_mean_max=float(abs(a.mean(0)-b.mean(0)).max()),
        per_target_phase_mean_max=abs(a.mean(0)-b.mean(0)).max(axis=1).tolist())
