"""Nested full-state cycle ensembles extending an existing midpoint trajectory."""
import numpy as np


def spread_order(size):
    if size<1:raise ValueError('Positive bin count required')
    selected=[0];available=set(range(1,size))
    while available:
        pick=max(sorted(available),key=lambda i:min(abs(i-j) for j in selected))
        selected.append(pick);available.remove(pick)
    return selected


def extension_phases(voltage,dt,seconds=32.,new_cycles=28):
    from scipy.signal import find_peaks
    v=np.asarray(voltage)
    if len(v)!=round(seconds/dt) or not np.isfinite(v).all():raise ValueError('Finite full extension trace required')
    peaks=find_peaks(v,prominence=1.,distance=max(1,round(.02/dt)))[0]
    if len(peaks)-1<new_cycles:raise ValueError('Insufficient complete extension cycles')
    bins=spread_order(32)[:new_cycles];centres=(np.asarray(bins)+.5)*seconds/32
    midpoints=(peaks[:-1]+peaks[1:])/2*dt
    cycles=[int(np.argmin(abs(midpoints-t))) for t in centres]
    if len(set(cycles))!=new_cycles:raise ValueError('Requested cycle-time samples are not distinct')
    indices=np.concatenate([np.rint(peaks[i]+np.arange(4)*(peaks[i+1]-peaks[i])/4).astype(int) for i in cycles])
    if len(set(indices.tolist()))!=4*new_cycles:raise ValueError('Unresolved extension phases')
    return indices,dict(seconds=seconds,dt=dt,new_cycles=new_cycles,bins=bins,requested_centres_s=centres.tolist(),selected_cycles=cycles,
        indices=indices.tolist(),cycle_mean_time_s=((indices.reshape(-1,4)+1)*dt).mean(1).tolist(),reference_voltage_mV=v.tolist())


def make_extension(config,rmd_index,dt,step_ms,seconds=32.):
    import jax
    import jax.numpy as jnp
    from .midpoint_network import advance_midpoint
    from .conductance_network import MembraneConductanceInput
    from .extrasynaptic import peptide_membrane_input,peptide_level_step
    from .peptide_state_replay import FullState
    n=round(dt*1000/step_ms)
    if n<1 or not np.isclose(n*step_ms,dt*1000):raise ValueError('Aligned neural/outer clocks required')
    @jax.jit
    def run(p,f,pp,exc,inh,start):
        def step(state,_):
            extra=peptide_membrane_input(state.peptide,exc,inh,config.synaptic_unit_nS)
            neural=advance_midpoint(state.neural,p,f,dt,n,config,membrane_input=MembraneConductanceInput(extra.conductance_nS,extra.weighted_reversal_pA))
            state=FullState(neural,peptide_level_step(state.peptide,neural.voltage_mV,f,dt,pp.log_tau_s))
            return state,state
        return jax.lax.scan(step,start,None,length=round(seconds/dt))
    def extend(p,f,pp,exc,inh,start):
        end,states=run(p,f,pp,exc,inh,start)
        reference=np.asarray(states.neural.voltage_mV[:,rmd_index]);indices,meta=extension_phases(reference,dt,seconds)
        finite=all(bool(jnp.isfinite(v).all()) for v in jax.tree.leaves(states))
        meta.update(finite=finite,voltage_min_mV=float(states.neural.voltage_mV.min()),voltage_max_mV=float(states.neural.voltage_mV.max()))
        return jax.tree.map(lambda a:a[indices],states),end,meta
    return extend


def ensemble_metrics(response,cycle_times,previous=None,tolerance=None):
    y=np.asarray(response);n=y.shape[0]//4
    if y.ndim!=3 or n<4 or y.shape[0]!=4*n or not np.isfinite(y).all():raise ValueError('Finite complete cycle/phase array required')
    cycles=y.reshape(n,4,*y.shape[1:]).mean(1);mean=cycles.mean(0)
    tol=float(1e-7+.1*abs(mean).max()) if tolerance is None else float(tolerance)
    if not np.isfinite(tol) or tol<=0:raise ValueError('Positive finite stability tolerance required')
    loo=float(max(abs((cycles.sum(0)-c)/(n-1)-mean).max() for c in cycles))
    order=np.argsort(cycle_times)
    if len(order)!=n:raise ValueError('Cycle time count mismatch')
    block=float(abs(cycles[order[:n//2]].mean(0)-cycles[order[n//2:]].mean(0)).max())
    prefix=None if previous is None else float(abs(mean-np.asarray(previous).mean(0)).max())
    return dict(cycles=n,tolerance=tol,leave_one_cycle_out_max_difference=loo,temporal_half_max_difference=block,prefix_mean_max_difference=prefix,
        leave_one_out_passed=loo<=tol,temporal_halves_passed=block<=tol,prefix_passed=prefix is not None and prefix<=tol,
        passed=bool(loo<=tol and block<=tol and prefix is not None and prefix<=tol),
        scope='Prospective deterministic panel stability; temporal/prefix checks prevent automatic pass merely from1/N leave-out shrinkage; not a confidence interval')
