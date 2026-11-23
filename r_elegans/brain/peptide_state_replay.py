"""Own-state atlas pilot with explicit peptide memory; separate from E120 solver.

Peptide conductances are held over each outer interval, then release memory is
updated. Both clocks must be refined for convergence; no spatial diffusion model.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from .conductance_network import advance_network, initial_network_state, MembraneConductanceInput
from .extrasynaptic import peptide_level_step, peptide_membrane_input
from .common_state_replay import phase_indices


class FullState(NamedTuple):
    neural: object
    peptide: object


def make_engine(config, rmd_indices, *, dt=.001, step_ms=.03125, warmup=60., seconds=12.):
    substeps = round(dt * 1000 / step_ms)
    if substeps < 1 or not np.isclose(substeps * step_ms, dt * 1000):
        raise ValueError('Neural step must divide outer clock')
    if seconds < 11.5 or warmup < 0:
        raise ValueError('Atlas observation window requires >=11.5 seconds')

    def advance(state, p, f, exc, inh, log_tau, stimulus):
        extra = peptide_membrane_input(state.peptide, exc, inh, config.synaptic_unit_nS)
        neural = advance_network(state.neural, p, f, dt, substeps, config,
            membrane_input=MembraneConductanceInput(extra.conductance_nS + stimulus,
                                                    extra.weighted_reversal_pA))
        return FullState(neural, peptide_level_step(state.peptide, neural.voltage_mV, f, dt, log_tau))

    @jax.jit
    def settle(p, f, exc, inh, log_tau):
        initial = initial_network_state(f, config)
        start = FullState(initial, jnp.zeros_like(initial.voltage_mV))
        zero = jnp.zeros_like(initial.voltage_mV)
        def step(s, _):
            s = advance(s, p, f, exc, inh, log_tau, zero)
            return s, s
        start = jax.lax.scan(lambda s, k: (step(s, k)[0], None), start, None, length=round(warmup/dt))[0]
        return jax.lax.scan(step, start, None, length=round(1./dt))[1]

    @jax.jit
    def replay(p, f, exc, inh, log_tau, starts, gs):
        def trial(start, g):
            def step(carry, k):
                state, total = carry
                state = advance(state, p, f, exc, inh, log_tau, g * (k < round(.5/dt)))
                total = total + jnp.where((k >= round(1.5/dt)) & (k < round(11.5/dt)), state.neural.indicator, 0.)
                return (state, total), (state.neural.voltage_mV[jnp.asarray(rmd_indices)],
                    jnp.mean(state.peptide), jnp.mean(state.neural.indicator))
            (final, total), traces = jax.lax.scan(step, (start, jnp.zeros_like(g)), jnp.arange(round(seconds/dt)))
            return total / round(10./dt), final, traces
        return jax.vmap(lambda s: jax.vmap(lambda g: trial(s, g))(gs))(starts)

    def anchors(p, f, exc, inh, log_tau):
        states = jax.tree.map(np.asarray, settle(p, f, exc, inh, log_tau))
        ix, cycle = phase_indices(states.neural.voltage_mV[:, rmd_indices[0]], dt)
        return jax.tree.map(lambda x: jnp.asarray(x[ix]), states), dict(indices=ix.tolist(), cycle=cycle)
    return anchors, replay
