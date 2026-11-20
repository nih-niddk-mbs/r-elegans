"""Simulated single-neuron stimulation responses of the conductance network; no new parameters.

A stimulus is an added membrane conductance (reversal ``reversal_mV``) on one neuron for ``stim_s`` seconds.
The response of every neuron is the calcium-indicator change of the stimulated run minus an identical unstimulated
run from the same settled state. The model is deterministic, so no noise model is involved; phase shifts of
intrinsic oscillators contribute to the difference and are averaged only by the response window.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .conductance_network import MembraneConductanceInput, advance_network, initial_network_state


def settle(params, fixed, config, *, warmup_s=60., dt_s=.1, substeps=20):
    """Free-run state after ``warmup_s`` from the fixed initial state."""
    frames = int(round(warmup_s / dt_s))
    step = lambda state, _: (advance_network(state, params, fixed, dt_s, substeps, config), None)
    return jax.jit(lambda state: jax.lax.scan(step, state, None, length=frames)[0])(initial_network_state(fixed, config))


def stimulation_windows(state0, params, fixed, config, targets, conductance_nS, *, reversal_mV=0., stim_s=.5,
                        record_s=12., window_s=(1.5, 11.5), dt_s=.1, substeps=20, batch_size=32):
    """Window-mean (stimulated - control) indicator for each (target index, conductance) pair: [K, neurons]."""
    frames, stim_frames = int(round(record_s / dt_s)), int(round(stim_s / dt_s))
    lo, hi = (int(round(w / dt_s)) for w in window_s)
    if not 0 <= lo < hi <= frames:
        raise ValueError('Response window must lie inside the recording')
    n = state0.voltage_mV.shape[0]

    def run(conductance):
        def step(state, k):
            on = (k < stim_frames).astype(conductance.dtype)
            membrane = MembraneConductanceInput(conductance * on, conductance * on * reversal_mV)
            state = advance_network(state, params, fixed, dt_s, substeps, config, membrane_input=membrane)
            return state, state.indicator
        return jax.lax.scan(step, state0, jnp.arange(frames))[1]

    control = jax.jit(run)(jnp.zeros(n))

    def response(arguments):
        target, g = arguments
        return (run(jnp.zeros(n).at[target].set(g)) - control)[lo:hi].mean(axis=0)

    mapped = jax.jit(lambda t, g: jax.lax.map(response, (t, g), batch_size=batch_size))
    return np.asarray(mapped(jnp.asarray(targets, jnp.int32), jnp.asarray(conductance_nS, float))), np.asarray(control)


def response_matrix(state0, params, fixed, config, targets, conductance_nS, **kwargs):
    """[targets, neurons] responses at one stimulus conductance."""
    values, _ = stimulation_windows(state0, params, fixed, config, np.asarray(targets), np.full(len(targets), conductance_nS), **kwargs)
    return values
