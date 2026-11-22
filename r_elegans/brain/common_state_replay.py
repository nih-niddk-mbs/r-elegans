"""E108 forward-only trials from exact shared full states; no parameter fitting."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal import find_peaks

from .conductance_network import advance_network, initial_network_state, MembraneConductanceInput

FIELDS = ('voltage_mV', 'gates', 'calcium_uM', 'indicator')


def phase_indices(voltage, dt):
    peaks = find_peaks(np.asarray(voltage), prominence=1., distance=max(1, round(.02 / dt)))[0]
    if len(peaks) < 2:
        raise ValueError('No complete reference RMDL cycle; phase trials unavailable')
    first, last = map(int, peaks[:2])
    indices = np.rint(first + np.arange(4) * (last - first) / 4).astype(int)
    if len(set(indices.tolist())) != 4:
        raise ValueError('Reference cycle too short to resolve quarter phases')
    return indices, [first, last]


def collect_anchors(params, fixed, config, rmd_indices, *, warmup=60., seconds=1., dt=.001, step_ms=.03125):
    steps = int(round(dt * 1000 / step_ms))
    if not np.isclose(steps * step_ms, dt * 1000):
        raise ValueError('Reference neural step must divide recording interval')
    @jax.jit
    def run(p, f):
        def step(s, _):
            s = advance_network(s, p, f, dt, steps, config)
            return s, s
        start = initial_network_state(f, config)
        start = jax.lax.scan(lambda s, k: (step(s, k)[0], None), start, None, length=round(warmup / dt))[0]
        return jax.lax.scan(step, start, None, length=round(seconds / dt))[1]
    states = jax.tree.map(np.asarray, run(params, fixed))
    ix, cycle = phase_indices(states.voltage_mV[:, rmd_indices[0]], dt)
    anchors = jax.tree.map(lambda a: a[ix], states)
    return anchors, dict(indices=ix.tolist(), peak_indices=cycle, time_s=((ix + 1) * dt).tolist(),
                         reference_rmd_mV=states.voltage_mV[:, rmd_indices], sample_dt_s=dt,
                         reference_step_ms=step_ms, warmup_seconds=warmup)


def make_replay(config, rmd_indices, step_ms, *,
           dt=.001, seconds=12., pulse_seconds=.5, short_seconds=.02, window=(1.5, 11.5)):
    steps = int(round(dt * 1000 / step_ms))
    frames = round(seconds / dt)
    lo, hi = [round(t / dt) for t in window]
    short_frame = round(short_seconds / dt) - 1
    if steps < 1 or not np.isclose(steps * step_ms, dt * 1000) or not 0 <= short_frame < frames or not 0 <= lo < hi <= frames:
        raise ValueError('Invalid shared-state replay clocks')
    @jax.jit
    def batch(p, f, starts, gs):
        def trial(state0, g):
            def step(carry, k):
                state, short, total = carry
                on = (k < round(pulse_seconds / dt)).astype(g.dtype)
                state = advance_network(state, p, f, dt, steps, config,
                    membrane_input=MembraneConductanceInput(g * on, jnp.zeros_like(g)))
                short = jax.tree.map(lambda a, b: jnp.where(k == short_frame, a, b), state, short)
                total = total + jnp.where((k >= lo) & (k < hi), state.indicator, 0.)
                return (state, short, total), state.voltage_mV[jnp.asarray(rmd_indices)]
            (_, short, total), voltage = jax.lax.scan(step, (state0, state0, jnp.zeros_like(state0.indicator)), jnp.arange(frames))
            return total / (hi - lo), short, voltage
        return jax.vmap(lambda s: jax.vmap(lambda g: trial(s, g))(gs))(starts)
    return batch


def replay(params, fixed, config, anchors, conductances, rmd_indices, step_ms, **options):
    batch = make_replay(config, rmd_indices, step_ms, **options)
    means, short, voltage = batch(params, fixed, jax.tree.map(jnp.asarray, anchors), jnp.asarray(conductances))
    return np.asarray(means), jax.tree.map(np.asarray, short), np.asarray(voltage)
