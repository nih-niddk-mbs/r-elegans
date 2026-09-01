"""Data-backed single-compartment C. elegans neuron dynamics.

The channel set and kinetics follow Nicoletti et al. (2019), PLOS ONE
14:e0218738.  Voltage is in mV, time in ms, current in pA, conductance in
nS, capacitance in pF, and intracellular calcium in micromolar.  With these
units ``nS * mV == pA`` and ``pA / pF == mV / ms``.

Neuron-specific maximal conductances are deliberately loaded from the
external scientific-data directory; only the reusable equations live here.
"""

from __future__ import annotations

from typing import NamedTuple

import diffrax
import jax
import jax.numpy as jnp

Array = jax.Array

CHANNEL_NAMES = (
    "SHL1",
    "SHK1",
    "KVS1",
    "EGL2",
    "EGL36",
    "KQT3",
    "EGL19",
    "UNC2",
    "CCA1",
    "SLO1_EGL19",
    "SLO1_UNC2",
    "SLO2_EGL19",
    "SLO2_UNC2",
    "KCNL",
    "NCA",
    "IRK",
    "LEAK",
)
CHANNEL_INDEX = {name: index for index, name in enumerate(CHANNEL_NAMES)}
GATE_COUNT = 35


class SingleCompartmentParams(NamedTuple):
    """Biophysical parameters for one isopotential neuron."""

    capacitance_pF: Array
    conductances_nS: Array
    potassium_reversal_mV: Array
    calcium_reversal_mV: Array
    sodium_reversal_mV: Array
    leak_reversal_mV: Array
    cell_volume_um3: Array
    calcium_equilibrium_uM: Array
    calcium_removal_ms: Array
    free_calcium_fraction: Array


class SingleCompartmentState(NamedTuple):
    """Membrane voltage, channel gates, and bulk intracellular calcium."""

    voltage_mV: Array
    gates: Array
    calcium_uM: Array


def _activation(voltage: Array, half: float, slope: float) -> Array:
    return jax.nn.sigmoid((voltage - half) / slope)


def _inactivation(voltage: Array, half: float, slope: float) -> Array:
    return jax.nn.sigmoid(-(voltage - half) / slope)


def _gate_targets_and_taus(
    voltage: Array, calcium: Array, gates: Array
) -> tuple[Array, Array]:
    """Return steady-state values and time constants for all dynamic gates."""

    target = jnp.zeros((GATE_COUNT,), dtype=voltage.dtype)
    tau = jnp.ones((GATE_COUNT,), dtype=voltage.dtype)

    # Parenthesized, neuron-calibrated values in Supplementary Table 1 are
    # used when they differ from the isolated-channel fits.
    # SHL-1: activation plus fast/slow inactivation components.
    shl_m = _activation(voltage, -6.8, 14.1)
    shl_h = _inactivation(voltage, -33.1, 8.3)
    target = target.at[0:3].set(jnp.array([shl_m, shl_h, shl_h]))
    tau = tau.at[0].set(
        1.4
        / (
            jnp.exp(-(voltage + 17.5) / 12.9)
            + jnp.exp((voltage + 3.7) / 6.5)
        )
        + 0.2
    )
    tau = tau.at[1].set(53.9 / (1.0 + jnp.exp((voltage + 28.2) / 4.9)) + 2.7)
    tau = tau.at[2].set(842.2 / (1.0 + jnp.exp((voltage + 37.7) / 6.4)) + 11.9)

    # SHK-1 and KVS-1.
    target = target.at[3].set(_activation(voltage, 20.4, 7.7))
    target = target.at[4].set(_inactivation(voltage, -7.0, 5.8))
    tau = tau.at[3].set(
        26.6
        / (
            jnp.exp(-(voltage + 33.7) / 15.8)
            + jnp.exp((voltage + 33.7) / 11.2)
        )
        + 3.8
    )
    tau = tau.at[4].set(1400.0)
    target = target.at[5].set(_activation(voltage, 27.1, 25.0))
    target = target.at[6].set(_inactivation(voltage, 17.3, 11.1))
    tau = tau.at[5].set(3.0 / (1.0 + jnp.exp((voltage - 18.1) / -20.0)) + 0.1)
    tau = tau.at[6].set(8.9 / (1.0 + jnp.exp((voltage - 50.0) / -15.0)) + 5.3)

    # EGL-2 and the three EGL-36 activation timescales.
    target = target.at[7].set(_activation(voltage, 6.9, 14.9))
    tau = tau.at[7].set(8.4 / (1.0 + jnp.exp((voltage + 122.6) / -13.8)) + 4.1)
    egl36_inf = _activation(voltage, 63.0, 28.5)
    target = target.at[8:11].set(egl36_inf)
    tau = tau.at[8:11].set(jnp.array([13.0, 63.0, 355.0]))

    # KQT-3 fast/slow activation and two slow modulatory gates.
    kqt_m = _activation(voltage, 7.7, 15.8)
    target = target.at[11:13].set(kqt_m)
    tau = tau.at[11].set(39.5 / (1.0 + ((voltage - 38.1) / 33.6) ** 2))
    tau = tau.at[12].set(
        550.3
        + 534.5 / (1.0 + 10.0 ** (0.0283 * (-23.9 - voltage)))
        + 459.1 / (1.0 + 10.0 ** (0.0357 * (14.2 + voltage)))
    )
    target = target.at[13].set(
        0.49 + 0.51 / (1.0 + jnp.exp((voltage + 1.084) / 28.78))
    )
    target = target.at[14].set(
        0.34 + 0.66 / (1.0 + jnp.exp((voltage + 45.3) / 12.3))
    )
    tau = tau.at[13].set(0.5 + 2.9 / (1.0 + ((voltage + 48.1) / 48.8) ** 2))
    tau = tau.at[14].set(500.0)

    # EGL-19, UNC-2 and CCA-1 calcium channels.
    egl19_m, egl19_h, egl19_tm, egl19_th = _egl19_kinetics(voltage)
    unc2_m, unc2_h, unc2_tm, unc2_th = _unc2_kinetics(voltage)
    target = target.at[15:17].set(jnp.array([egl19_m, egl19_h]))
    tau = tau.at[15:17].set(jnp.array([egl19_tm, egl19_th]))
    target = target.at[17:19].set(jnp.array([unc2_m, unc2_h]))
    tau = tau.at[17:19].set(jnp.array([unc2_tm, unc2_th]))
    target = target.at[19].set(_activation(voltage, -57.7, 2.4))
    target = target.at[20].set(_inactivation(voltage, -73.0, 8.1))
    tau = tau.at[19].set(20.0 / (1.0 + jnp.exp((voltage + 92.5) / 21.1)) + 0.4)
    tau = tau.at[20].set(22.4 / (1.0 + jnp.exp((voltage + 75.7) / 9.4)) + 1.6)

    # Four 1:1 BK-CaV nanodomain complexes. Each triple is BK activation,
    # CaV activation, and CaV inactivation.
    for start, cav_kind, slo_kind in (
        (21, 0, 1),
        (24, 1, 1),
        (27, 0, 2),
        (30, 1, 2),
    ):
        if cav_kind == 0:
            cav_m, cav_h, cav_tm, cav_th = egl19_m, egl19_h, egl19_tm, egl19_th
        else:
            cav_m, cav_h, cav_tm, cav_th = unc2_m, unc2_h, unc2_tm, unc2_th
        bk_inf, bk_tau = _bk_kinetics(
            voltage, gates[start + 1], cav_m, cav_tm, slo_kind
        )
        target = target.at[start : start + 3].set(jnp.array([bk_inf, cav_m, cav_h]))
        tau = tau.at[start : start + 3].set(jnp.array([bk_tau, cav_tm, cav_th]))

    target = target.at[33].set(calcium / (0.33 + calcium))
    tau = tau.at[33].set(6.3)
    target = target.at[34].set(_inactivation(voltage, -86.5, 28.0))
    tau = tau.at[34].set(
        17.1 / (jnp.exp(-(voltage + 17.8) / 20.3) + jnp.exp((voltage + 43.4) / 11.2)) + 3.8
    )
    return target, jnp.maximum(tau, 1e-4)


def _egl19_kinetics(voltage: Array) -> tuple[Array, Array, Array, Array]:
    m = _activation(voltage, -4.4, 7.5)
    h = (1.43 / (1.0 + jnp.exp(-(voltage - 14.9) / 12.0)) + 0.14) * (
        5.96 / (1.0 + jnp.exp((voltage + 20.5) / 8.1)) + 0.60
    )
    tm = 2.9 * jnp.exp(-((voltage + 4.8) / 6.0) ** 2) + 1.9 * jnp.exp(
        -((voltage + 8.6) / 30.0) ** 2
    ) + 2.3
    th = 0.4 * (
        44.6 / (1.0 + jnp.exp((voltage + 33.0) / 5.0))
        + 36.4 / (1.0 + jnp.exp((voltage - 18.7) / 3.7))
        + 43.1
    )
    return m, h, tm, th


def _unc2_kinetics(voltage: Array) -> tuple[Array, Array, Array, Array]:
    m = _activation(voltage, -37.2, 4.0)
    h = _inactivation(voltage, -77.5, 5.6)
    tm = (
        4.5
        / (
            jnp.exp(-(voltage + 38.2) / 9.1)
            + jnp.exp((voltage + 38.2) / 15.4)
        )
        + 0.3
    )
    th = 142.5 / (1.0 + jnp.exp(-(voltage - 22.9) / 3.5)) + 122.6 / (
        1.0 + jnp.exp((voltage + 6.1) / 3.6)
    )
    return m, h, tm, th


def _bk_kinetics(
    voltage: Array, cav_gate: Array, cav_inf: Array, cav_tau: Array, slo_kind: int
) -> tuple[Array, Array]:
    """SLO-1/2 activation in a CaV nanodomain (Nicoletti Eqs. 21-22)."""

    if slo_kind == 1:
        wyx, wxy, wom, wop, kxy, nxy, kyx, nyx = (
            0.013, -0.028, 3.15, 0.16, 55.73, 1.30, 0.034, 0.0001
        )
    else:
        wyx, wxy, wom, wop, kxy, nxy, kyx, nyx = (
            0.019, -0.024, 0.87, 0.028, 93.45, 1.84, 3294.55, 0.00001
        )
    cain = jnp.where(
        voltage < 60.0,
        -0.04 * (voltage - 60.0) * 1e9
        / (8.0 * jnp.pi * 0.013 * 250.0 * 96485.0)
        * jnp.exp(-0.013 / jnp.sqrt(250.0 / (500.0 * 30.0))),
        0.0001,
    )
    alpha = cav_inf / cav_tau
    beta = 1.0 / cav_tau - alpha
    wm = wom * jnp.exp(-wyx * voltage)
    wp = wop * jnp.exp(-wxy * voltage)
    kom = wm / (1.0 + (cain / kyx) ** nyx)
    kop = wp / (1.0 + (kxy / cain) ** nxy)
    kcm = wm / (1.0 + (0.05 / kyx) ** nyx)
    denominator = (kop + kom) * (kcm + alpha) + beta * kcm
    return cav_gate * kop * (alpha + beta + kcm) / denominator, (alpha + beta + kcm) / denominator


def initial_single_compartment_state(
    voltage_mV: float | Array,
    params: SingleCompartmentParams,
) -> SingleCompartmentState:
    """Initialize every gate at equilibrium at the requested voltage."""

    voltage = jnp.asarray(voltage_mV)
    calcium = jnp.asarray(params.calcium_equilibrium_uM)
    gates = jnp.zeros((GATE_COUNT,), dtype=voltage.dtype)
    # The BK equilibrium depends weakly on its associated CaV gate; iterating
    # once after filling all other equilibrium gates resolves that dependency.
    target, _ = _gate_targets_and_taus(voltage, calcium, gates)
    target, _ = _gate_targets_and_taus(voltage, calcium, target)
    return SingleCompartmentState(voltage, target, calcium)


def ionic_currents(
    state: SingleCompartmentState,
    params: SingleCompartmentParams,
) -> Array:
    """Return outward-positive current for every name in ``CHANNEL_NAMES``."""

    v, x = state.voltage_mV, state.gates
    g = params.conductances_nS
    k = v - params.potassium_reversal_mV
    ca = v - params.calcium_reversal_mV
    currents = jnp.zeros((len(CHANNEL_NAMES),), dtype=v.dtype)
    currents = currents.at[0].set(g[0] * x[0] ** 3 * (0.7 * x[1] + 0.3 * x[2]) * k)
    currents = currents.at[1].set(g[1] * x[3] * x[4] * k)
    currents = currents.at[2].set(g[2] * x[5] * x[6] * k)
    currents = currents.at[3].set(g[3] * x[7] * k)
    currents = currents.at[4].set(g[4] * (0.33 * x[8] + 0.36 * x[9] + 0.39 * x[10]) * k)
    currents = currents.at[5].set(g[5] * (0.3 * x[11] + 0.7 * x[12]) * x[13] * x[14] * k)
    currents = currents.at[6].set(g[6] * x[15] * x[16] * ca)
    currents = currents.at[7].set(g[7] * x[17] * x[18] * ca)
    currents = currents.at[8].set(g[8] * x[19] ** 2 * x[20] * ca)
    currents = currents.at[9].set(g[9] * x[21] * x[23] * k)
    currents = currents.at[10].set(g[10] * x[24] * x[26] * k)
    currents = currents.at[11].set(g[11] * x[27] * x[29] * k)
    currents = currents.at[12].set(g[12] * x[30] * x[32] * k)
    currents = currents.at[13].set(g[13] * x[33] * k)
    currents = currents.at[14].set(g[14] * (v - params.sodium_reversal_mV))
    currents = currents.at[15].set(g[15] * x[34] * k)
    currents = currents.at[16].set(g[16] * (v - params.leak_reversal_mV))
    return currents


def single_compartment_rhs(
    time: Array,
    state: SingleCompartmentState,
    args: tuple[SingleCompartmentParams, Array],
) -> SingleCompartmentState:
    """Evaluate the current-clamp ODE with outward-positive ionic currents."""

    del time
    params, injected_current_pA = args
    currents = ionic_currents(state, params)
    target, tau = _gate_targets_and_taus(state.voltage_mV, state.calcium_uM, state.gates)
    calcium_current = currents[6] + currents[7] + currents[8]
    calcium_influx = jnp.where(
        state.voltage_mV <= params.calcium_reversal_mV,
        -params.free_calcium_fraction
        * calcium_current
        * 1e6
        / (2.0 * 96485.0 * params.cell_volume_um3),
        0.0,
    )
    return SingleCompartmentState(
        (injected_current_pA - jnp.sum(currents)) / params.capacitance_pF,
        (target - state.gates) / tau,
        calcium_influx
        - (state.calcium_uM - params.calcium_equilibrium_uM)
        / params.calcium_removal_ms,
    )


def integrate_single_compartment(
    state: SingleCompartmentState,
    params: SingleCompartmentParams,
    injected_current_pA: float | Array,
    duration_ms: float | Array,
    *,
    initial_step_ms: float = 0.01,
) -> SingleCompartmentState:
    """Integrate a constant-current interval and return the final state."""

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(single_compartment_rhs),
        diffrax.Kvaerno5(),
        t0=0.0,
        t1=duration_ms,
        dt0=initial_step_ms,
        y0=state,
        args=(params, jnp.asarray(injected_current_pA)),
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-7),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=100_000,
    )
    return jax.tree.map(lambda value: value[0], solution.ys)
