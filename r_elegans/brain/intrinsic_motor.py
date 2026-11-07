"""Reduced motor-cell excitability hypothesis; NOT fitted worm ion channels.

Morris–Lecar-type calcium activation and delayed potassium recovery generate
the rhythm in membrane state, without an imposed phase or time-varying input.
DA/VA have a fixed permissive bias; DB/VB require steady depolarizing drive.
The coefficients are demonstration values, not measured UNC-2 kinetics. The
voltage transform and clock scale connect to our normalized actor convention,
not a validated physical-unit calibration. See LAB_NOTEBOOK E024.
"""
from dataclasses import dataclass
import re

import jax
import jax.numpy as jnp

from .dynamics import neural_currents


@dataclass(frozen=True)
class IntrinsicMotorConfig:
    calcium_conductance: float = 4.4
    potassium_conductance: float = 8.0
    leak_conductance: float = 2.0
    capacitance: float = 20.0
    recovery_rate: float = 0.04
    clock_scale: float = 50.0
    a_bias: float = 90.0
    applied_current_scale: float = 900.0
    coupling_current_scale: float = 100.0

    def __post_init__(self):
        import math
        if not all(math.isfinite(x) for x in vars(self).values()):
            raise ValueError("Intrinsic motor coefficients must be finite")
        if (min(self.calcium_conductance, self.potassium_conductance,
                self.leak_conductance) < 0 or
                min(self.capacitance, self.recovery_rate, self.clock_scale,
                    self.applied_current_scale, self.coupling_current_scale) <= 0):
            raise ValueError("Invalid intrinsic motor coefficients")


def motor_class_masks(neuron_ids):
    """Canonical numbered DA/VA and DB/VB only (not DVA, AVA, AVB, etc.)."""
    a = jnp.asarray([bool(re.fullmatch(r"(?:DA|VA)\d+", n)) for n in neuron_ids])
    b = jnp.asarray([bool(re.fullmatch(r"(?:DB|VB)\d+", n)) for n in neuron_ids])
    return a, b


def recovery_equilibrium(voltage):
    potential = 100.0 * (voltage + 0.35) - 60.0
    return 0.5 * (1.0 + jnp.tanh((potential - 2.0) / 30.0))


def initial_motor_state(voltage):
    """Two array rows: normalized voltage and potassium activation."""
    return jnp.stack((voltage, recovery_equilibrium(voltage)))


def motor_cell_rhs(voltage, recovery, bias, applied, config=IntrinsicMotorConfig()):
    """Autonomous conductance dynamics, with optional constant current input.

    Arguments broadcast for isolated cell assays and full-network integration.
    No time argument, sinusoid, or phase state is present.
    """
    potential = 100.0 * (voltage + 0.35) - 60.0
    calcium_gate = 0.5 * (1.0 + jnp.tanh((potential + 1.2) / 18.0))
    calcium = config.calcium_conductance * calcium_gate * (120.0 - potential)
    potassium = config.potassium_conductance * recovery * (-84.0 - potential)
    leak = config.leak_conductance * (-60.0 - potential)
    dv = config.clock_scale * (calcium + potassium + leak + bias + applied)
    dv /= config.capacitance * 100.0
    dw = (config.clock_scale * config.recovery_rate
          * (recovery_equilibrium(voltage) - recovery)
          * jnp.cosh((potential - 2.0) / 60.0))
    return dv, dw


def rk4_step(state, rhs, dt):
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0


def integrate_intrinsic_motor(state, params, external, a_mask, b_mask,
                              dt, substeps, config=IntrinsicMotorConfig()):
    """Coupled neural scan; all anatomical synaptic/electrical edges retained.

    Selected motor cells replace the generic leak/tau equation with active
    membrane dynamics. Generic threshold/slope still govern transmitter output.
    For selected cells leak_reversal/time_constant no longer govern voltage.
    Nonmotor cells retain the generic ODE; their recovery row remains constant.
    RK4 requires timestep convergence checks; unlike the legacy exponential
    integrator it does not promise stability for arbitrarily large conductances.
    """
    active = a_mask | b_mask

    def rhs(s):
        v, w = s
        leak_external, gap, chemical = neural_currents(v, params, external)
        generic = (leak_external + gap + chemical) / params.time_constant
        applied = (config.applied_current_scale * external
                   + config.coupling_current_scale * (gap + chemical))
        dv, dw = motor_cell_rhs(v, w, config.a_bias * a_mask, applied, config)
        return jnp.stack((jnp.where(active, dv, generic), jnp.where(active, dw, 0.0)))

    def step(s, _):
        return rk4_step(s, rhs, dt / substeps), None
    return jax.lax.scan(step, state, None, length=substeps)[0]
