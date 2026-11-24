"""Opt-in exponential midpoint neural integrator; legacy solver is unchanged.

Predict a half-step, reevaluate all neural coefficients there, then advance the
original state for a full step. Held outer peptide input is still first order.
No parameter, channel law or reversal is changed. No sparse release override.
"""
import jax
import jax.numpy as jnp
from .conductance_network import (ConductanceNetworkConfig,ConductanceNetworkState,
    channel_open_fractions,network_synapses,chemical_reversal_drive,indicator_equilibrium)
from .single_compartment import _gate_targets_and_taus


def advance_midpoint(state,params,fixed,dt_s,substeps,config=ConductanceNetworkConfig(),*,membrane_input=None):
    if substeps<1 or dt_s<=0:raise ValueError('Positive duration/substeps required')
    if membrane_input is not None and (membrane_input.conductance_nS.shape!=state.voltage_mV.shape or membrane_input.weighted_reversal_pA.shape!=state.voltage_mV.shape):
        raise ValueError('Membrane input shape')
    h=dt_s*1000/substeps;cells=fixed.cells
    g=cells.conductances_nS*jnp.exp(params.log_channel_scale)
    chemical,gap=network_synapses(params,fixed,config);gap_total=gap.sum(1)
    reversals=jnp.stack([cells.calcium_reversal_mV if i in (6,7,8) else cells.sodium_reversal_mV if i==14 else cells.leak_reversal_mV if i==16 else cells.potassium_reversal_mV for i in range(17)],axis=1)
    def coefficients(s):
        conductance=g*channel_open_fractions(s.gates)
        release=jax.nn.sigmoid((s.voltage_mV-fixed.release_threshold_mV)/fixed.release_slope_mV)
        total=conductance.sum(1)+chemical@release+gap_total
        drive=(conductance*reversals).sum(1)+chemical_reversal_drive(chemical,release,fixed.synapse_reversal_mV)+gap@s.voltage_mV+fixed.bias_pA
        if membrane_input is not None:total=total+membrane_input.conductance_nS;drive=drive+membrane_input.weighted_reversal_pA
        target,tau=jax.vmap(_gate_targets_and_taus)(s.voltage_mV,s.calcium_uM,s.gates)
        ca_current=jnp.sum(conductance[:,6:9]*(s.voltage_mV[:,None]-cells.calcium_reversal_mV[:,None]),axis=1)
        influx=jnp.where(s.voltage_mV<=cells.calcium_reversal_mV,-cells.free_calcium_fraction*ca_current*1e6/(2.*96485.*cells.cell_volume_um3),0.)
        return total,drive,target,tau,cells.calcium_equilibrium_uM+influx*cells.calcium_removal_ms,indicator_equilibrium(s.calcium_uM,config)
    def frozen(s,coef,length):
        total,drive,target,tau,ca_target,indicator_target=coef
        return ConductanceNetworkState(
            s.voltage_mV+(drive/jnp.maximum(total,1e-12)-s.voltage_mV)*(-jnp.expm1(-length*total/cells.capacitance_pF)),
            s.gates+(target-s.gates)*(-jnp.expm1(-length/tau)),
            s.calcium_uM+(ca_target-s.calcium_uM)*(-jnp.expm1(-length/cells.calcium_removal_ms)),
            s.indicator+(indicator_target-s.indicator)*(-jnp.expm1(-length/(1000.*config.indicator_tau_s))))
    def step(s,_):
        midpoint=frozen(s,coefficients(s),h/2)
        return frozen(s,coefficients(midpoint),h),None
    return jax.lax.scan(step,state,None,length=substeps)[0]
