"""All-cell channel network initialized from a normalized full-network actor.

Each cell has the existing 17-current library, 35 gates and intracellular Ca.
There is no spike/reset rule. Most starting channel compositions are transferred
priors, NOT measured neuron-specific expression. Chemical currents retain parent
reversal priors (effective cation/anion currents, not identified receptor models).
The normalized-to-physical migration is an explicit hypothesis, not a fit.
"""
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .circuit import decode_subcircuit_params
from .single_compartment import (
    CHANNEL_INDEX, SingleCompartmentParams, SingleCompartmentState,
    initial_single_compartment_state, ionic_currents, _gate_targets_and_taus,
)
from r_elegans.data.physiology import load_neuron_parameters


@dataclass(frozen=True)
class ConductanceNetworkConfig:
    voltage_scale_mV: float = 100.
    voltage_offset_mV: float = -25.
    synaptic_unit_nS: float = .03
    transferred_channel_scale: float = .01
    indicator_kd_uM: float = .3
    indicator_hill: float = 2.
    indicator_tau_s: float = .5

    def __post_init__(self):
        import math
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("Finite network configuration required")
        if any(v <= 0 for k,v in vars(self).items() if k != "voltage_offset_mV"):
            raise ValueError("Positive conductance/indicator scales required")


class ConductanceNetworkParams(NamedTuple):
    log_channel_scale: jax.Array
    raw_chemical: jax.Array
    raw_gap: jax.Array


class ConductanceNetworkFixed(NamedTuple):
    cells: SingleCompartmentParams
    chemical_mask: jax.Array
    gap_mask: jax.Array
    synapse_reversal_mV: jax.Array
    release_threshold_mV: jax.Array
    release_slope_mV: jax.Array
    bias_pA: jax.Array
    initial_voltage_mV: jax.Array


class ConductanceNetworkState(NamedTuple):
    voltage_mV: jax.Array
    gates: jax.Array
    calcium_uM: jax.Array
    indicator: jax.Array


class MembraneConductanceInput(NamedTuple):
    """Additional physical membrane channels, held during one body interval.

    ``conductance_nS`` is nonnegative; ``weighted_reversal_pA`` is sum(g*E).
    Net inward current is sum(g*E)-sum(g)*V, not a prescribed voltage/current.
    Input producers must validate units, shapes, bounds and reversal potentials.
    """
    conductance_nS: jax.Array
    weighted_reversal_pA: jax.Array


def indicator_equilibrium(calcium, config):
    fraction = jnp.maximum(calcium, 0.) / config.indicator_kd_uM
    power = fraction**config.indicator_hill
    return power/(1.+power)


def initialize_conductance_network(parent, neuron_ids, config=ConductanceNetworkConfig()):
    """Retain parent edges/release fields; add explicitly transferred cell priors.

    Vphysical=100*Vparent-25 mV by default. A common g unit and C_i=g*tau_i
    preserve the passive time-constant convention and symmetric electrical edges.
    Channel dynamics change the function; old skills must be requalified.
    Normalized fields may lie outside physical ranges after behavioral fitting;
    migration bounds voltage to [-80,-35], reversal to [-85,0], release threshold
    to [-65,-20] mV, slope to [2,20] mV and capacitance to [1,20] pF.
    Fixed bias balances initial intrinsic current only; recurrence remains active.
    """
    from r_elegans.model import load_builtin_model
    if tuple(neuron_ids) != load_builtin_model().neuron_ids:
        raise ValueError("Canonical full 302-neuron ordering required")
    awc = load_neuron_parameters("AWCON").params
    rmd = load_neuron_parameters("RMD").params
    cells = jax.tree.map(lambda a,b: jnp.stack([b if n.startswith("RMD") else a for n in neuron_ids]),awc,rmd)
    scale = jnp.asarray([1. if n.startswith("RMD") else config.transferred_channel_scale for n in neuron_ids])
    g = cells.conductances_nS * scale[:,None]
    # An explicit small floor keeps transferred-zero currents available to fit.
    g = jnp.maximum(g, 1e-5).at[:,CHANNEL_INDEX["LEAK"]].set(config.synaptic_unit_nS)
    neural = decode_subcircuit_params(parent.neural)
    cells = cells._replace(conductances_nS=g,
        capacitance_pF=jnp.clip(1000.*neural.time_constant*config.synaptic_unit_nS,1.,20.),
        leak_reversal_mV=jnp.clip(config.voltage_scale_mV*neural.leak_reversal+config.voltage_offset_mV,-80.,-35.))
    params = ConductanceNetworkParams(jnp.zeros_like(g), parent.neural.raw_chemical,parent.neural.raw_gap)
    fixed = ConductanceNetworkFixed(cells,parent.neural.chemical_mask,parent.neural.gap_mask,
        jnp.clip(config.voltage_scale_mV*neural.synapse_reversal+config.voltage_offset_mV,-85.,0.),
        jnp.clip(config.voltage_scale_mV*neural.threshold+config.voltage_offset_mV,-65.,-20.),
        jnp.clip(config.voltage_scale_mV*neural.slope,2.,20.),jnp.zeros(302),cells.leak_reversal_mV)
    state = initial_network_state(fixed,config)
    currents = jax.vmap(ionic_currents)(SingleCompartmentState(state.voltage_mV,state.gates,state.calcium_uM),cells)
    return params,fixed._replace(bias_pA=currents.sum(axis=1))


def initial_network_state(fixed,config):
    states = jax.vmap(initial_single_compartment_state)(fixed.initial_voltage_mV,fixed.cells)
    return ConductanceNetworkState(states.voltage_mV,states.gates,states.calcium_uM,
        indicator_equilibrium(states.calcium_uM,config))


def network_synapses(params,fixed,config):
    chemical = config.synaptic_unit_nS*jax.nn.softplus(params.raw_chemical)*fixed.chemical_mask
    gap = config.synaptic_unit_nS*jax.nn.softplus(.5*(params.raw_gap+params.raw_gap.T))*fixed.gap_mask
    return chemical,gap


def channel_open_fractions(x):
    """Open probabilities in the exact ordering of single_compartment currents."""
    return jnp.stack((x[:,0]**3*(.7*x[:,1]+.3*x[:,2]),x[:,3]*x[:,4],x[:,5]*x[:,6],
        x[:,7],.33*x[:,8]+.36*x[:,9]+.39*x[:,10],
        (.3*x[:,11]+.7*x[:,12])*x[:,13]*x[:,14],
        x[:,15]*x[:,16],x[:,17]*x[:,18],x[:,19]**2*x[:,20],
        x[:,21]*x[:,23],x[:,24]*x[:,26],x[:,27]*x[:,29],x[:,30]*x[:,32],
        x[:,33],jnp.ones(x.shape[0]),x[:,34],jnp.ones(x.shape[0])),axis=1)


def advance_network(state,params,fixed,dt_s,substeps,config=ConductanceNetworkConfig(),*,membrane_input=None):
    """Frozen-conductance exponential voltage/gate/Ca update; first order.

    All neurons evolve simultaneously, including unobserved ones. No calcium
    target or behavior is injected. Check convergence by doubling substeps.
    Optional membrane_input adds physical conductances during the interval;
    omitting it retains the historical no-input calculation unchanged.
    """
    h = dt_s*1000./substeps
    if membrane_input is not None:
        expected = state.voltage_mV.shape
        if (membrane_input.conductance_nS.shape != expected or
                membrane_input.weighted_reversal_pA.shape != expected):
            raise ValueError('Membrane conductance input must have one value per neuron')
    g = fixed.cells.conductances_nS*jnp.exp(params.log_channel_scale)
    chemical,gap = network_synapses(params,fixed,config)
    gap_total = gap.sum(axis=1)
    cells = fixed.cells
    reversals = jnp.stack([cells.calcium_reversal_mV if i in (6,7,8)
        else cells.sodium_reversal_mV if i == 14 else cells.leak_reversal_mV if i == 16
        else cells.potassium_reversal_mV for i in range(17)],axis=1)

    def step(s,_):
        conductance = g*channel_open_fractions(s.gates)
        release = jax.nn.sigmoid((s.voltage_mV-fixed.release_threshold_mV)/fixed.release_slope_mV)
        total = conductance.sum(axis=1)+chemical@release+gap_total
        drive = ((conductance*reversals).sum(axis=1)+chemical@(release*fixed.synapse_reversal_mV)
                 +gap@s.voltage_mV+fixed.bias_pA)
        if membrane_input is not None:
            total = total+membrane_input.conductance_nS
            drive = drive+membrane_input.weighted_reversal_pA
        voltage = s.voltage_mV+(drive/jnp.maximum(total,1e-12)-s.voltage_mV)*(-jnp.expm1(-h*total/cells.capacitance_pF))
        target,tau = jax.vmap(_gate_targets_and_taus)(s.voltage_mV,s.calcium_uM,s.gates)
        gates = s.gates+(target-s.gates)*(-jnp.expm1(-h/tau))
        calcium_current = jnp.sum(conductance[:,6:9]*(s.voltage_mV[:,None]-cells.calcium_reversal_mV[:,None]),axis=1)
        influx = jnp.where(s.voltage_mV<=cells.calcium_reversal_mV,
            -cells.free_calcium_fraction*calcium_current*1e6/(2.*96485.*cells.cell_volume_um3),0.)
        ca_target = cells.calcium_equilibrium_uM+influx*cells.calcium_removal_ms
        calcium = s.calcium_uM+(ca_target-s.calcium_uM)*(-jnp.expm1(-h/cells.calcium_removal_ms))
        indicator = s.indicator+(indicator_equilibrium(s.calcium_uM,config)-s.indicator)*(-jnp.expm1(-h/(1000.*config.indicator_tau_s)))
        return ConductanceNetworkState(voltage,gates,calcium,indicator),None
    return jax.lax.scan(jax.checkpoint(step),state,None,length=substeps)[0]


def network_recording(params,fixed,dt_s,frames,substeps,config=ConductanceNetworkConfig()):
    return network_recording_intervals(params,fixed,jnp.full(frames,dt_s),substeps,config)


def network_recording_intervals(params,fixed,intervals_s,substeps,config=ConductanceNetworkConfig()):
    def step(state,dt):
        state = advance_network(state,params,fixed,dt,substeps,config)
        return state,state
    return jax.lax.scan(jax.checkpoint(step),initial_network_state(fixed,config),intervals_s)[1]


def save_network(path,params,fixed,config):
    """Versioned self-contained checkpoint, including fixed channels and units."""
    import json
    from dataclasses import asdict
    from r_elegans.model import load_builtin_model
    values={"schema_version":np.array(1),"config_json":np.array(json.dumps(asdict(config)))}
    values['neuron_ids']=np.asarray(load_builtin_model().neuron_ids)
    values.update({"params_"+k:np.asarray(v) for k,v in params._asdict().items()})
    values.update({"cells_"+k:np.asarray(v) for k,v in fixed.cells._asdict().items()})
    values.update({"fixed_"+k:np.asarray(v) for k,v in fixed._asdict().items() if k != "cells"})
    np.savez_compressed(path,**values)


def load_network(path):
    import json
    from r_elegans.model import load_builtin_model
    with np.load(path,allow_pickle=False) as d:
        if int(d["schema_version"]) != 1:raise ValueError("Unsupported conductance checkpoint")
        if tuple(d['neuron_ids'].tolist()) != load_builtin_model().neuron_ids:
            raise ValueError('Noncanonical checkpoint neuron ordering')
        params=ConductanceNetworkParams(**{k:jnp.asarray(d["params_"+k]) for k in ConductanceNetworkParams._fields})
        cells=SingleCompartmentParams(**{k:jnp.asarray(d["cells_"+k]) for k in SingleCompartmentParams._fields})
        fixed=ConductanceNetworkFixed(cells,**{k:jnp.asarray(d["fixed_"+k]) for k in ConductanceNetworkFixed._fields if k != "cells"})
        config=ConductanceNetworkConfig(**json.loads(str(d["config_json"])))
    if (params.log_channel_scale.shape != (302,17)
            or params.raw_chemical.shape != (302,302) or params.raw_gap.shape != (302,302)
            or fixed.chemical_mask.shape != (302,302) or fixed.gap_mask.shape != (302,302)
            or any(v.shape != (302,) for k,v in fixed._asdict().items() if k not in ('cells','chemical_mask','gap_mask'))
            or any(v.shape != ((302,17) if k=='conductances_nS' else (302,)) for k,v in cells._asdict().items())):
        raise ValueError("Invalid conductance network shapes")
    if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves((params,fixed))):
        raise ValueError("Nonfinite conductance checkpoint")
    if (not np.array_equal(fixed.gap_mask,fixed.gap_mask.T)
            or not np.all(np.isin(fixed.chemical_mask,[0,1])) or not np.all(np.isin(fixed.gap_mask,[0,1]))
            or np.any(cells.conductances_nS < 0) or np.any(cells.capacitance_pF <= 0)
            or np.any(fixed.release_slope_mV <= 0)):
        raise ValueError('Invalid conductance, mask or release values')
    return params,fixed,config
