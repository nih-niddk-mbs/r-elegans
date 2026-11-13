"""Experimental SI world/body-to-neuron transduction for the channel brain.

Source-supported routing/signs, NOT fitted receptor kinetics or conductances:
AWA odor increase; AWC odor removal; ASEL salt increase; ASER salt decrease;
anterior bend -> DB/VB; DVA global bending-magnitude proxy; optional SMDD.
No desired gait, phase clock, source bearing or distance reaches the neurons.
AWC uses an effective signed conductance closure, not a literal cGMP cascade.
"""
from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.brain.conductance_network import MembraneConductanceInput
from .mechanics import relative_segment_centers, segment_orientations
from .neuromuscular import muscle_longitudinal_positions


CUE_NAMES = ('diacetyl', 'benzaldehyde', 'NaCl')


@dataclass(frozen=True)
class EmbodiedSensoryConfig:
    """All numerical defaults are exploratory, exposed and recorded, not fitted."""
    chemical_gain_nS: float = .005
    b_bend_gain_nS: float = .005
    dva_gain_nS: float = .002
    smdd_gain_nS: float = 0.
    receptor_time_constant_s: float = .1
    adaptation_time_constant_s: float = 2.
    bend_time_constant_s: float = .02
    chemical_reference_um: tuple[float, float, float] = (1., 1., 50000.)
    curvature_reference_per_m: float = 2000.
    anterior_offset_body_lengths: float = .1
    receptive_width_body_lengths: float = .15
    excitatory_reversal_mV: float = 0.
    inhibitory_reversal_mV: float = -80.

    def __post_init__(self):
        # JSON round trips turn tuples into lists; retain a hashable static config.
        object.__setattr__(self, 'chemical_reference_um', tuple(self.chemical_reference_um))
        scalars = [v for v in vars(self).values() if not isinstance(v, tuple)]
        if (not all(math.isfinite(v) for v in (*scalars, *self.chemical_reference_um))
                or len(self.chemical_reference_um) != 3):
            raise ValueError('Finite sensory parameters and three reference concentrations required')
        if min(self.chemical_gain_nS, self.b_bend_gain_nS, self.dva_gain_nS, self.smdd_gain_nS) < 0:
            raise ValueError('Sensory conductance gains must be nonnegative')
        if min(self.receptor_time_constant_s, self.adaptation_time_constant_s,
               self.bend_time_constant_s, self.curvature_reference_per_m,
               self.receptive_width_body_lengths, *self.chemical_reference_um) <= 0:
            raise ValueError('Positive time/curvature/width/concentration scales required')
        if not 0 <= self.anterior_offset_body_lengths <= 1:
            raise ValueError('Anterior offset must lie within one body length')
        if self.inhibitory_reversal_mV >= self.excitatory_reversal_mV:
            raise ValueError('Inhibitory reversal must be below excitatory reversal')


class ChemicalWorld(NamedTuple):
    """Prescribed Gaussian test fields, SI geometry, micromolar concentration.

    Independent ligand channels; onset/washout are experimental bath schedules,
    not a fluid/food-transport solver. These values never enter neurons directly.
    """
    source_position_m: jax.Array  # [3,2]
    width_m: jax.Array           # [3]
    background_um: jax.Array    # [3]
    peak_um: jax.Array          # [3]
    onset_s: jax.Array          # [3]
    washout_s: jax.Array        # [3]


class ChemicalBathSchedule(NamedTuple):
    """Piecewise-constant environmental concentrations; no desired motor state."""
    times_s: jax.Array
    concentration_um: jax.Array  # [events,3], uniform experimental bath


def default_chemical_world():
    """Controlled stimulus fixture, not measured bacterial odor concentrations."""
    return ChemicalWorld(jnp.array([[.001, 0.]]*3), jnp.full(3, .0005),
        jnp.array([0., 0., 50000.]), jnp.array([1., 1., 10000.]),
        jnp.array([1., 2., 3.]), jnp.array([2., 3., 4.]))


def validate_chemical_world(world):
    if isinstance(world,ChemicalBathSchedule):
        t,c=map(np.asarray,world)
        if (t.ndim!=1 or len(t)<2 or t[0]!=0 or np.any(np.diff(t)<=0)
                or c.shape!=(len(t),3) or not np.isfinite(t).all()
                or not np.isfinite(c).all() or np.any(c<0)):
            raise ValueError('Ordered nonnegative bath concentrations required')
        return
    arrays = [np.asarray(x) for x in world]
    if (arrays[0].shape != (3, 2) or any(x.shape != (3,) for x in arrays[1:])
            or not all(np.isfinite(x).all() for x in arrays)
            or np.any(arrays[1] <= 0) or np.any(arrays[2] < 0) or np.any(arrays[3] < 0)
            or np.any(arrays[4] < 0) or np.any(arrays[5] <= arrays[4])):
        raise ValueError('Invalid SI chemical test fields/schedules')


class SensoryMaps(NamedTuple):
    chemical_indices: jax.Array  # AWAL/R,AWCL/R,ASEL/R
    b_indices: jax.Array
    b_dorsal_sign: jax.Array
    b_receptive_weights: jax.Array  # [18,11], actual shape tail->head
    dva_index: jax.Array
    smdd_indices: jax.Array
    head_weights: jax.Array


def build_sensory_maps(model, config=EmbodiedSensoryConfig()):
    """Estimate receptive fields from anatomical NMJ output plus anterior offset.

    Output territories are NOT measured stretch-receptor locations. No learned
    somatic position or direct imposed traveling-wave phase is inserted.
    """
    ids = model.neuron_ids
    b_names = tuple([f'DB{i:02d}' for i in range(1, 8)]+[f'VB{i:02d}' for i in range(1, 12)])
    indices = np.array([ids.index(n) for n in b_names])
    nmj = np.asarray(model.neuromuscular_params.synapse_weights)[:, indices]
    sums = nmj.sum(axis=0)
    if np.any(sums <= 0):
        raise ValueError('Each B motor neuron needs an anatomical output territory')
    muscle_s = 1.-np.asarray(muscle_longitudinal_positions())
    centers = muscle_s @ nmj / sums
    receptive_centers = np.minimum(1., centers+config.anterior_offset_body_lengths)
    joint_s = np.arange(1, 12)/12.
    weights = np.exp(-.5*((joint_s[None, :]-receptive_centers[:, None])/
                         config.receptive_width_body_lengths)**2)
    weights /= weights.sum(axis=1, keepdims=True)
    head_weights = np.where(joint_s >= .75, 1., 0.)
    head_weights /= head_weights.sum()
    return SensoryMaps(jnp.array([ids.index(n) for n in ('AWAL','AWAR','AWCL','AWCR','ASEL','ASER')]),
        jnp.asarray(indices), jnp.array([1.]*7+[-1.]*11), jnp.asarray(weights),
        jnp.asarray(ids.index('DVA')), jnp.array([ids.index('SMDDL'),ids.index('SMDDR')]),
        jnp.asarray(head_weights))


class EmbodiedSensoryState(NamedTuple):
    receptor: jax.Array  # three bounded occupancies
    adapted: jax.Array
    curvature_per_m: jax.Array


class SensoryObservation(NamedTuple):
    head_position_m: jax.Array
    concentration_um: jax.Array
    chemical_response: jax.Array
    curvature_per_m: jax.Array
    chemical_excitatory_nS: jax.Array
    chemical_inhibitory_nS: jax.Array
    proprioceptive_nS: jax.Array


def sensory_head_position(body, body_params):
    """True anterior endpoint; final rod is the head in this model."""
    ds = body_params.mechanics.segment_length
    center = relative_segment_centers(body.joint_angles, ds)[-1]
    angle = segment_orientations(body.joint_angles)[-1]
    endpoint = center+.5*ds*jnp.array([jnp.cos(angle), jnp.sin(angle)])
    c, s = jnp.cos(body.heading), jnp.sin(body.heading)
    return body.position+jnp.array([[c, -s], [s, c]]) @ endpoint


def sample_chemical_world(head_position_m, time_s, world):
    if isinstance(world,ChemicalBathSchedule):
        index=jnp.clip(jnp.searchsorted(world.times_s,time_s,side='right')-1,0,len(world.times_s)-1)
        return world.concentration_um[index]
    squared = jnp.sum((head_position_m-world.source_position_m)**2, axis=1)
    active = (time_s >= world.onset_s) & (time_s < world.washout_s)
    return world.background_um+world.peak_um*jnp.exp(-.5*squared/world.width_m**2)*active


def initial_embodied_sensory_state(body, body_params, world, config=EmbodiedSensoryConfig()):
    head = sensory_head_position(body, body_params)
    concentration = sample_chemical_world(head, body.time, world)
    occupancy = concentration/(concentration+jnp.array(config.chemical_reference_um))
    return EmbodiedSensoryState(occupancy, occupancy, body.joint_angles/body_params.mechanics.segment_length)


def transduce_sensory_signals(state, body, body_params, world, maps, dt,
                             config=EmbodiedSensoryConfig(), *, neuron_count=302,
                             gain_multipliers=None):
    """Body/world -> conductances, with memory held across chunks.

    Conductance is an effective bounded transduction closure. AWC odor-on
    hyperpolarization is represented by inhibitory conductance; this does not
    claim to implement TAX-2/4 closing. DVA's magnitude proxy is not a resolved
    strain measurement of its neurite. No physical sensory signs are fitted here.
    """
    gains = jnp.ones(3) if gain_multipliers is None else gain_multipliers
    head = sensory_head_position(body, body_params)
    concentration = sample_chemical_world(head, body.time, world)
    occupancy = concentration/(concentration+jnp.array(config.chemical_reference_um))
    receptor = state.receptor+(-jnp.expm1(-dt/config.receptor_time_constant_s))*(occupancy-state.receptor)
    # Use old adapted level for this causal interval; update both memories.
    response = receptor-state.adapted
    adapted = state.adapted+(-jnp.expm1(-dt/config.adaptation_time_constant_s))*(receptor-state.adapted)
    actual_curvature = body.joint_angles/body_params.mechanics.segment_length
    curvature = state.curvature_per_m+(-jnp.expm1(-dt/config.bend_time_constant_s))*(actual_curvature-state.curvature_per_m)
    signed_chemo = jnp.array([response[0], response[0], -response[1], -response[1], response[2], -response[2]])
    chemical_excitatory = jnp.zeros(neuron_count).at[maps.chemical_indices].set(
        config.chemical_gain_nS*gains[0]*jnp.maximum(signed_chemo, 0.))
    # Only AWC is modeled with a signed inhibitory response. AWA/ASE are ON/OFF
    # rectified excitatory proxies, not artificial mirror-image hyperpolarization.
    chemical_inhibitory = jnp.zeros(neuron_count).at[maps.chemical_indices[2:4]].set(
        config.chemical_gain_nS*gains[0]*jnp.maximum(response[1], 0.))
    local_bend = maps.b_receptive_weights @ curvature/config.curvature_reference_per_m
    b_activation = jnp.maximum(jnp.tanh(maps.b_dorsal_sign*local_bend), 0.)
    proprio = jnp.zeros(neuron_count).at[maps.b_indices].set(config.b_bend_gain_nS*gains[1]*b_activation)
    magnitude = (jnp.sqrt(jnp.mean((curvature/config.curvature_reference_per_m)**2)+1e-12)-1e-6)
    proprio = proprio.at[maps.dva_index].set(config.dva_gain_nS*gains[2]*jnp.tanh(jnp.maximum(magnitude, 0.)))
    head_bend = maps.head_weights @ curvature/config.curvature_reference_per_m
    proprio = proprio.at[maps.smdd_indices].set(config.smdd_gain_nS*jnp.maximum(jnp.tanh(head_bend), 0.))
    excitatory = chemical_excitatory+proprio
    membrane = MembraneConductanceInput(excitatory+chemical_inhibitory,
        config.excitatory_reversal_mV*excitatory+config.inhibitory_reversal_mV*chemical_inhibitory)
    observed = SensoryObservation(head, concentration, response, actual_curvature,
        chemical_excitatory, chemical_inhibitory, proprio)
    return EmbodiedSensoryState(receptor, adapted, curvature), membrane, observed
