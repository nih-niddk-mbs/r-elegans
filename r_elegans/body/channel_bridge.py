"""Unpaced channel-network → anatomical NMJs → planar body diagnostic.

Common execution state does not imply proprioception: body state is NOT fed
back into neurons here. Physical network release fields are used explicitly;
the frozen parent provides NMJ multipliers/signs and muscle transfer parameters.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from r_elegans.brain.conductance_network import (
    advance_network,initial_network_state,ConductanceNetworkState,
)
from .actuation import MuscleBodyState,initialize_muscle_body,muscle_body_step
from .neuromuscular import build_muscle_projection,project_muscles_to_joints


class ChannelBodyState(NamedTuple):
    brain: ConductanceNetworkState
    body: MuscleBodyState


def channel_muscle_activations(voltage_mV,fixed,parent,model):
    """Same NMJ/muscle transform as the parent full actor, physical release units.

    No resting-tone subtraction or synthetic gait. Unknown NMJ polarities retain
    their frozen parent values. Threshold/slope are those of the channel network,
    including its documented migration bounds, not silently unmapped mV inputs.
    """
    nmj=model.neuromuscular_params
    mask=(nmj.synapse_weights>0)
    weights=nmj.synapse_weights*jax.nn.softplus(parent.raw_nmj)
    signs=jnp.where(nmj.synapse_signs!=0,nmj.synapse_signs,jnp.tanh(parent.unknown_polarity))*mask
    release=jax.nn.sigmoid((voltage_mV-fixed.release_threshold_mV)/fixed.release_slope_mV)
    drive=jnp.matmul(weights*signs,release,precision=jax.lax.Precision.HIGHEST)
    drive/=jnp.maximum(weights.sum(axis=1),1.)
    activation=jax.nn.sigmoid((drive-parent.muscle_threshold)/(.02+jax.nn.softplus(parent.raw_muscle_slope)))
    return jnp.clip(activation,1e-5,1.-1e-5)


def initial_channel_body_state(fixed,config):
    return ChannelBodyState(initial_network_state(fixed,config),initialize_muscle_body(12))


def channel_body_chunk(state,params,fixed,config,parent,model,body_params,*,
                       dt=.002,steps=500,brain_substeps=20,hold_voltage=False):
    """Return continued state and synchronized neural/muscle/body samples.

    hold_voltage freezes the brain for a posture-relaxation control. Otherwise
    neurons evolve without applied pacing. Body feedback/sensory currents absent.
    """
    projection=build_muscle_projection(11,head_at_end=True)
    def step(s,_):
        brain=s.brain if hold_voltage else advance_network(s.brain,params,fixed,dt,brain_substeps,config)
        muscles=channel_muscle_activations(brain.voltage_mV,fixed,parent,model)
        dorsal,ventral=project_muscles_to_joints(muscles,projection)
        body=muscle_body_step(s.body,body_params,dorsal,ventral,dt)
        return ChannelBodyState(brain,body),(brain.voltage_mV,brain.calcium_uM,muscles,body)
    return jax.lax.scan(step,state,None,length=steps)
