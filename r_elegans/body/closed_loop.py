"""Unpaced, sensory-connected channel brain -> anatomical muscles -> SI body.

The legacy channel_bridge stays the exact no-feedback entry point. This branch
adds only physical observation/transduction, retaining recurrent control edges.
Transduction parameters are exploratory and need separate qualification.
"""
from typing import NamedTuple

import jax

from r_elegans.brain.conductance_network import advance_network
from .actuation import muscle_body_step
from .channel_bridge import initial_channel_body_state, channel_muscle_activations
from .embodied_sensory import (
    EmbodiedSensoryConfig, EmbodiedSensoryState, transduce_sensory_signals,
    initial_embodied_sensory_state,
)
from .neuromuscular import build_muscle_projection, project_muscles_to_joints


class EmbodiedState(NamedTuple):
    brain_body: object
    sensory: EmbodiedSensoryState


class EmbodiedTrace(NamedTuple):
    voltage_mV: object
    calcium_uM: object
    muscles: object
    body: object
    sensory: object
    input_current_pA: object  # current at pre-update voltage, not mean over interval


def initial_embodied_state(fixed, network_config, body_params, world,
                          sensory_config=EmbodiedSensoryConfig()):
    state = initial_channel_body_state(fixed, network_config)
    sensory = initial_embodied_sensory_state(state.body, body_params, world, sensory_config)
    return EmbodiedState(state, sensory)


def embodied_neural_step(state, params, fixed, network_config, parent, model, body_params,
                        world, maps, dt, brain_substeps, sensory_config=EmbodiedSensoryConfig(),
                        gain_multipliers=None):
    """Advance sensing and brain; body update follows in the same physical step.

    Also used by the host wet-surface reference integrator. Recording neural
    commands first and replaying them is not valid for this feedback branch.
    """
    sensory, membrane, observed = transduce_sensory_signals(state.sensory,
        state.brain_body.body, body_params, world, maps, dt, sensory_config,
        gain_multipliers=gain_multipliers)
    current = membrane.weighted_reversal_pA-membrane.conductance_nS*state.brain_body.brain.voltage_mV
    brain = advance_network(state.brain_body.brain, params, fixed, dt, brain_substeps,
                            network_config, membrane_input=membrane)
    muscles = channel_muscle_activations(brain.voltage_mV, fixed, parent, model)
    return brain, sensory, muscles, observed, current


def embodied_channel_body_chunk(state, params, fixed, network_config, parent, model,
                               body_params, world, maps, *, dt=.002, steps=500,
                               brain_substeps=20, sensory_config=EmbodiedSensoryConfig()):
    """Causal sensing -> neuron -> NMJ -> body -> next sensing, with no teacher."""
    projection = build_muscle_projection(11, head_at_end=True)
    def step(s, _):
        brain, sensory, muscles, observed, current = embodied_neural_step(s, params, fixed,
            network_config, parent, model, body_params, world, maps, dt, brain_substeps, sensory_config)
        dorsal, ventral = project_muscles_to_joints(muscles, projection)
        body = muscle_body_step(s.brain_body.body, body_params, dorsal, ventral, dt)
        next_state = EmbodiedState(s.brain_body._replace(brain=brain, body=body), sensory)
        return next_state, EmbodiedTrace(brain.voltage_mV, brain.calcium_uM, muscles, body, observed, current)
    return jax.lax.scan(step, state, None, length=steps)
