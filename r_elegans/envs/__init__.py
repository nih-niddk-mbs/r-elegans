"""Differentiable environments and sensory transduction models.

``gymnax_petri_dish`` is not imported here even though it lives in this
package: it requires the optional ``gymnax``/``flax`` dependencies (the
``env`` extra), and this package is imported by ``r_elegans.demo``, which must
keep working in a clean install with only the core dependencies. Import it
explicitly, e.g. ``from r_elegans.envs.gymnax_petri_dish import
PetriDishGymnaxEnv``.
"""

from .petri_dish import (
    PetriDishParams,
    PetriObservation,
    PetriWormState,
    SensoryPolicyParams,
    decode_sensory_policy,
    default_petri_dish_params,
    food_concentration,
    head_position,
    initialize_petri_worm,
    petri_navigation_loss,
    simulate_neural_petri_dish,
    simulate_petri_dish,
)

__all__ = [
    "PetriDishParams",
    "PetriObservation",
    "PetriWormState",
    "SensoryPolicyParams",
    "decode_sensory_policy",
    "default_petri_dish_params",
    "food_concentration",
    "head_position",
    "initialize_petri_worm",
    "petri_navigation_loss",
    "simulate_neural_petri_dish",
    "simulate_petri_dish",
]
