"""Continuous neural dynamics."""

from .dynamics import (
    NeuralParams,
    effective_chemical_weights,
    effective_gap_weights,
    integrate_neural_state,
    neural_rhs,
)
from .motor_control import (
    MOTOR_FEATURE_COUNT,
    effective_neural_motor_coefficients,
    motor_command_features,
    motor_features_from_phase,
    neural_motor_loss,
    neural_motor_voltage,
)
from .single_compartment import (
    CHANNEL_INDEX,
    CHANNEL_NAMES,
    GATE_COUNT,
    SingleCompartmentParams,
    SingleCompartmentState,
    initial_single_compartment_state,
    integrate_single_compartment,
    ionic_currents,
    single_compartment_rhs,
)

__all__ = [
    "CHANNEL_INDEX",
    "CHANNEL_NAMES",
    "GATE_COUNT",
    "MOTOR_FEATURE_COUNT",
    "NeuralParams",
    "SingleCompartmentParams",
    "SingleCompartmentState",
    "effective_chemical_weights",
    "effective_gap_weights",
    "effective_neural_motor_coefficients",
    "integrate_neural_state",
    "initial_single_compartment_state",
    "integrate_single_compartment",
    "ionic_currents",
    "motor_command_features",
    "motor_features_from_phase",
    "neural_motor_loss",
    "neural_motor_voltage",
    "neural_rhs",
    "single_compartment_rhs",
]
