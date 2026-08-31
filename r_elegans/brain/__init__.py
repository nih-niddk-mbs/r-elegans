"""Continuous neural dynamics."""

from .dynamics import (
    NeuralParams,
    effective_chemical_weights,
    effective_gap_weights,
    integrate_neural_state,
    neural_rhs,
)

__all__ = [
    "NeuralParams",
    "effective_chemical_weights",
    "effective_gap_weights",
    "integrate_neural_state",
    "neural_rhs",
]

