"""Planar body kinematics and low-Reynolds-number mechanics."""

from .mechanics import (
    BodyParams,
    BodyState,
    body_velocity,
    prescribed_traveling_wave,
    relative_segment_centers,
    simulate_traveling_wave,
)

__all__ = [
    "BodyParams",
    "BodyState",
    "body_velocity",
    "prescribed_traveling_wave",
    "relative_segment_centers",
    "simulate_traveling_wave",
]

