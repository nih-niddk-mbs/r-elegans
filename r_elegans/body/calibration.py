"""Published adult-worm calibrations for the reduced-order body model.

All quantities in this module use SI units. The liquid model uses Lighthill's
local resistive-force coefficients. The agar model uses effective surface-drag
coefficients: it is a calibrated low-Re crawling law, not a simulation of the
microscopic lubrication layer or deformable gel.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi
from typing import Literal

import jax.numpy as jnp

from .actuation import MuscleBodyParams
from .mechanics import BodyParams

MediumKind = Literal["liquid", "agar"]


@dataclass(frozen=True)
class AdultWormCalibration:
    """Geometry and passive material properties of a representative adult."""

    body_length_m: float = 1.0e-3
    body_radius_m: float = 40.0e-6
    bending_modulus_n_m2: float = 9.5e-14
    internal_viscosity_upper_bound_n_m2_s: float = 5.0e-16
    muscle_time_constant_s: float = 0.1
    maximum_curvature_per_m: float = 1.0e4


@dataclass(frozen=True)
class MediumCalibration:
    """Physical medium metadata and whole-body drag coefficients."""

    kind: MediumKind
    density_kg_m3: float
    dynamic_viscosity_pa_s: float | None
    parallel_drag_kg_s: float
    perpendicular_drag_kg_s: float
    drag_model: str

    @property
    def drag_ratio(self) -> float:
        return self.perpendicular_drag_kg_s / self.parallel_drag_kg_s


def lighthill_drag_per_length(
    dynamic_viscosity_pa_s: float,
    body_radius_m: float,
    wave_length_m: float,
) -> tuple[float, float]:
    """Return tangential and normal RFT coefficients in kg/(m s).

    Uses ``q = 0.09 * wavelength`` and the coefficient convention employed by
    Boyle et al. (2012). The logarithm requires the centerline wavelength to
    be sufficiently larger than the body radius.
    """

    if dynamic_viscosity_pa_s <= 0.0:
        raise ValueError("Dynamic viscosity must be positive")
    if body_radius_m <= 0.0 or wave_length_m <= 0.0:
        raise ValueError("Body radius and wavelength must be positive")
    logarithm = log(2.0 * 0.09 * wave_length_m / body_radius_m)
    if logarithm <= 0.0:
        raise ValueError("Lighthill RFT requires 0.18*wavelength > body radius")
    parallel = 2.0 * pi * dynamic_viscosity_pa_s / logarithm
    perpendicular = 4.0 * pi * dynamic_viscosity_pa_s / (logarithm + 0.5)
    return parallel, perpendicular


def liquid_medium_calibration(
    *,
    dynamic_viscosity_pa_s: float = 1.0e-3,
    density_kg_m3: float = 1000.0,
    wave_length_m: float = 1.5e-3,
    worm: AdultWormCalibration = AdultWormCalibration(),
) -> MediumCalibration:
    """Return an unbounded-Newtonian-liquid calibration (water/M9 default)."""

    parallel_per_length, perpendicular_per_length = lighthill_drag_per_length(
        dynamic_viscosity_pa_s, worm.body_radius_m, wave_length_m
    )
    return MediumCalibration(
        kind="liquid",
        density_kg_m3=density_kg_m3,
        dynamic_viscosity_pa_s=dynamic_viscosity_pa_s,
        parallel_drag_kg_s=parallel_per_length * worm.body_length_m,
        perpendicular_drag_kg_s=perpendicular_per_length * worm.body_length_m,
        drag_model="Lighthill local resistive-force theory",
    )


def agar_medium_calibration() -> MediumCalibration:
    """Return literature effective drag for an adult crawling on agar."""

    return MediumCalibration(
        kind="agar",
        density_kg_m3=1000.0,
        dynamic_viscosity_pa_s=None,
        parallel_drag_kg_s=3.2e-3,
        perpendicular_drag_kg_s=128.0e-3,
        drag_model="effective anisotropic agar surface drag",
    )


def reynolds_number(
    density_kg_m3: float,
    speed_m_s: float,
    characteristic_length_m: float,
    dynamic_viscosity_pa_s: float,
) -> float:
    """Return ``rho * U * ell / mu`` for a stated characteristic length."""

    if dynamic_viscosity_pa_s <= 0.0:
        raise ValueError("Dynamic viscosity must be positive")
    return (
        density_kg_m3
        * abs(speed_m_s)
        * characteristic_length_m
        / dynamic_viscosity_pa_s
    )


def physical_muscle_body_params(
    num_segments: int = 12,
    *,
    medium: MediumKind = "agar",
    worm: AdultWormCalibration = AdultWormCalibration(),
    liquid_viscosity_pa_s: float = 1.0e-3,
    liquid_wave_length_m: float = 1.5e-3,
) -> MuscleBodyParams:
    """Build a physical SI-valued adult body in liquid or on agar.

    Muscle moment is represented as an activation-dependent preferred
    curvature. Its scale is therefore tied to the measured bending modulus;
    it is not claimed as an independently measured muscle-force calibration.
    """

    if num_segments < 2:
        raise ValueError("An actuated body requires at least two segments")
    if medium == "liquid":
        environment = liquid_medium_calibration(
            dynamic_viscosity_pa_s=liquid_viscosity_pa_s,
            wave_length_m=liquid_wave_length_m,
            worm=worm,
        )
    elif medium == "agar":
        environment = agar_medium_calibration()
    else:
        raise ValueError(f"Unknown medium: {medium!r}")

    segment_length = worm.body_length_m / num_segments
    joint_stiffness = worm.bending_modulus_n_m2 / segment_length
    joint_damping = (
        worm.internal_viscosity_upper_bound_n_m2_s / segment_length
    )
    maximum_joint_angle = worm.maximum_curvature_per_m * segment_length
    return MuscleBodyParams(
        mechanics=BodyParams(
            segment_length=jnp.asarray(segment_length),
            parallel_drag=jnp.asarray(
                environment.parallel_drag_kg_s / worm.body_length_m
            ),
            perpendicular_drag=jnp.asarray(
                environment.perpendicular_drag_kg_s / worm.body_length_m
            ),
            solve_regularization=jnp.asarray(1.0e-7),
        ),
        bending_stiffness=jnp.asarray(joint_stiffness),
        bending_damping=jnp.asarray(joint_damping),
        muscle_moment_scale=jnp.asarray(joint_stiffness * maximum_joint_angle),
        activation_time_constant=jnp.asarray(worm.muscle_time_constant_s),
        max_joint_angle=jnp.asarray(maximum_joint_angle),
    )

