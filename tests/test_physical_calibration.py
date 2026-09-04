import jax.numpy as jnp
import numpy as np

from r_elegans.body import (
    AdultWormCalibration,
    agar_medium_calibration,
    body_velocity,
    generalized_resistance_matrix,
    initialize_muscle_body,
    lighthill_drag_per_length,
    liquid_medium_calibration,
    muscle_body_step,
    physical_muscle_body_params,
    reynolds_number,
)


def test_lighthill_water_coefficients_have_expected_anisotropy() -> None:
    parallel, perpendicular = lighthill_drag_per_length(
        1.0e-3, 40.0e-6, 1.5e-3
    )

    assert parallel > 0.0
    assert 1.4 < perpendicular / parallel < 1.7


def test_medium_calibrations_use_whole_worm_drag_units() -> None:
    water = liquid_medium_calibration()
    agar = agar_medium_calibration()

    assert 1.4 < water.drag_ratio < 1.7
    np.testing.assert_allclose(agar.parallel_drag_kg_s, 3.2e-3)
    np.testing.assert_allclose(agar.perpendicular_drag_kg_s, 128.0e-3)
    np.testing.assert_allclose(agar.drag_ratio, 40.0)
    assert agar.perpendicular_drag_kg_s > 1.0e4 * water.perpendicular_drag_kg_s


def test_adult_physical_parameters_are_in_si_units() -> None:
    worm = AdultWormCalibration()
    params = physical_muscle_body_params(12, medium="liquid", worm=worm)

    np.testing.assert_allclose(
        params.mechanics.segment_length, worm.body_length_m / 12.0
    )
    np.testing.assert_allclose(
        params.bending_stiffness,
        worm.bending_modulus_n_m2 / (worm.body_length_m / 12.0),
    )
    np.testing.assert_allclose(
        params.max_joint_angle,
        worm.maximum_curvature_per_m * worm.body_length_m / 12.0,
    )


def test_generalized_resistance_is_symmetric_positive_definite() -> None:
    params = physical_muscle_body_params(12, medium="liquid")
    angles = 0.2 * jnp.sin(jnp.linspace(0.0, 2.0 * jnp.pi, 11))
    resistance = generalized_resistance_matrix(angles, params.mechanics)
    eigenvalues = np.linalg.eigvalsh(np.asarray(resistance))

    np.testing.assert_allclose(resistance, resistance.T, rtol=1e-5, atol=1e-12)
    assert np.all(eigenvalues > 0.0)


def test_prescribed_shape_motion_has_zero_net_force_and_torque() -> None:
    params = physical_muscle_body_params(12, medium="liquid")
    angles = 0.2 * jnp.sin(jnp.linspace(0.0, 2.0 * jnp.pi, 11))
    rates = 0.8 * jnp.cos(jnp.linspace(0.0, 2.0 * jnp.pi, 11))
    velocity = body_velocity(angles, rates, params.mechanics)
    body_length = params.mechanics.segment_length * 12
    scaled_velocity = jnp.concatenate(
        (velocity[:2], (body_length * velocity[2])[None], body_length * rates)
    )
    resistance = generalized_resistance_matrix(angles, params.mechanics)
    rigid_residual = resistance[:3] @ scaled_velocity

    np.testing.assert_allclose(rigid_residual, jnp.zeros(3), atol=1e-9)


def test_external_load_changes_torque_driven_bending_rate() -> None:
    water = physical_muscle_body_params(12, medium="liquid")
    agar = physical_muscle_body_params(12, medium="agar")
    state = initialize_muscle_body(12)
    dorsal = jnp.ones((11,))
    ventral = jnp.zeros((11,))

    water_next = muscle_body_step(state, water, dorsal, ventral, 0.001)
    agar_next = muscle_body_step(state, agar, dorsal, ventral, 0.001)

    assert jnp.linalg.norm(water_next.joint_angles) > jnp.linalg.norm(
        agar_next.joint_angles
    )


def test_reynolds_number_requires_declared_length_scale() -> None:
    # 0.45 mm/s in water: radius-based Re is small; length-based Re remains < 1.
    radius_re = reynolds_number(1000.0, 0.45e-3, 40.0e-6, 1.0e-3)
    length_re = reynolds_number(1000.0, 0.45e-3, 1.0e-3, 1.0e-3)

    np.testing.assert_allclose(radius_re, 0.018)
    np.testing.assert_allclose(length_re, 0.45)
