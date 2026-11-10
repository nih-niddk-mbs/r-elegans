"""Independent force/unit, dissipation and linear-reference mechanics checks."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from r_elegans.body import physical_muscle_body_params, torque_driven_body_velocity
from r_elegans.body.mechanics import generalized_resistance_matrix
from r_elegans.body.wet_surface import (
    drag_terms, linear_drag, quadrature_kinematics, replay_surface_commands,
    solve_surface_velocity, wet_agar_drag,
)


def test_measured_whole_worm_towing_force_and_length_scaling():
    law = wet_agar_drag(1.7, regularization_speed_m_s=1e-10)
    # Published force fits use micrometers/s and microNewtons for a 1mm worm.
    force, _, _ = drag_terms(np.array([100e-6, 100e-6]), law)
    np.testing.assert_allclose(force*1e-3, np.array([.32*100**.58, 6.8*100**.31])*1e-6, rtol=1e-10)
    jacobian, weights = quadrature_kinematics(np.zeros(11), 1e-3/12)
    v = np.r_[100e-6, 0., np.zeros(12)]
    force, _, _ = drag_terms(np.einsum('paj,j->pa', jacobian, v), law)
    generalized = np.einsum('paj,pa,p->j', jacobian, force, weights)
    np.testing.assert_allclose(generalized[0], .32*100**.58*1e-6, rtol=1e-10)
    np.testing.assert_allclose(generalized[1:3], 0., atol=1e-18)
    # Halving the contact length halves the force; coefficients aren't per rod.
    np.testing.assert_allclose(weights.sum(), .001, rtol=1e-14)


def test_surface_resistance_is_passive_odd_and_has_consistent_derivatives():
    law = wet_agar_drag()
    velocity = np.array([[0., 0.], [2e-7, -3e-7], [1e-4, -2e-4]])
    force, slope, potential = drag_terms(velocity, law)
    assert np.all(slope > 0) and np.all(force*velocity >= 0) and np.all(potential >= 0)
    np.testing.assert_allclose(drag_terms(-velocity, law)[0], -force)
    h = 1e-10
    for axis in range(2):
        delta = np.eye(2)[axis]*h
        left, right = drag_terms(velocity-delta, law), drag_terms(velocity+delta, law)
        np.testing.assert_allclose((right[0][:, axis]-left[0][:, axis])/(2*h), slope[:, axis], rtol=1e-6)
        np.testing.assert_allclose((right[2][:, axis]-left[2][:, axis])/(2*h), force[:, axis], rtol=1e-6, atol=1e-16)


def test_linear_limit_matches_independent_jax_resistance_and_torque_solve():
    with jax.experimental.enable_x64():
        body = physical_muscle_body_params(12, medium='liquid')
        body = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), body)
        q = .1*np.sin(np.arange(11)*.6)
        moments = 1e-10*np.cos(np.arange(11)*.5)
        law = linear_drag(float(body.mechanics.parallel_drag), float(body.mechanics.perpendicular_drag))
        jacobian, weights = quadrature_kinematics(q, float(body.mechanics.segment_length))
        drag = np.array([float(body.mechanics.parallel_drag), float(body.mechanics.perpendicular_drag)])
        resistance = np.einsum('pai,a,paj,p->ij', jacobian, drag, jacobian, weights)
        expected = generalized_resistance_matrix(jnp.asarray(q), body.mechanics)
        np.testing.assert_allclose(resistance, expected, rtol=1e-11, atol=1e-20)
        velocity, info = solve_surface_velocity(q, moments, body, law, .002)
        rigid, rates = torque_driven_body_velocity(jnp.asarray(q), jnp.asarray(moments),
            body.mechanics, body.bending_stiffness, body.bending_damping, .002)
        length = float(body.mechanics.segment_length)*12
        reference = np.r_[np.asarray(rigid)[:2], float(rigid[2])*length, np.asarray(rates)*length]
        # Existing solver adds relative diagonal regularization (1e-7).
        np.testing.assert_allclose(velocity, reference, rtol=3e-5, atol=1e-11)
        assert info['relative_force_residual'] <= 1e-8


def test_nonlinear_passive_relaxation_and_zero_command_quiet_control():
    body = physical_muscle_body_params(12, medium='agar')
    law = wet_agar_drag()
    q = .01*np.sin(np.arange(11)*.5)
    energy = .5*float(body.bending_stiffness)*np.dot(q, q)
    for _ in range(8):
        velocity, info = solve_surface_velocity(q, np.zeros(11), body, law, .002)
        q = q+.002*velocity[3:]/(float(body.mechanics.segment_length)*12)
        next_energy = .5*float(body.bending_stiffness)*np.dot(q, q)
        assert next_energy <= energy and info['relative_force_residual'] <= 1e-8
        energy = next_energy
    trace = replay_surface_commands(np.zeros((25, 11)), body, law, .002)
    np.testing.assert_array_equal(trace['joint_angles'], 0.)
    np.testing.assert_array_equal(trace['position'], 0.)
    with pytest.raises(ValueError):
        wet_agar_drag(2.0)
    with pytest.raises(ValueError):
        wet_agar_drag(regularization_speed_m_s=0)
