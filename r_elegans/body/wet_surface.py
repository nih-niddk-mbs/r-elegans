"""Float64 reference mechanics for a measured, effective wet-agar drag law.

Rabets et al. (2014), doi:10.1016/j.bpj.2014.09.006, Methods p1982:
F [microN, normalized to a 1mm worm] = a * |v [micrometer/s]|**gamma.
The drag acts opposite velocity. This is an empirical local closure for the
specified laboratory agar, not resolved capillarity, gel memory or natural soil.
Near-zero regularization is an explicit modeling assumption, not measured data.

This independent NumPy solver is an assessment reference, not a replacement
for existing JAX training environments. Angular velocities use the same body-
length scaling as mechanics.py. Gauss integration includes finite-rod rotation.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


SOURCE_DOI = "10.1016/j.bpj.2014.09.006"
# concentration: (a_t, a_n, gamma_t, gamma_n), published fits, not raw data.
RABETS_FITS = {
    1.0: (.35, 4.6, .55, .27),
    1.7: (.32, 6.8, .58, .31),
    3.0: (.26, 7.9, .68, .30),
    4.0: (.24, 7.3, .77, .34),
    6.0: (.20, 5.8, .81, .38),
}


@dataclass(frozen=True)
class SurfaceDrag:
    """Force per length at 1um/s, power exponents and regularization speed."""

    force_at_um_s_n_m: tuple[float, float]
    exponents: tuple[float, float]
    regularization_speed_m_s: float
    name: str

    def __post_init__(self):
        if (len(self.force_at_um_s_n_m) != 2 or len(self.exponents) != 2
                or not np.all(np.isfinite((*self.force_at_um_s_n_m,
                                           *self.exponents, self.regularization_speed_m_s)))
                or min(self.force_at_um_s_n_m) <= 0
                or min(self.exponents) <= 0 or max(self.exponents) > 1
                or self.regularization_speed_m_s <= 0):
            raise ValueError("Positive finite coefficients, 0<gamma<=1 and positive speed required")


def wet_agar_drag(concentration_percent=1.7, *, regularization_speed_m_s=1e-7):
    """Published wet NGM fits; no interpolation between agar concentrations."""
    if concentration_percent not in RABETS_FITS:
        raise ValueError("Choose a measured agar concentration: " + str(tuple(RABETS_FITS)))
    at, an, gt, gn = RABETS_FITS[concentration_percent]
    # microN / reference 1mm -> N/m, not drag for an individual rod.
    return SurfaceDrag((at * 1e-3, an * 1e-3), (gt, gn),
                       regularization_speed_m_s, f"Rabets2014-{concentration_percent:g}percent")


def linear_drag(parallel_per_length, perpendicular_per_length, *, name="linear"):
    return SurfaceDrag((parallel_per_length * 1e-6, perpendicular_per_length * 1e-6),
                       (1., 1.), 1e-7, name)


def drag_terms(velocity, law):
    """Return signed resistance/length, derivative and dissipation potential.

    Last axis is [tangential, normal]. Resistance has the sign of velocity;
    the physical environmental force is its negative. Smooth odd resistance
    is a*v*(v*v+eps*eps)**((gamma-1)/2) in declared micrometer units.
    Potential's velocity derivative is resistance; Hessian stays positive.
    """
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape[-1:] != (2,) or not np.isfinite(velocity).all():
        raise ValueError("Expected finite tangential/normal velocities")
    r = velocity / 1e-6
    eps = law.regularization_speed_m_s / 1e-6
    a, g = np.asarray(law.force_at_um_s_n_m), np.asarray(law.exponents)
    squared = r*r + eps*eps
    force = a * r * squared**((g-1)/2)
    derivative = (a/1e-6) * squared**((g-3)/2) * (eps*eps + g*r*r)
    # log1p/expm1 avoids subtracting almost equal powers near rest.
    potential = a*1e-6/(g+1) * eps**(g+1) * np.expm1(
        (g+1)/2 * np.log1p((r/eps)**2))
    return force, derivative, potential


@lru_cache(maxsize=8)
def _gauss_rule(order):
    return np.polynomial.legendre.leggauss(order)


def quadrature_kinematics(joint_angles, segment_length, *, order=4):
    """Local-velocity Jacobians and arclength weights at Gauss rod points."""
    q = np.asarray(joint_angles, dtype=float)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or not np.isfinite(segment_length) or segment_length <= 0:
        raise ValueError("Finite joint vector and positive segment length required")
    if order < 2:
        raise ValueError("At least two quadrature points per finite rod required")
    n = len(q)+1
    length = segment_length*n
    angle = np.r_[0., np.cumsum(q)]
    tangent = np.column_stack((np.cos(angle), np.sin(angle)))
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    rotation = np.tril(np.ones((n, n-1)), k=-1)
    segment_derivative = segment_length * normal[:, :, None] * rotation[:, None, :]
    starts = np.vstack((np.zeros(2), np.cumsum(segment_length*tangent[:-1], axis=0)))
    centers = starts + .5*segment_length*tangent
    centers -= centers.mean(axis=0)
    center_derivative = np.concatenate((np.zeros((1, 2, n-1)),
        np.cumsum(segment_derivative[:-1], axis=0)), axis=0) + .5*segment_derivative
    center_derivative -= center_derivative.mean(axis=0)
    abscissa, weights = _gauss_rule(order)
    offsets = .5*segment_length*abscissa
    points = centers[:, None, :] + offsets[None, :, None]*tangent[:, None, :]
    shape = center_derivative[:, None, :, :] + (
        offsets[None, :, None, None]*normal[:, None, :, None]*rotation[:, None, None, :])
    rigid_rotation = np.stack((-points[..., 1], points[..., 0]), axis=-1)[..., None]
    jacobian = np.concatenate((np.broadcast_to(np.eye(2), (n, order, 2, 2)),
                              rigid_rotation/length, shape/length), axis=-1)
    axes = np.stack((tangent, normal), axis=1)
    local = np.einsum("nac,nkcj->nkaj", axes, jacobian).reshape(-1, 2, n+2)
    integration_weights = np.broadcast_to(.5*segment_length*weights, (n, order)).ravel()
    return local, integration_weights


def solve_surface_velocity(q, active_moments, body, law, dt, *, initial_velocity=None,
                           quadrature_order=4, tolerance=1e-8, maximum_iterations=60):
    """Solve free rigid motion and torque-driven shape with damped Newton.

    Geometry is evaluated at old q; passive elastic torque uses q+dt*qdot.
    No velocity/force clipping or hidden fallback. An unconverged solve raises.
    Returns scaled generalized velocity and dimensionless force residual.
    """
    q = np.asarray(q, dtype=float)
    moments = np.asarray(active_moments, dtype=float)
    if moments.shape != q.shape or not np.isfinite(moments).all() or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Finite joint moments and positive timestep required")
    ds = float(body.mechanics.segment_length)
    length = ds*(len(q)+1)
    local, weights = quadrature_kinematics(q, ds, order=quadrature_order)
    stiffness, damping = float(body.bending_stiffness), float(body.bending_damping)
    diagonal = np.r_[np.zeros(3), np.full(len(q), (damping+dt*stiffness)/length**2)]
    rhs = np.r_[np.zeros(3), (moments-stiffness*q)/length]
    v = np.zeros(len(q)+3) if initial_velocity is None else np.asarray(initial_velocity, dtype=float).copy()
    if v.shape != rhs.shape or not np.isfinite(v).all():
        raise ValueError("Invalid initial velocity")
    scale = max(np.linalg.norm(rhs, ord=np.inf), 1e-15)

    def terms(x, hessian=True):
        speed = np.einsum("paj,j->pa", local, x)
        force, derivative, potential = drag_terms(speed, law)
        gradient = np.einsum("paj,pa,p->j", local, force, weights) + diagonal*x-rhs
        value = float(np.sum(weights[:, None]*potential) + .5*np.dot(diagonal*x, x)-np.dot(rhs, x))
        matrix = None
        if hessian:
            matrix = np.einsum("pai,pa,paj,p->ij", local, derivative, local, weights)
            matrix += np.diag(diagonal)
        return value, gradient, matrix, speed

    for iteration in range(maximum_iterations+1):
        value, gradient, matrix, speed = terms(v)
        residual = float(np.linalg.norm(gradient, ord=np.inf)/scale)
        if residual <= tolerance:
            return v, dict(relative_force_residual=residual, newton_iterations=iteration,
                maximum_point_speed_m_s=float(np.linalg.norm(speed, axis=1).max()),
                fraction_components_below_regularization=float(np.mean(np.abs(speed) < law.regularization_speed_m_s)))
        if iteration == maximum_iterations:
            break
        direction = np.linalg.solve(matrix, -gradient)
        slope = float(gradient @ direction)
        fraction = 1.
        for _ in range(32):
            trial = v+fraction*direction
            trial_value, trial_gradient, _, _ = terms(trial, hessian=False)
            # Either Armijo decrease or residual convergence at roundoff scale.
            if (trial_value <= value + 1e-4*fraction*slope or
                    np.linalg.norm(trial_gradient, ord=np.inf) <= tolerance*scale):
                v = trial
                break
            fraction *= .5
        else:
            raise RuntimeError(f"Surface Newton line search failed: relative residual {residual:g}")
    raise RuntimeError(f"Surface force balance did not converge: relative residual {residual:g}")


def replay_surface_commands(commands, body, law, dt, *, substeps=1, quadrature_order=4, progress=None):
    """Replay endogenous dorsal-minus-ventral commands from a one-way network.

    All bodies begin straight and relaxed. Commands are held over each source
    interval. Substeps refine body and muscle kinetics only, not neural dynamics.
    Float64 reference solver is intentionally separate from the training path.
    """
    commands = np.asarray(commands, dtype=float)
    if (commands.ndim != 2 or commands.shape[1] < 1 or not len(commands)
            or not np.isfinite(commands).all() or np.max(np.abs(commands)) > 1+1e-6
            or substeps not in (1, 2, 4) or not np.isfinite(dt) or dt <= 0):
        raise ValueError("Finite bounded joint commands, positive dt and substeps 1/2/4 required")
    samples, joints = commands.shape
    q, drive, position, heading = np.zeros(joints), np.zeros(joints), np.zeros(2), 0.
    v = np.zeros(joints+3)
    trace = dict(joint_angles=np.empty_like(commands), muscle_activation=np.empty_like(commands),
        position=np.empty((samples, 2)), heading=np.empty(samples),
        relative_force_residual=np.empty(samples), maximum_point_speed_m_s=np.empty(samples),
        fraction_components_below_regularization=np.empty(samples), newton_iterations=np.empty(samples, dtype=int))
    step = dt/substeps
    activation_fraction = -np.expm1(-step/float(body.activation_time_constant))
    length = float(body.mechanics.segment_length)*(joints+1)
    for i, command in enumerate(commands):
        diagnostics = []
        for _ in range(substeps):
            drive += activation_fraction*(command-drive)
            v, info = solve_surface_velocity(q, float(body.muscle_moment_scale)*drive,
                body, law, step, initial_velocity=v, quadrature_order=quadrature_order)
            next_q = q+step*v[3:]/length
            if np.max(np.abs(next_q)) >= float(body.max_joint_angle):
                raise RuntimeError("Reference assay reached joint limit; constrained contact solve is required")
            c, s = np.cos(heading), np.sin(heading)
            position += step*np.array(((c, -s), (s, c))) @ v[:2]
            heading += step*v[2]/length
            q = next_q
            diagnostics.append(info)
        trace['joint_angles'][i], trace['muscle_activation'][i] = q, drive
        trace['position'][i], trace['heading'][i] = position, heading
        for key in ('relative_force_residual', 'maximum_point_speed_m_s', 'newton_iterations'):
            trace[key][i] = max(row[key] for row in diagnostics)
        trace['fraction_components_below_regularization'][i] = np.mean(
            [row['fraction_components_below_regularization'] for row in diagnostics])
        if progress is not None and ((i+1) % 1000 == 0 or i+1 == samples):
            progress(i+1, samples)
    if not all(np.isfinite(value).all() for value in trace.values()):
        raise RuntimeError("Nonfinite surface trajectory")
    return trace


def surface_body_step(state, signed_command, body, law, dt, *, initial_velocity=None):
    """One reference body step for genuinely interleaved neural feedback.

    Unlike replay_surface_commands, this takes a fresh endogenous command each
    step, so the resulting shape can feed back before the next neural update.
    Returns a NumPy-valued MuscleBodyState and the solver's warm-start velocity.
    """
    from .actuation import MuscleBodyState
    command = np.asarray(signed_command, dtype=float)
    q = np.asarray(state.joint_angles, dtype=float)
    if (command.shape != q.shape or not np.isfinite(command).all()
            or np.max(np.abs(command)) > 1+1e-6 or not np.isfinite(dt) or dt <= 0):
        raise ValueError('Bounded finite command and positive body timestep required')
    drive = np.asarray(state.muscle_activation, dtype=float)
    drive = drive+(-np.expm1(-dt/float(body.activation_time_constant)))*(command-drive)
    velocity, info = solve_surface_velocity(q, float(body.muscle_moment_scale)*drive,
        body, law, dt, initial_velocity=initial_velocity)
    length = float(body.mechanics.segment_length)*(len(q)+1)
    next_q = q+dt*velocity[3:]/length
    if np.max(np.abs(next_q)) >= float(body.max_joint_angle):
        raise RuntimeError('Reference body reached joint limit; constrained solver required')
    heading = float(state.heading)
    c, s = np.cos(heading), np.sin(heading)
    position = np.asarray(state.position)+dt*np.array(((c, -s), (s, c))) @ velocity[:2]
    return MuscleBodyState(position, np.asarray(heading+dt*velocity[2]/length),
        next_q, drive, np.asarray(float(state.time)+dt)), velocity, info
