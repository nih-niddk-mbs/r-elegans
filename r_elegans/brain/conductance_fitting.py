"""Physical-unit clamp prediction and constrained conductance inference.

Kinetics/reversal potentials are fixed. A fitted maximal conductance is a
conditional whole-cell estimate, not a molecular channel count. Synaptic fits
require residual postsynaptic currents after intrinsic/capacitive subtraction;
calcium traces or wiring counts alone cannot be passed off as these recordings.
"""
import numpy as np
import jax
import jax.numpy as jnp

from .single_compartment import (
    SingleCompartmentState, initial_single_compartment_state,
    ionic_currents, single_compartment_rhs, _gate_targets_and_taus,
)


def voltage_clamp_trace(params, voltage_mV, dt_ms, *, holding_mV=-60., substeps=8):
    """Predict ionic current [time] in pA for piecewise constant voltage.

    Samples are interval-end currents. Exponential gate/calcium updates with
    frozen targets are first order; double substeps to test convergence.
    No capacitive or series-resistance current is modeled. Initial gates are
    equilibrated at holding voltage and bulk calcium starts at its preset.
    """
    state = initial_single_compartment_state(holding_mV, params)
    h = dt_ms/substeps

    def sample(state, voltage):
        def advance(s, _):
            s = s._replace(voltage_mV=voltage)
            target, tau = _gate_targets_and_taus(voltage, s.calcium_uM, s.gates)
            derivative = single_compartment_rhs(0., s, (params, jnp.array(0.)))
            gates = s.gates + (target-s.gates) * (-jnp.expm1(-h/tau))
            # Exact solution for influx frozen over this substep plus removal.
            ca_target = s.calcium_uM + derivative.calcium_uM * params.calcium_removal_ms
            ca = s.calcium_uM + (ca_target-s.calcium_uM) * (-jnp.expm1(-h/params.calcium_removal_ms))
            return SingleCompartmentState(voltage, gates, ca), None
        state = jax.lax.scan(advance, state, None, length=substeps)[0]
        return state, jnp.sum(ionic_currents(state, params))

    return jax.lax.scan(sample, state, voltage_mV)[1]


def sensitivity_diagnostics(jacobian):
    """Data-only, column-normalized sensitivity; not confidence intervals."""
    original = np.asarray(jacobian)
    precision = np.finfo(original.dtype).eps if np.issubdtype(original.dtype,np.floating) else np.finfo(float).eps
    matrix = np.asarray(original, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Finite matrix required for sensitivity diagnostics")
    norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix/np.maximum(norms, 1e-30)
    singular = np.linalg.svd(normalized, compute_uv=False)
    tolerance = max(normalized.shape) * precision * (singular[0] if singular.size else 0.)
    rank = int(np.sum(singular > tolerance))
    return dict(column_norms=norms.tolist(), singular_values=singular.tolist(),
                rank=rank, parameters=int(matrix.shape[1]), rank_tolerance=float(tolerance),
                practical_rank_relative_1e3=int(np.sum(singular > .001*singular[0])) if singular.size else 0,
                condition_number=(float(singular[0]/singular[-1])
                    if rank == matrix.shape[1] and singular.size else None),
                interpretation="Local fixed-kinetics sensitivity, not parameter uncertainty")


def fit_clamp_conductances(predict, target, prior_nS, *, log_prior_weight=.001,
                           max_evaluations=100):
    """Nonnegative conductances, fixed prior, explicit data/prior residuals.

    ``predict(g)`` returns training ionic currents in pA. Whole voltage steps
    must be withheld by the caller, never included in this target or the prior.
    """
    from scipy.optimize import least_squares
    target = np.asarray(target, dtype=float)
    prior = np.maximum(np.asarray(prior_nS, dtype=float), .01)
    if (not np.isfinite(target).all() or not np.isfinite(prior).all()
            or log_prior_weight < 0 or max_evaluations < 1):
        raise ValueError("Invalid fitting inputs")
    scale = max(float(np.sqrt(np.mean(target**2))), 1.)
    log_prior = jnp.log(jnp.asarray(prior))

    def data_residual(log_g):
        return (predict(jnp.exp(log_g)).ravel()-jnp.asarray(target).ravel())/scale

    def residual(log_g):
        return jnp.concatenate((data_residual(log_g)/np.sqrt(target.size),
            jnp.sqrt(log_prior_weight) * (log_g-log_prior)/np.sqrt(prior.size)))

    fun = jax.jit(residual)
    jac = jax.jit(jax.jacfwd(residual))
    result = least_squares(lambda x: np.asarray(fun(x), dtype=float), np.log(prior),
        jac=lambda x: np.asarray(jac(x), dtype=float), bounds=(np.log(1e-6), np.log(100.)),
        max_nfev=max_evaluations, ftol=1e-6, xtol=1e-6, gtol=1e-6)
    diagnostics = sensitivity_diagnostics(jax.jit(jax.jacfwd(data_residual))(jnp.asarray(result.x)))
    return np.exp(result.x), dict(optimizer_success=bool(result.success),
        message=result.message, evaluations=int(result.nfev), data_scale_pA=scale,
        log_prior_weight=log_prior_weight, sensitivity=diagnostics)


def synaptic_design(voltage_mV, chemical_mask, gap_mask, reversal_mV,
                     threshold_mV, slope_mV):
    """Sparse design for outward residual current [time, post] in pA.

    Chemical masks use [post,pre]. Each undirected gap pair has one conductance.
    Release is a fixed sigmoid; no kinetics or receptor signs are inferred.
    All cell voltages must be provided. Unknown reversal potentials are an error.
    Output column metadata identifies anatomical edges; absent edges never fit.
    """
    from scipy.sparse import coo_matrix
    from scipy.special import expit
    v = np.asarray(voltage_mV, dtype=float)
    chemical, gap = np.asarray(chemical_mask), np.asarray(gap_mask)
    reversal = np.asarray(reversal_mV, dtype=float)
    if v.ndim != 2 or min(v.shape) < 1:
        raise ValueError("Voltage must have shape [time, neuron]")
    t, n = v.shape
    if any(x.shape != (n,n) for x in (chemical,gap,reversal)):
        raise ValueError("Topology/reversal shapes must match neuron count")
    if not np.array_equal(gap, gap.T) or np.any(np.diag(gap)):
        raise ValueError("Gap topology must be symmetric without self edges")
    threshold, slope = np.asarray(threshold_mV), np.asarray(slope_mV)
    if (threshold.shape != (n,) or slope.shape != (n,) or not np.isfinite(v).all()
            or not np.isfinite(threshold).all() or not np.isfinite(slope).all()
            or np.any(slope <= 0) or not np.isfinite(reversal[chemical != 0]).all()):
        raise ValueError("Known finite voltages/reversals and positive release slopes required")
    release = expit((v-threshold)/slope)
    rows, cols, values, edges = [], [], [], []

    def append(post, column, coefficient):
        rows.extend((np.arange(t)*n+post).tolist())
        cols.extend([column]*t)
        values.extend(coefficient.tolist())

    for post, pre in zip(*np.nonzero(chemical)):
        column = len(edges)
        edges.append(("chemical", int(post), int(pre)))
        append(post, column, release[:,pre]*(v[:,post]-reversal[post,pre]))
    for first, second in zip(*np.nonzero(np.triu(gap, 1))):
        column = len(edges)
        edges.append(("gap", int(first), int(second)))
        difference = v[:,first]-v[:,second]
        append(first, column, difference)
        append(second, column, -difference)
    if not edges:
        raise ValueError("No anatomical edges to fit")
    return coo_matrix((values,(rows,cols)), shape=(t*n,len(edges))).tocsr(), edges


def fit_synaptic_conductances(design, residual_current_pA, *, prior_nS=None,
                              prior_weight=0., current_scale_pA=1.):
    """Fit nonnegative edge nS to intrinsic-subtracted clamp current.

    A regularized solution is conditional on its prior. Direct recovery requires
    sufficient independent stimulation, known intrinsic currents, and fixed
    release/reversal parameters. Caller must reserve separate test trials.
    """
    from scipy.optimize import lsq_linear
    from scipy.sparse import eye, vstack
    target = np.asarray(residual_current_pA, dtype=float).ravel()
    if (target.size != design.shape[0] or not np.isfinite(target).all()
            or prior_weight < 0 or not np.isfinite(prior_weight)
            or current_scale_pA <= 0 or not np.isfinite(current_scale_pA)):
        raise ValueError("Invalid residual-current fitting inputs")
    matrix, rhs = design/current_scale_pA, target/current_scale_pA
    if prior_weight:
        prior = np.asarray(prior_nS, dtype=float)
        if prior.shape != (design.shape[1],) or not np.isfinite(prior).all() or np.any(prior < 0):
            raise ValueError("Regularization requires a finite nonnegative edge prior")
        matrix = vstack((matrix/np.sqrt(target.size), np.sqrt(prior_weight)*eye(design.shape[1])))
        rhs = np.concatenate((rhs/np.sqrt(target.size), np.sqrt(prior_weight)*prior))
    fit = lsq_linear(matrix, rhs, bounds=(0., np.inf), tol=1e-10, lsmr_tol=1e-12)
    return fit.x, dict(optimizer_success=bool(fit.success), message=fit.message,
        train_rmse_pA=float(np.sqrt(np.mean((design@fit.x-target)**2))),
        prior_weight=prior_weight, current_scale_pA=current_scale_pA,
        interpretation="Effective conductances conditional on fixed intrinsic/release/reversal models")
