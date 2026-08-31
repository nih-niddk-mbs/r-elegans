"""Graded-potential neural dynamics implemented as pure JAX functions.

All connectivity matrices follow the convention ``[post, pre]``. Raw
chemical and electrical parameters are transformed with ``softplus`` so the
effective magnitudes remain nonnegative. Excitation and inhibition arise from
the presynaptic reversal potential rather than signed connection weights.
"""

from __future__ import annotations

from typing import NamedTuple

import diffrax
import jax
import jax.numpy as jnp

Array = jax.Array


class NeuralParams(NamedTuple):
    """Parameters for a connectome-constrained graded-potential network."""

    raw_chemical: Array
    chemical_mask: Array
    raw_gap: Array
    gap_mask: Array
    leak_reversal: Array
    synapse_reversal: Array
    time_constant: Array
    threshold: Array
    slope: Array


def effective_chemical_weights(params: NeuralParams) -> Array:
    """Return nonnegative chemical strengths restricted to the fixed mask."""

    return jax.nn.softplus(params.raw_chemical) * params.chemical_mask


def effective_gap_weights(params: NeuralParams) -> Array:
    """Return symmetric, nonnegative gap conductances under the fixed mask."""

    raw_symmetric = 0.5 * (params.raw_gap + params.raw_gap.T)
    mask_symmetric = params.gap_mask * params.gap_mask.T
    return jax.nn.softplus(raw_symmetric) * mask_symmetric


def neural_currents(
    voltage: Array,
    params: NeuralParams,
    external_current: Array,
) -> tuple[Array, Array, Array]:
    """Compute leak, electrical, and chemical current terms."""

    chemical = effective_chemical_weights(params)
    gap = effective_gap_weights(params)

    activation = jax.nn.sigmoid(
        (voltage - params.threshold) / params.slope
    )
    gap_current = jnp.sum(gap * (voltage[None, :] - voltage[:, None]), axis=1)
    chemical_current = jnp.sum(
        chemical
        * activation[None, :]
        * (params.synapse_reversal[None, :] - voltage[:, None]),
        axis=1,
    )
    leak_current = params.leak_reversal - voltage
    return leak_current + external_current, gap_current, chemical_current


def neural_rhs(
    time: Array,
    voltage: Array,
    args: tuple[NeuralParams, Array],
) -> Array:
    """Evaluate ``dV/dt`` for Diffrax or direct JAX use.

    ``time`` is accepted for the Diffrax vector-field interface. External
    currents are constant over one solver call; callers can make them
    time-dependent by defining a higher-level vector field.
    """

    del time
    params, external_current = args
    leak, gap, chemical = neural_currents(voltage, params, external_current)
    return (leak + gap + chemical) / params.time_constant


def integrate_neural_state(
    voltage: Array,
    params: NeuralParams,
    external_current: Array,
    duration: float | Array,
    *,
    initial_step_size: float = 1e-2,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> Array:
    """Integrate one neural interval and return only its final voltage."""

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(neural_rhs),
        diffrax.Tsit5(),
        t0=0.0,
        t1=duration,
        dt0=initial_step_size,
        y0=voltage,
        args=(params, external_current),
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=4096,
    )
    return solution.ys[0]

