"""Extra per-cell parameters (beyond channel densities and edge strengths) for all-parameter gradient fitting.

Offsets are measured from an anchor checkpoint's values and applied to its fixed fields. Positive quantities use log offsets.
Absolute bounds are the project's existing initialization clip limits (capacitance 1-20 pF, leak reversal -80..-35 mV,
release threshold -65..-20 mV, release slope 2..20 mV) plus an assumed factor-3 range for calcium removal time and free
calcium fraction; none of these bounds is measured. Gating kinetics, ion reversals, cell volume, calcium equilibrium, bias,
initial voltage, anatomy and edge sign/reversal stay fixed.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class ExtraOffsets(NamedTuple):
    log_capacitance: jax.Array
    leak_reversal_mV: jax.Array
    release_threshold_mV: jax.Array
    log_release_slope: jax.Array
    log_calcium_removal: jax.Array
    log_free_calcium: jax.Array


class AllParams(NamedTuple):
    network: object
    extra: ExtraOffsets


# Optimizer works on offset / scale so one learning rate gives comparable steps in log units and mV.
EXTRA_SCALE = ExtraOffsets(1., 5., 5., 1., 1., 1.)
FACTOR_RANGE = float(np.log(3.))


def zero_extra(n=302, dtype=float):
    return ExtraOffsets(*(jnp.zeros(n, dtype) for _ in ExtraOffsets._fields))


def apply_extra(extra, fixed):
    """Copy of the anchor `fixed` with the offsets applied; offsets are in physical units (not divided by EXTRA_SCALE)."""
    cells = fixed.cells._replace(
        capacitance_pF=fixed.cells.capacitance_pF * jnp.exp(extra.log_capacitance),
        leak_reversal_mV=fixed.cells.leak_reversal_mV + extra.leak_reversal_mV,
        calcium_removal_ms=fixed.cells.calcium_removal_ms * jnp.exp(extra.log_calcium_removal),
        free_calcium_fraction=fixed.cells.free_calcium_fraction * jnp.exp(extra.log_free_calcium))
    return fixed._replace(cells=cells, release_threshold_mV=fixed.release_threshold_mV + extra.release_threshold_mV,
                          release_slope_mV=fixed.release_slope_mV * jnp.exp(extra.log_release_slope))


def to_physical(u):
    return ExtraOffsets(*(a * s for a, s in zip(u, EXTRA_SCALE)))


def offset_bounds(fixed):
    """(lower, upper) physical offsets from the anchor, each containing 0."""
    c, s = np.asarray(fixed.cells.capacitance_pF), np.asarray(fixed.release_slope_mV)
    leak, thr = np.asarray(fixed.cells.leak_reversal_mV), np.asarray(fixed.release_threshold_mV)
    factor = np.full_like(c, FACTOR_RANGE)
    lower = ExtraOffsets(np.log(1. / c), -80. - leak, -65. - thr, np.log(2. / s), -factor, -factor)
    upper = ExtraOffsets(np.log(20. / c), -35. - leak, -20. - thr, np.log(20. / s), factor, factor)
    return (ExtraOffsets(*(np.minimum(x, 0.) for x in lower)), ExtraOffsets(*(np.maximum(x, 0.) for x in upper)))


def offsets_between(start, anchor):
    """Physical offsets that turn the anchor's extra fields into the start's."""
    ratio = lambda a, b: np.log(np.asarray(a, float) / np.asarray(b, float))
    return ExtraOffsets(ratio(start.cells.capacitance_pF, anchor.cells.capacitance_pF),
                        np.asarray(start.cells.leak_reversal_mV, float) - np.asarray(anchor.cells.leak_reversal_mV, float),
                        np.asarray(start.release_threshold_mV, float) - np.asarray(anchor.release_threshold_mV, float),
                        ratio(start.release_slope_mV, anchor.release_slope_mV),
                        ratio(start.cells.calcium_removal_ms, anchor.cells.calcium_removal_ms),
                        ratio(start.cells.free_calcium_fraction, anchor.cells.free_calcium_fraction))


def fixed_fields_equal_except_extra(a, b):
    """True when two fixed-field sets differ only in the fields that `apply_extra` can change."""
    def neutral(fixed):
        zero = lambda x: jnp.zeros_like(x)
        cells = fixed.cells._replace(capacitance_pF=zero(fixed.cells.capacitance_pF), leak_reversal_mV=zero(fixed.cells.leak_reversal_mV),
                                     calcium_removal_ms=zero(fixed.cells.calcium_removal_ms),
                                     free_calcium_fraction=zero(fixed.cells.free_calcium_fraction))
        return fixed._replace(cells=cells, release_threshold_mV=zero(fixed.release_threshold_mV), release_slope_mV=zero(fixed.release_slope_mV))
    return all(np.array_equal(np.asarray(x), np.asarray(y))
               for x, y in zip(jax.tree_util.tree_leaves(neutral(a)), jax.tree_util.tree_leaves(neutral(b))))
