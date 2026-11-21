import jax.numpy as jnp
import numpy as np

from r_elegans.brain.all_parameters import (
    EXTRA_SCALE, ExtraOffsets, apply_extra, fixed_fields_equal_except_extra, offset_bounds, offsets_between, to_physical, zero_extra)
from r_elegans.brain.conductance_network import load_network

BUNDLE = 'r_elegans/assets/mac_rhythm_v1_polarized/network.npz'


def test_zero_offsets_are_identity_and_roundtrip():
    _, fixed, _ = load_network(BUNDLE)
    same = apply_extra(zero_extra(), fixed)
    assert fixed_fields_equal_except_extra(fixed, same)
    assert all(np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(jax_leaves(fixed), jax_leaves(same)))
    offsets = ExtraOffsets(*(jnp.full(302, .01 * (i + 1)) for i in range(6)))
    changed = apply_extra(offsets, fixed)
    back = offsets_between(changed, fixed)
    assert all(np.allclose(np.asarray(b), np.asarray(o), atol=1e-5) for b, o in zip(back, offsets))


def test_bounds_contain_zero_and_physical_limits():
    _, fixed, _ = load_network(BUNDLE)
    lower, upper = offset_bounds(fixed)
    assert all(np.all(l <= 0) and np.all(u >= 0) for l, u in zip(lower, upper))
    top = apply_extra(ExtraOffsets(*(jnp.asarray(u) for u in upper)), fixed)
    assert np.all(np.asarray(top.cells.capacitance_pF) <= 20. + 1e-4)
    assert np.all(np.asarray(top.release_threshold_mV) <= -20. + 1e-4)
    assert np.all(np.asarray(top.release_slope_mV) <= 20. + 1e-4)


def test_other_fixed_fields_are_detected():
    _, fixed, _ = load_network(BUNDLE)
    other = fixed._replace(bias_pA=fixed.bias_pA + 1.)
    assert not fixed_fields_equal_except_extra(fixed, other)
    scaled = to_physical(ExtraOffsets(*(jnp.ones(302) for _ in range(6))))
    assert np.allclose(np.asarray(scaled.leak_reversal_mV), EXTRA_SCALE.leak_reversal_mV)


def jax_leaves(tree):
    import jax
    return jax.tree_util.tree_leaves(tree)
