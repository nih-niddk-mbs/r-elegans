import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.brain import (
    SENSORY_LOCAL_INDICES,
    SUBCIRCUIT_NEURON_NAMES,
    NeuralParams,
    build_subcircuit_params,
    decode_subcircuit_params,
    integrate_neural_fixed_step,
    integrate_neural_state,
)
from r_elegans.data.connectome import load_connectome


def test_subcircuit_slices_real_anatomical_connectivity() -> None:
    connectome = load_connectome()
    raw = build_subcircuit_params(connectome)

    count = len(SUBCIRCUIT_NEURON_NAMES)
    assert raw.chemical_mask.shape == (count, count)
    assert raw.gap_mask.shape == (count, count)
    assert raw.leak_reversal.shape == (count,)

    # The bundled connectome must contain real edges among these 14 neurons;
    # if this ever fails, the chosen neuron set needs revisiting.
    assert jnp.sum(raw.chemical_mask) > 0
    assert jnp.sum(raw.gap_mask) > 0
    assert set(np.unique(np.asarray(raw.chemical_mask))) <= {0.0, 1.0}
    assert set(np.unique(np.asarray(raw.gap_mask))) <= {0.0, 1.0}


def test_build_subcircuit_params_rejects_unknown_neuron_name() -> None:
    connectome = load_connectome()
    try:
        build_subcircuit_params(connectome, neuron_names=("NOT_A_NEURON",))
    except ValueError as error:
        assert "NOT_A_NEURON" in str(error)
    else:
        raise AssertionError("expected a ValueError for an unknown neuron name")


def test_decode_subcircuit_params_keeps_time_constant_and_slope_positive() -> None:
    connectome = load_connectome()
    raw = build_subcircuit_params(connectome)
    decoded = decode_subcircuit_params(raw, tau_min=0.02, slope_min=0.02)

    assert isinstance(decoded, NeuralParams)
    assert jnp.all(decoded.time_constant > 0.02)
    assert jnp.all(decoded.slope > 0.02)
    # Decoding near-zero raw values should still clear the floor comfortably.
    driven_down = raw._replace(
        raw_time_constant=jnp.full_like(raw.raw_time_constant, -50.0),
        raw_slope=jnp.full_like(raw.raw_slope, -50.0),
    )
    still_valid = decode_subcircuit_params(driven_down, tau_min=0.02, slope_min=0.02)
    assert jnp.all(still_valid.time_constant >= 0.02)
    assert jnp.all(still_valid.slope >= 0.02)


def test_anatomical_masks_receive_exactly_zero_gradient() -> None:
    """Regression test for the dynamics.py stop_gradient fix.

    Once a full NeuralParams-shaped object lives inside an optimized pytree,
    the anatomical 0/1 masks must never receive nonzero gradient -- otherwise
    Adam would slowly corrupt fixed topology into arbitrary floats.
    """

    connectome = load_connectome()
    raw = build_subcircuit_params(connectome)
    count = len(SUBCIRCUIT_NEURON_NAMES)
    voltage = jnp.linspace(-0.5, 0.1, count)
    external = jnp.zeros((count,)).at[0].set(0.1)

    def loss(raw_params):
        params = decode_subcircuit_params(raw_params)
        next_voltage = integrate_neural_fixed_step(voltage, params, external, dt=0.02)
        return jnp.sum(next_voltage**2)

    gradient = jax.grad(loss)(raw)
    assert jnp.all(gradient.chemical_mask == 0.0)
    assert jnp.all(gradient.gap_mask == 0.0)
    # A sanity check that gradients elsewhere are *not* trivially all zero.
    assert jnp.any(gradient.raw_chemical != 0.0)


def test_fixed_step_integration_matches_adaptive_solver_loosely() -> None:
    connectome = load_connectome()
    raw = build_subcircuit_params(connectome)
    params = decode_subcircuit_params(raw)
    count = len(SUBCIRCUIT_NEURON_NAMES)
    voltage = jnp.full((count,), -0.35)
    external = jnp.zeros((count,)).at[jnp.asarray(SENSORY_LOCAL_INDICES)].set(0.1)

    fixed = integrate_neural_fixed_step(voltage, params, external, dt=0.02, substeps=8)
    adaptive = integrate_neural_state(voltage, params, external, duration=0.02)

    assert jnp.all(jnp.isfinite(fixed))
    np.testing.assert_allclose(np.asarray(fixed), np.asarray(adaptive), atol=5e-3)
