import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.brain import (
    NeuralParams,
    effective_chemical_weights,
    effective_gap_weights,
    neural_rhs,
)


def neural_fixture(neurons: int = 6) -> NeuralParams:
    chemical_mask = jnp.eye(neurons, k=-1) + jnp.eye(neurons, k=neurons - 1)
    gap_mask = jnp.eye(neurons, k=1) + jnp.eye(neurons, k=-1)
    return NeuralParams(
        raw_chemical=jnp.full((neurons, neurons), -2.0),
        chemical_mask=chemical_mask,
        raw_gap=jnp.full((neurons, neurons), -3.0),
        gap_mask=gap_mask,
        leak_reversal=jnp.full((neurons,), -0.35),
        synapse_reversal=jnp.linspace(-0.8, 0.2, neurons),
        time_constant=jnp.full((neurons,), 0.1),
        threshold=jnp.full((neurons,), -0.2),
        slope=jnp.full((neurons,), 0.1),
    )


def test_effective_weights_respect_masks_and_gap_symmetry() -> None:
    params = neural_fixture()
    chemical = effective_chemical_weights(params)
    gap = effective_gap_weights(params)

    np.testing.assert_array_equal(chemical != 0, params.chemical_mask != 0)
    np.testing.assert_array_equal(gap != 0, params.gap_mask != 0)
    np.testing.assert_allclose(gap, gap.T)
    assert jnp.all(chemical >= 0)
    assert jnp.all(gap >= 0)


def test_rhs_is_finite_and_jittable() -> None:
    params = neural_fixture()
    voltage = jnp.linspace(-0.5, 0.1, 6)
    external = jnp.zeros_like(voltage).at[0].set(0.1)

    derivative = jax.jit(neural_rhs)(0.0, voltage, (params, external))

    assert derivative.shape == voltage.shape
    assert jnp.all(jnp.isfinite(derivative))


def test_rhs_supports_batched_voltages() -> None:
    params = neural_fixture()
    voltages = jnp.stack((jnp.full((6,), -0.4), jnp.full((6,), -0.2)))
    external = jnp.zeros((6,))

    derivatives = jax.vmap(lambda voltage: neural_rhs(0.0, voltage, (params, external)))(
        voltages
    )

    assert derivatives.shape == voltages.shape
    assert jnp.all(jnp.isfinite(derivatives))


def test_rhs_supports_full_302_neuron_shape() -> None:
    params = neural_fixture(neurons=302)
    voltage = jnp.full((302,), -0.35)
    external = jnp.zeros_like(voltage).at[0].set(0.1)

    derivative = jax.jit(neural_rhs)(0.0, voltage, (params, external))

    assert derivative.shape == (302,)
    assert jnp.all(jnp.isfinite(derivative))
