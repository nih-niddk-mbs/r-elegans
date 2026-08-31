import jax
import jax.numpy as jnp

from r_elegans.brain import NeuralParams, integrate_neural_state


def small_network(raw_chemical: jax.Array) -> NeuralParams:
    neurons = raw_chemical.shape[0]
    chemical_mask = jnp.eye(neurons, k=-1) + jnp.eye(neurons, k=neurons - 1)
    gap_mask = jnp.eye(neurons, k=1) + jnp.eye(neurons, k=-1)
    return NeuralParams(
        raw_chemical=raw_chemical,
        chemical_mask=chemical_mask,
        raw_gap=jnp.full_like(raw_chemical, -4.0),
        gap_mask=gap_mask,
        leak_reversal=jnp.full((neurons,), -0.35),
        synapse_reversal=jnp.array([0.2, -0.8, 0.2, -0.8]),
        time_constant=jnp.full((neurons,), 0.1),
        threshold=jnp.full((neurons,), -0.25),
        slope=jnp.full((neurons,), 0.1),
    )


def test_gradient_flows_through_diffrax_solver() -> None:
    initial_raw = jnp.full((4, 4), -2.0)
    initial_voltage = jnp.array([-0.4, -0.3, -0.2, -0.1])
    external = jnp.array([0.15, 0.0, 0.0, 0.0])

    def loss(raw_chemical: jax.Array) -> jax.Array:
        final_voltage = integrate_neural_state(
            initial_voltage,
            small_network(raw_chemical),
            external,
            duration=0.1,
        )
        return jnp.sum(final_voltage**2)

    gradient = jax.jit(jax.grad(loss))(initial_raw)

    assert jnp.all(jnp.isfinite(gradient))
    active_mask = small_network(initial_raw).chemical_mask.astype(bool)
    assert jnp.any(jnp.abs(gradient[active_mask]) > 1e-8)
    assert jnp.all(gradient[~active_mask] == 0)

