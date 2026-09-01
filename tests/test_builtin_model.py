import jax.numpy as jnp
import numpy as np

from r_elegans.body import default_muscle_body_params
from r_elegans.data import load_connectome, load_neuron_parameters
from r_elegans.envs import default_petri_dish_params, simulate_neural_petri_dish
from r_elegans.model import load_builtin_model


def test_builtin_checkpoint_contains_full_runtime_topology(monkeypatch) -> None:
    monkeypatch.delenv("R_ELEGANS_DATA_DIR", raising=False)
    connectome = load_connectome()
    model = load_builtin_model()

    assert len(model.neuron_ids) == 302
    assert connectome.chemical_counts.shape == (302, 302)
    assert connectome.gap_counts.shape == (302, 302)
    assert np.count_nonzero(connectome.chemical_counts) == 3709
    assert np.count_nonzero(connectome.gap_counts) == 2186
    np.testing.assert_array_equal(connectome.gap_counts, connectome.gap_counts.T)
    assert np.count_nonzero(model.neuromuscular_params.synapse_weights) == 956
    assert model.neural_motor_coefficients.shape == (302, 13)


def test_builtin_electrophysiology_parameters_need_no_data_root(
    monkeypatch,
) -> None:
    monkeypatch.delenv("R_ELEGANS_DATA_DIR", raising=False)

    record = load_neuron_parameters("AWCON")

    assert record.neuron_class == "AWCON"
    assert record.params.conductances_nS.shape == (17,)


def test_builtin_model_drives_neural_body_without_external_files() -> None:
    model = load_builtin_model()
    dish = default_petri_dish_params(jnp.asarray([-0.6771441, 0.34037605]))

    final, _, observations, voltage = simulate_neural_petri_dish(
        model.raw_sensory_policy,
        dish,
        default_muscle_body_params(12),
        model.gait_params,
        model.neural_motor_coefficients,
        model.neuromuscular_params,
        heading=jnp.asarray(2.057443),
        steps=5,
    )

    assert voltage.shape == (5, 302)
    assert observations.muscle_activation.shape == (5, 95)
    assert jnp.all(jnp.isfinite(final.body.position))
