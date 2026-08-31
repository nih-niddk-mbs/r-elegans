import jax.numpy as jnp
import numpy as np
import pytest

from r_elegans.data import validate_connectome


def test_valid_302_neuron_connectome_passes() -> None:
    neuron_ids = tuple(f"N{index:03d}" for index in range(302))
    chemical = jnp.zeros((302, 302))
    gap = jnp.zeros((302, 302))

    validate_connectome(neuron_ids, chemical, gap)


@pytest.mark.parametrize("failure", ["duplicate", "shape", "negative", "asymmetric"])
def test_invalid_connectome_is_rejected(failure: str) -> None:
    neuron_ids = [f"N{index:03d}" for index in range(302)]
    chemical = np.zeros((302, 302))
    gap = np.zeros((302, 302))

    if failure == "duplicate":
        neuron_ids[-1] = neuron_ids[0]
    elif failure == "shape":
        chemical = chemical[:-1]
    elif failure == "negative":
        chemical[0, 1] = -1
    else:
        gap[0, 1] = 1

    with pytest.raises(ValueError):
        validate_connectome(neuron_ids, chemical, gap)

