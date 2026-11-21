import numpy as np

from scripts.make_polarity_variants import variant_matrices


def toy():
    rng = np.random.default_rng(0)
    n = 12
    chemical = rng.random((n, n)) * (rng.random((n, n)) < .6)
    sign = np.where(chemical > 0, rng.choice([-1, 0, 1], size=(n, n), p=[.3, .2, .5]), 0)
    source = np.where(sign != 0, rng.choice([1, 2], size=(n, n)), 0)
    old = rng.choice([-85., -40., 0.], size=n)
    return old, sign, source, chemical


def test_every_variant_changes_only_determined_edges():
    old, sign, source, chemical = toy()
    base = np.broadcast_to(old[None, :], sign.shape)
    for name, matrix in variant_matrices(old, sign, source, chemical).items():
        assert np.array_equal(matrix[sign == 0], base[sign == 0]), name


def test_subsets_partition_and_alpha_is_a_blend():
    old, sign, source, chemical = toy()
    base = np.broadcast_to(old[None, :], sign.shape)
    variants = variant_matrices(old, sign, source, chemical)
    changed = lambda m: ~np.isclose(m, base)
    heavy, light = changed(variants['heavy50']), changed(variants['light50'])
    assert not np.any(heavy & light)
    inhibitory, excitatory = changed(variants['inhibitory_only']), changed(variants['excitatory_only'])
    assert not np.any(inhibitory & excitatory)
    full = variants['alpha075'] - base
    assert np.allclose(variants['alpha025'] - base, full / 3.)
    assert np.all(variants['inhibitory_only'][sign < 0] == -85.)
