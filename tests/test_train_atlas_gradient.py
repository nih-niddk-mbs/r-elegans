import jax.numpy as jnp
import numpy as np

from scripts.train_atlas_gradient import correlation_loss, dense_pairs


def test_correlation_loss_bounds():
    z = jnp.asarray(np.random.default_rng(0).normal(size=(6, 10)))
    w = jnp.ones_like(z)
    assert float(correlation_loss(z * 3. + 1., z, w)[0]) < 1e-9
    assert abs(float(correlation_loss(-z, z, w)[0]) - 2.) < 1e-9


def test_correlation_loss_ignores_unscored_entries_and_zero_response():
    z = jnp.asarray(np.random.default_rng(1).normal(size=(4, 8)))
    w = jnp.zeros_like(z).at[:, :6].set(1.)
    junk = z.at[:, 6:].set(1e6)
    assert abs(float(correlation_loss(junk, z, w)[0]) - float(correlation_loss(z, z, w)[0])) < 1e-9
    assert np.isfinite(float(correlation_loss(jnp.zeros_like(z), z, w)[0]))


def test_dense_pairs_clips_and_skips_unknown_names():
    stats = dict(pairs=[('A', 'B'), ('A', 'X'), ('Z', 'B')], mean_z=np.array([9., 1., 2.]))
    z, w = dense_pairs(stats, ['A'], ['A', 'B'], 5.)
    assert z.shape == (1, 2) and w.sum() == 1. and z[0, 1] == 5.
