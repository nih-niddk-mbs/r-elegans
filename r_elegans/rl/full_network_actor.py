"""Full recurrent connectome with anatomical muscle output and optional freezing.

Normalized graded-potential dynamics, not a fitted ion-channel model. Anatomical
counts initialize relative magnitudes; they are not measured conductances.
Unknown NMJ signs start at zero and can be learned; known signs stay fixed.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.brain.circuit import build_subcircuit_params, decode_subcircuit_params
from r_elegans.brain.dynamics import effective_chemical_weights, effective_gap_weights
from .actor_interface import ActorInterface
from .policy import action_std, gaussian_log_prob


class FullNetworkActorParams(NamedTuple):
    neural: object
    sensory_weights: jax.Array
    raw_nmj: jax.Array
    unknown_polarity: jax.Array
    muscle_threshold: jax.Array
    raw_muscle_slope: jax.Array
    log_std: jax.Array


def init_full_network_actor(model, key):
    neural = build_subcircuit_params(model.connectome, model.neuron_ids)
    # Count-based magnitudes start small enough to avoid a strongly coupled,
    # saturated initial network. All existing chemical and gap edges can vary.
    neural = neural._replace(
        raw_chemical=jnp.log(jnp.expm1(0.01 * jnp.maximum(model.connectome.chemical_counts, 1))),
        raw_gap=jnp.log(jnp.expm1(0.005 * jnp.maximum(model.connectome.gap_counts, 1))),
        leak_reversal=neural.leak_reversal + 0.02 * jax.random.normal(key, (302,)),
    )
    nmj = model.neuromuscular_params
    # Unit-weight multipliers preserve the bundled contact-count prior.
    return FullNetworkActorParams(
        neural, 0.1 * jax.random.normal(jax.random.fold_in(key, 1), (4, 3)),
        jnp.full((95, 302), jnp.log(jnp.expm1(1.0))),
        jnp.zeros((95, 302)), nmj.muscle_threshold,
        jnp.full((95,), jnp.log(jnp.expm1(0.08))), jnp.full((95,), -1.5),
    )


def integrate_frozen_conductance(voltage, params, external, dt, substeps):
    """Exponential Euler with frozen gates and neighboring voltages per substep.

    Each scalar leak/conductance equation is solved exactly at frozen inputs.
    Positive conductances yield a bounded relaxation, even for dense/high-degree
    nodes. This is first order, not the adaptive reference solver; substep
    convergence must be checked for any fitted parameter regime.
    """
    chemical = effective_chemical_weights(params)
    gap = effective_gap_weights(params)
    gap_sum = gap.sum(axis=1)

    def step(v, _):
        gates = jax.nn.sigmoid((v - params.threshold) / params.slope)
        total = 1.0 + gap_sum + jnp.matmul(chemical, gates, precision=jax.lax.Precision.HIGHEST)
        drive = (params.leak_reversal + external
                 + jnp.matmul(gap, v, precision=jax.lax.Precision.HIGHEST)
                 + jnp.matmul(chemical, gates * params.synapse_reversal, precision=jax.lax.Precision.HIGHEST))
        fraction = -jnp.expm1(-total * (dt / substeps) / params.time_constant)
        return v + fraction * (drive / total - v), None

    return jax.lax.scan(step, voltage, None, length=substeps)[0]


def make_full_network_actor_interface(model, *, dt=0.02, substeps=4,
                                      train_group="all", train_neurons=None):
    """Keep all 302 nodes and all topology while choosing which fields learn.

    Groups: all, neural, nmj, sensory. Neuron selection restricts incoming
    chemical rows, incident electrical edges, per-neuron fields, sensory rows,
    and presynaptic NMJ columns. Muscle-wide fields/noise freeze with a neuron
    subset. Selection changes gradients, never the forward computation.
    """
    if dt <= 0 or substeps < 1:
        raise ValueError("dt and substeps must be positive")
    if train_group not in ("all", "neural", "nmj", "sensory"):
        raise ValueError("Unknown train_group")
    ids = model.neuron_ids
    names = ids if train_neurons is None else tuple(train_neurons)
    if not names or set(names) - set(ids):
        raise ValueError("Select nonempty canonical neuron names")
    selected = jnp.asarray([name in names for name in ids], dtype=jnp.float32)
    sensory_indices = jnp.asarray([ids.index(n) for n in ("AWCL", "AWCR", "ASEL", "ASER")])
    nmj = model.neuromuscular_params
    nmj_mask = (nmj.synapse_weights > 0).astype(jnp.float32)
    known_sign = nmj.synapse_signs

    def restrict(value, mask):
        fixed = jax.lax.stop_gradient(value)
        return fixed + jax.lax.stop_gradient(jnp.asarray(mask)) * (value - fixed)

    def project(actor):
        neural_on = train_group in ("all", "neural")
        fields = {}
        for name, value in actor.neural._asdict().items():
            if name in ("chemical_mask", "gap_mask"):
                mask = 0.0
            elif name == "raw_chemical":
                mask = selected[:, None] * actor.neural.chemical_mask * neural_on
            elif name == "raw_gap":
                mask = jnp.maximum(selected[:, None], selected[None, :]) * actor.neural.gap_mask * neural_on
            else:
                mask = selected * neural_on
            fields[name] = restrict(value, mask)
        nmj_on = train_group in ("all", "nmj")
        global_on = nmj_on and train_neurons is None
        return actor._replace(
            neural=actor.neural._replace(**fields),
            sensory_weights=restrict(actor.sensory_weights, selected[sensory_indices, None] * (train_group in ("all", "sensory"))),
            raw_nmj=restrict(actor.raw_nmj, nmj_mask * selected[None, :] * nmj_on),
            unknown_polarity=restrict(actor.unknown_polarity, nmj_mask * (known_sign == 0) * selected[None, :] * nmj_on),
            muscle_threshold=restrict(actor.muscle_threshold, global_on),
            raw_muscle_slope=restrict(actor.raw_muscle_slope, global_on),
            log_std=restrict(actor.log_std, train_group == "all" and train_neurons is None),
        )

    def distribution_step(actor, voltage, obs):
        actor = project(actor)
        params = decode_subcircuit_params(actor.neural)
        external = jnp.zeros((302,)).at[sensory_indices].set(actor.sensory_weights @ jnp.tanh(obs))
        next_voltage = integrate_frozen_conductance(voltage, params, external, dt, substeps)
        # NMJ activation uses the SAME normalized voltage units as the circuit.
        # No silent substitution of the older mV teacher's thresholds.
        weights = nmj.synapse_weights * jax.nn.softplus(actor.raw_nmj)
        signs = jnp.where(known_sign != 0, known_sign, jnp.tanh(actor.unknown_polarity)) * nmj_mask
        presynaptic = jax.nn.sigmoid((next_voltage - params.threshold) / params.slope)
        drive = jnp.matmul(weights * signs, presynaptic, precision=jax.lax.Precision.HIGHEST)
        drive /= jnp.maximum(weights.sum(axis=1), 1.0)
        # Smooth muscle transfer permits learning even from initially inhibitory
        # net drive. Unlike the motor teacher, it has a nonzero resting tone.
        activation = jax.nn.sigmoid((drive - actor.muscle_threshold)
                                   / (0.02 + jax.nn.softplus(actor.raw_muscle_slope)))
        # Gaussian in latent logit space; sigmoid maps samples into [0, 1].
        activation = jnp.clip(activation, 1e-5, 1 - 1e-5)
        mean = jnp.log(activation) - jnp.log1p(-activation)
        return mean, action_std(actor.log_std), next_voltage

    def step(actor, carry, obs, key):
        mean, std, next_carry = distribution_step(actor, carry, obs)
        raw = mean + std * jax.random.normal(key, mean.shape)
        return raw, jax.nn.sigmoid(raw), jnp.sum(gaussian_log_prob(raw, mean, std)), next_carry

    def deterministic(actor, carry, obs):
        mean, _, next_carry = distribution_step(actor, carry, obs)
        return jax.nn.sigmoid(mean), next_carry

    return ActorInterface(step, lambda a, v, o: distribution_step(a, v, o)[:2],
                          distribution_step, lambda a: project(a).neural.leak_reversal,
                          deterministic)


def actor_to_arrays(actor):
    return {**{"neural_" + k: np.asarray(v) for k, v in actor.neural._asdict().items()},
            **{k: np.asarray(v) for k, v in actor._asdict().items() if k != "neural"}}


def actor_from_arrays(arrays):
    from r_elegans.brain.circuit import RawSubcircuitParams
    neural = RawSubcircuitParams(**{k: jnp.asarray(arrays["neural_" + k]) for k in RawSubcircuitParams._fields})
    actor = FullNetworkActorParams(neural, **{k: jnp.asarray(arrays[k]) for k in FullNetworkActorParams._fields if k != "neural"})
    expected = ((302, 302), (302, 302), (302, 302), (302, 302), *((302,),) * 5)
    shapes = {"sensory_weights": (4, 3), "raw_nmj": (95, 302), "unknown_polarity": (95, 302),
              "muscle_threshold": (95,), "raw_muscle_slope": (95,), "log_std": (95,)}
    if (any(v.shape != s for v, s in zip(neural, expected))
            or any(getattr(actor, k).shape != s for k, s in shapes.items())):
        raise ValueError("Invalid full-network checkpoint shapes")
    if not all(np.all(np.isfinite(v)) for v in jax.tree_util.tree_leaves(actor)):
        raise ValueError("Nonfinite full-network checkpoint")
    return actor
