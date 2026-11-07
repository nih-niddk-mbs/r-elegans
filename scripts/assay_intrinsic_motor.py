"""Assay autonomous A and conditionally oscillatory B cells without pacing.

Fixed-coefficient experiment, not training or a behavioral locomotion assay.
Raw arrays and source/environment records must stay outside the repository.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.brain.intrinsic_motor import (
    IntrinsicMotorConfig, initial_motor_state, motor_cell_rhs, motor_class_masks,
    rk4_step,
)
from r_elegans.model import load_builtin_model
from r_elegans.rl.full_network_actor import (
    init_full_network_actor, make_full_network_actor_interface,
)
if __package__:
    from .experiment_records import capture_run_record
else:
    from experiment_records import capture_run_record


def isolated_trace(config, *, dt=0.02, substeps=4, steps=1500, initial=-0.35):
    """A, quiet B, tonic B, and B whose tonic input ends halfway through."""
    bias = jnp.array([config.a_bias, 0., 0., 0.])

    def step(state, index):
        current = jnp.array([0., 0., 90., jnp.where(index < steps//2, 90., 0.)])

        def rhs(s):
            dv, dw = motor_cell_rhs(s[0], s[1], bias, current, config)
            return jnp.stack((dv, dw))

        def substep(s, _):
            return rk4_step(s, rhs, dt/substeps), None
        state = jax.lax.scan(substep, state, None, length=substeps)[0]
        return state, state

    return jax.lax.scan(step, initial_motor_state(jnp.full(4, initial)),
                        jnp.arange(steps))[1]


def cellular_metrics(trace, dt):
    tail = np.asarray(trace)[-round(10/dt):, 0]
    return {
        name: {"voltage_p2p": float(np.ptp(tail[:, i])),
               "upward_crossings": int(np.sum((tail[:-1, i] < 0.) & (tail[1:, i] >= 0.)))}
        for i, name in enumerate(("A_autonomous", "B_quiet", "B_tonic", "B_withdrawn"))
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if output == repo or repo in output.parents:
        parser.error("Keep artifacts outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    capture_run_record(output, repo, args)
    config = IntrinsicMotorConfig()
    dt = .02
    run = jax.jit(lambda: isolated_trace(config))
    coarse = np.asarray(run())
    fine = np.asarray(jax.jit(lambda: isolated_trace(config, substeps=8))())
    perturbed = np.asarray(jax.jit(lambda: isolated_trace(config, initial=-.34))())
    blocked = np.asarray(jax.jit(lambda: isolated_trace(
        IntrinsicMotorConfig(calcium_conductance=0.)))())
    metrics = cellular_metrics(coarse, dt)
    block_metrics = cellular_metrics(blocked, dt)
    perturb_metrics = cellular_metrics(perturbed, dt)
    model = load_builtin_model()
    actor = init_full_network_actor(model, jax.random.PRNGKey(0))
    interface = make_full_network_actor_interface(
        model, input_mode="current", intrinsic_motor=config)

    def network_scan():
        def step(state, _):
            muscle, state = interface.deterministic_step(actor, state, jnp.zeros(302))
            return state, (state, muscle)
        return jax.lax.scan(step, interface.init_carry(actor), None, length=1500)[1]

    network_state, network_muscle = jax.jit(network_scan)()
    a, b = motor_class_masks(model.neuron_ids)
    network_tail = np.asarray(network_state)[-500:, 0]
    network_metrics = {}
    for name, mask in (("A", a), ("B", b)):
        selected = network_tail[:, np.asarray(mask)]
        crossings = np.sum((selected[:-1] < 0.) & (selected[1:] >= 0.), axis=0)
        amplitudes = np.ptp(selected, axis=0)
        network_metrics[name] = {
            "cells": int(np.sum(mask)),
            "cells_with_at_least_three_crossings": int(np.sum(crossings >= 3)),
            "minimum_voltage_p2p": float(amplitudes.min()),
            "maximum_voltage_p2p": float(amplitudes.max()),
            "minimum_crossings": int(crossings.min()),
            "maximum_crossings": int(crossings.max()),
        }
    convergence = float(np.max(np.abs(coarse-fine)))
    gates = {
        "finite": bool(all(np.isfinite(t).all() for t in (coarse, fine, perturbed, blocked))),
        "A_sustained": metrics["A_autonomous"]["upward_crossings"] >= 3
                        and metrics["A_autonomous"]["voltage_p2p"] > .2,
        "B_quiet": metrics["B_quiet"]["voltage_p2p"] < 1e-4,
        "B_tonic_sustained": metrics["B_tonic"]["upward_crossings"] >= 3
                             and metrics["B_tonic"]["voltage_p2p"] > .2,
        "B_stops_after_withdrawal": metrics["B_withdrawn"]["voltage_p2p"] < 1e-4,
        "calcium_ablation": block_metrics["A_autonomous"]["voltage_p2p"] < 1e-4,
        "initial_perturbation": perturb_metrics["A_autonomous"]["upward_crossings"] >= 3,
        "substep_convergence": convergence < .002,
        "full_network_finite": bool(np.isfinite(network_state).all()
                                    and np.isfinite(network_muscle).all()),
    }
    summary = {
        "experiment": "E024", "status": "completed", "passed": all(gates.values()),
        "scope": "reduced cellular dynamics; no biological fit or locomotion claim",
        "config": asdict(config), "seed": 0, "dt": dt, "steps": 1500,
        "substeps": 4, "comparison_substeps": 8, "metrics_last_10_seconds": metrics,
        "calcium_ablation": block_metrics, "perturbed_initial": perturb_metrics,
        "maximum_state_difference": convergence, "gates": gates,
        "connected_network_last_10_seconds": network_metrics,
        "A_neurons": [n for n, m in zip(model.neuron_ids, a) if m],
        "B_neurons": [n for n, m in zip(model.neuron_ids, b) if m],
        "full_network_voltage_p2p": np.ptp(np.asarray(network_state)[-500:, 0], axis=0).tolist(),
        "network_input": "zero applied current; anatomical recurrent coupling enabled",
        "cell_inputs": "A bias 90; B tonic current 90, B withdrawal at 15 seconds",
    }
    np.savez_compressed(output/"traces.npz", isolated=coarse, refined=fine,
                        perturbed=perturbed, calcium_blocked=blocked,
                        network_state=np.asarray(network_state), muscles=np.asarray(network_muscle))
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"passed": summary["passed"], "gates": gates, "metrics": metrics,
                      "maximum_state_difference": convergence}), flush=True)
    raise SystemExit(0 if summary["passed"] else 2)


if __name__ == "__main__":
    main()
