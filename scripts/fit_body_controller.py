"""Fit a compact forward-locomotion controller with JAX gradients."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.body import (
    body_motion_loss,
    decode_periodic_controller,
    default_muscle_body_params,
    simulate_periodic_controller,
)


def fit(iterations: int, learning_rate: float) -> tuple[jax.Array, list[float]]:
    """Fit five raw controller parameters with Adam."""

    body_params = default_muscle_body_params(12)
    loss_and_gradient = jax.jit(jax.value_and_grad(body_motion_loss))
    raw = jnp.asarray([0.0, -0.5, -0.35, 0.0, 0.0])
    first_moment = jnp.zeros_like(raw)
    second_moment = jnp.zeros_like(raw)
    losses: list[float] = []

    for iteration in range(1, iterations + 1):
        loss, gradient = loss_and_gradient(raw, body_params)
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        corrected_first = first_moment / (1.0 - 0.9**iteration)
        corrected_second = second_moment / (1.0 - 0.999**iteration)
        raw = raw - learning_rate * corrected_first / (
            jnp.sqrt(corrected_second) + 1e-8
        )
        losses.append(float(loss))
        if iteration == 1 or iteration % 20 == 0 or iteration == iterations:
            print(f"iteration={iteration:4d} loss={float(loss): .6f}")

    return raw, losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    raw, losses = fit(args.iterations, args.learning_rate)
    controller = decode_periodic_controller(raw)
    final, trajectory, activations = simulate_periodic_controller(
        controller, default_muscle_body_params(12)
    )
    print(f"initial_loss={losses[0]:.6f}")
    print(f"final_loss={losses[-1]:.6f}")
    print(f"controller={controller}")
    print(
        "final_position="
        f"({float(final.position[0]):.6f}, {float(final.position[1]):.6f})"
    )
    print(f"heading_rad={float(final.heading):.6f}")
    print(f"mean_activation={float(jnp.mean(activations)):.6f}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            raw_controller=np.asarray(raw),
            controller=np.asarray(controller),
            losses=np.asarray(losses),
            time=np.asarray(trajectory.time),
            position=np.asarray(trajectory.position),
            heading=np.asarray(trajectory.heading),
            joint_angles=np.asarray(trajectory.joint_angles),
            muscle_activations=np.asarray(activations),
        )
        print(f"output={args.output}")


if __name__ == "__main__":
    main()
