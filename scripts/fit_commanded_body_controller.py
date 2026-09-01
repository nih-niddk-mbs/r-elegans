"""Fit the continuous [speed, steering] body action interface."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.body import (
    commanded_body_motion_loss,
    controller_for_command,
    decode_commanded_controller,
    default_muscle_body_params,
    simulate_periodic_controller,
)


COMMANDS = jnp.asarray(
    [
        (speed, steering)
        for speed in (-1.0, -0.5, 0.5, 1.0)
        for steering in (-0.5, 0.0, 0.5)
    ]
)


def fit(iterations: int, learning_rate: float) -> tuple[jax.Array, list[float]]:
    """Fit shared forward/reverse gaits over a 12-command training grid."""

    body_params = default_muscle_body_params(12)
    objective = jax.jit(jax.value_and_grad(commanded_body_motion_loss))
    raw = jnp.asarray(
        [
            2.34,
            -0.45,
            -0.256,
            0.64,
            0.011,
            2.47,
            -0.535,
            0.369,
            0.714,
            -0.018,
            0.2,
            -0.2,
            0.5,
            0.0,
        ]
    )
    first_moment = jnp.zeros_like(raw)
    second_moment = jnp.zeros_like(raw)
    losses: list[float] = []

    for iteration in range(1, iterations + 1):
        loss, gradient = objective(raw, body_params, COMMANDS)
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        first_corrected = first_moment / (1.0 - 0.9**iteration)
        second_corrected = second_moment / (1.0 - 0.999**iteration)
        raw = raw - learning_rate * first_corrected / (
            jnp.sqrt(second_corrected) + 1e-8
        )
        losses.append(float(loss))
        if iteration == 1 or iteration % 20 == 0 or iteration == iterations:
            print(f"iteration={iteration:4d} loss={float(loss): .6f}")
    return raw, losses


def evaluate(raw: jax.Array) -> dict[str, np.ndarray]:
    """Roll out every command and return compact metrics plus trajectories."""

    params = decode_commanded_controller(raw)
    body_params = default_muscle_body_params(12)
    positions = []
    headings = []
    position_trajectories = []
    heading_trajectories = []
    joint_trajectories = []
    muscle_trajectories = []
    for command in COMMANDS:
        controller = controller_for_command(command, params)
        final, trajectory, activations = simulate_periodic_controller(
            controller, body_params
        )
        positions.append(np.asarray(final.position))
        headings.append(np.asarray(final.heading))
        position_trajectories.append(np.asarray(trajectory.position))
        heading_trajectories.append(np.asarray(trajectory.heading))
        joint_trajectories.append(np.asarray(trajectory.joint_angles))
        muscle_trajectories.append(np.asarray(activations))
        print(
            f"command=({float(command[0]):+.1f},{float(command[1]):+.1f}) "
            f"position=({float(final.position[0]):+.3f},"
            f"{float(final.position[1]):+.3f}) "
            f"heading={float(final.heading):+.3f}"
        )
    return {
        "commands": np.asarray(COMMANDS),
        "final_positions": np.asarray(positions),
        "final_headings": np.asarray(headings),
        "positions": np.asarray(position_trajectories),
        "headings": np.asarray(heading_trajectories),
        "joint_angles": np.asarray(joint_trajectories),
        "muscle_activations": np.asarray(muscle_trajectories),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    raw, losses = fit(args.iterations, args.learning_rate)
    metrics = evaluate(raw)
    print(f"initial_loss={losses[0]:.6f}")
    print(f"final_loss={losses[-1]:.6f}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            raw_controller=np.asarray(raw),
            losses=np.asarray(losses),
            **metrics,
        )
        print(f"output={args.output}")


if __name__ == "__main__":
    main()
