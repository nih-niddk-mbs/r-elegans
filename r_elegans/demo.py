"""Zero-configuration command-line demonstration of the bundled worm model."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from r_elegans.body import default_muscle_body_params
from r_elegans.envs import (
    default_petri_dish_params,
    simulate_neural_petri_dish,
    simulate_petri_dish,
)
from r_elegans.model import load_builtin_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bundled engineered-sensory/motor-teacher chemotaxis "
            "baseline; the recurrent connectome is not active"
        )
    )
    parser.add_argument(
        "--mode",
        choices=("neural", "direct"),
        default="neural",
        help=(
            "'neural' uses the supervised 302-output motor teacher and NMJs; "
            "'direct' sends commands to the fitted body gait"
        ),
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--source-x", type=float, default=-0.6771441)
    parser.add_argument("--source-y", type=float, default=0.34037605)
    parser.add_argument("--heading", type=float, default=2.057443)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = load_builtin_model()
    dish = default_petri_dish_params(
        jnp.asarray([args.source_x, args.source_y])
    )
    body_params = default_muscle_body_params(12)
    if args.mode == "neural":
        final, _, observations, voltage = simulate_neural_petri_dish(
            model.raw_sensory_policy,
            dish,
            body_params,
            model.gait_params,
            model.neural_motor_coefficients,
            model.neuromuscular_params,
            heading=jnp.asarray(args.heading),
            steps=args.steps,
        )
    else:
        final, _, observations = simulate_petri_dish(
            model.raw_sensory_policy,
            dish,
            body_params,
            model.gait_params,
            heading=jnp.asarray(args.heading),
            steps=args.steps,
        )
        voltage = None

    minimum_distance = float(jnp.min(observations.distance_to_source))
    final_distance = float(observations.distance_to_source[-1])
    print(f"model={model.model_id}")
    print(f"mode={args.mode} steps={args.steps} duration_s={args.steps * 0.02:.2f}")
    print("sensory_controller=engineered-seven-parameter-policy")
    print(
        "motor_path="
        + (
            "supervised-302-output-teacher+anatomical-nmj"
            if args.mode == "neural"
            else "direct-fitted-body-gait"
        )
    )
    print("recurrent_connectome_active=False")
    print(f"minimum_distance={minimum_distance:.6f}")
    print(f"final_distance={final_distance:.6f}")
    print(f"reached_food={minimum_distance < 0.12}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "source_position": np.asarray(dish.source_position),
            "head_position": np.asarray(observations.head_position),
            "commands": np.asarray(observations.command),
            "muscle_activation": np.asarray(observations.muscle_activation),
            "distance_to_source": np.asarray(
                observations.distance_to_source
            ),
            "final_body_position": np.asarray(final.body.position),
        }
        if voltage is not None:
            arrays["voltage"] = np.asarray(voltage)
        np.savez_compressed(args.output, **arrays)
        print(f"output={args.output}")


if __name__ == "__main__":
    main()
