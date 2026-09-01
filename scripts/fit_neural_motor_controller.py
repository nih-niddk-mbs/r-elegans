"""Fit 302 neural voltage outputs to the saved 95-muscle motion library."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.body import (
    NeuromuscularParams,
    decode_commanded_controller,
    default_muscle_body_params,
    simulate_muscle_trajectory,
)
from r_elegans.brain import (
    MOTOR_FEATURE_COUNT,
    motor_command_features,
    neural_motor_loss,
)
from r_elegans.data import load_neuromuscular_connectome


def fit(
    features: jax.Array,
    target_muscles: jax.Array,
    neuromuscular_params: NeuromuscularParams,
    trainable_neurons: jax.Array,
    *,
    iterations: int,
    learning_rate: float,
) -> tuple[jax.Array, list[float]]:
    """Fit a compact command/phase-to-voltage map with Adam."""

    key = jax.random.PRNGKey(0)
    coefficients = 0.01 * jax.random.normal(
        key, (trainable_neurons.shape[0], MOTOR_FEATURE_COUNT)
    )
    first_moment = jnp.zeros_like(coefficients)
    second_moment = jnp.zeros_like(coefficients)

    def objective(value: jax.Array) -> jax.Array:
        return neural_motor_loss(
            value,
            features,
            target_muscles,
            neuromuscular_params,
            trainable_neurons,
        )[0]

    loss_and_gradient = jax.jit(jax.value_and_grad(objective))
    losses: list[float] = []
    for iteration in range(1, iterations + 1):
        loss, gradient = loss_and_gradient(coefficients)
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        first_corrected = first_moment / (1.0 - 0.9**iteration)
        second_corrected = second_moment / (1.0 - 0.999**iteration)
        coefficients = coefficients - learning_rate * first_corrected / (
            jnp.sqrt(second_corrected) + 1e-8
        )
        losses.append(float(loss))
        if iteration == 1 or iteration % 50 == 0 or iteration == iterations:
            print(f"iteration={iteration:4d} loss={float(loss):.8f}")
    return coefficients, losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--motion-library",
        default="results/body_fits/commanded_body_fit_v1.npz",
    )
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    motion_path = args.data_root / args.motion_library
    with np.load(motion_path, allow_pickle=False) as motion:
        commands = jnp.asarray(motion["commands"])
        target_muscles = jnp.asarray(motion["muscle_activations"])
        gait_params = decode_commanded_controller(
            jnp.asarray(motion["raw_controller"])
        )
        target_positions = np.asarray(motion["final_positions"])
    times = jnp.arange(target_muscles.shape[1]) * 0.02
    features = motor_command_features(commands, times, gait_params)

    connectome = load_neuromuscular_connectome(root=args.data_root)
    if connectome.synapse_signs is None:
        raise ValueError("The processed neuromuscular artifact has no polarity")
    neuron_count = len(connectome.neuron_ids)
    neuromuscular_params = NeuromuscularParams(
        synapse_weights=connectome.chemical_counts,
        synapse_signs=connectome.synapse_signs,
        neuron_threshold=jnp.full((neuron_count,), -20.0),
        neuron_slope=jnp.full((neuron_count,), 5.0),
        muscle_threshold=jnp.full((95,), 0.05),
        muscle_slope=jnp.full((95,), 0.1),
    )
    trainable_neurons = jnp.any(connectome.synapse_signs != 0.0, axis=0)
    print(
        f"training {int(jnp.sum(trainable_neurons))} signed NMJ neurons "
        f"across {commands.shape[0]} commands"
    )

    coefficients, losses = fit(
        features,
        target_muscles,
        neuromuscular_params,
        trainable_neurons,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
    )
    final_loss, (voltages, predicted_muscles) = neural_motor_loss(
        coefficients,
        features,
        target_muscles,
        neuromuscular_params,
        trainable_neurons,
    )
    body_params = default_muscle_body_params(12)
    predicted_final, predicted_trajectories = jax.vmap(
        lambda activations: simulate_muscle_trajectory(
            activations, body_params
        )
    )(predicted_muscles)
    muscle_rmse = float(jnp.sqrt(jnp.mean((predicted_muscles - target_muscles) ** 2)))
    endpoint_rmse = float(
        jnp.sqrt(
            jnp.mean(
                (predicted_final.position - jnp.asarray(target_positions)) ** 2
            )
        )
    )
    print(f"final_loss={float(final_loss):.8f}")
    print(f"muscle_rmse={muscle_rmse:.6f}")
    print(f"body_endpoint_rmse={endpoint_rmse:.6f}")
    print(
        f"voltage_range=({float(jnp.min(voltages)):.3f},"
        f" {float(jnp.max(voltages)):.3f}) mV"
    )
    for command, target, predicted in zip(
        np.asarray(commands), target_positions, np.asarray(predicted_final.position)
    ):
        print(
            f"command=({command[0]:+.1f},{command[1]:+.1f}) "
            f"target=({target[0]:+.3f},{target[1]:+.3f}) "
            f"predicted=({predicted[0]:+.3f},{predicted[1]:+.3f})"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            coefficients=np.asarray(coefficients),
            trainable_neurons=np.asarray(trainable_neurons),
            neuron_ids=np.asarray(connectome.neuron_ids, dtype="U16"),
            commands=np.asarray(commands),
            times=np.asarray(times),
            losses=np.asarray(losses),
            voltages=np.asarray(voltages),
            predicted_muscles=np.asarray(predicted_muscles),
            target_muscles=np.asarray(target_muscles),
            body_positions=np.asarray(predicted_trajectories.position),
            body_joint_angles=np.asarray(predicted_trajectories.joint_angles),
            final_positions=np.asarray(predicted_final.position),
        )
        print(f"output={args.output}")


if __name__ == "__main__":
    main()
