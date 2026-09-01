"""Train a compact head-sensory policy to find diffusing food."""

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
)
from r_elegans.brain import effective_neural_motor_coefficients
from r_elegans.data import load_neuromuscular_connectome
from r_elegans.envs import (
    decode_sensory_policy,
    default_petri_dish_params,
    petri_navigation_loss,
    simulate_neural_petri_dish,
    simulate_petri_dish,
)


def make_trials(seed: int, count: int) -> tuple[jax.Array, jax.Array]:
    """Sample source locations and initial headings reproducibly."""

    generator = np.random.default_rng(seed)
    angles = generator.uniform(-np.pi, np.pi, count)
    radii = generator.uniform(0.65, 0.85, count)
    sources = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    headings = generator.uniform(-np.pi, np.pi, count)
    return jnp.asarray(sources, dtype=jnp.float32), jnp.asarray(
        headings, dtype=jnp.float32
    )


def fit(
    gait_params,
    body_params,
    sources: jax.Array,
    headings: jax.Array,
    *,
    iterations: int,
    learning_rate: float,
    steps: int,
) -> tuple[jax.Array, list[float]]:
    """Fit seven sensory-policy parameters with Adam."""

    objective = jax.jit(
        jax.value_and_grad(
            lambda raw: petri_navigation_loss(
                raw,
                sources,
                headings,
                body_params,
                gait_params,
                steps=steps,
            )
        )
    )
    raw = jnp.asarray([0.0, 0.05, -0.05, 0.05, -0.05, 0.0, -2.0])
    first_moment = jnp.zeros_like(raw)
    second_moment = jnp.zeros_like(raw)
    best_raw = raw
    best_loss = float("inf")
    losses: list[float] = []

    for iteration in range(1, iterations + 1):
        loss, gradient = objective(raw)
        numeric_loss = float(loss)
        losses.append(numeric_loss)
        if numeric_loss < best_loss:
            best_loss, best_raw = numeric_loss, raw
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        first_corrected = first_moment / (1.0 - 0.9**iteration)
        second_corrected = second_moment / (1.0 - 0.999**iteration)
        raw = raw - learning_rate * first_corrected / (
            jnp.sqrt(second_corrected) + 1e-8
        )
        if iteration == 1 or iteration % 25 == 0 or iteration == iterations:
            print(f"iteration={iteration:4d} loss={numeric_loss:.6f}")
    return best_raw, losses


def evaluate_direct(
    raw_policy,
    sources,
    headings,
    body_params,
    gait_params,
    *,
    steps: int,
) -> dict[str, np.ndarray]:
    """Evaluate body-direct chemotaxis and retain compact trajectories."""

    minimum_distances = []
    final_distances = []
    head_paths = []
    commands = []
    concentrations = []
    for source, heading in zip(sources, headings):
        _, _, observations = simulate_petri_dish(
            raw_policy,
            default_petri_dish_params(source),
            body_params,
            gait_params,
            heading=heading,
            steps=steps,
        )
        distances = np.asarray(observations.distance_to_source)
        minimum_distances.append(distances.min())
        final_distances.append(distances[-1])
        head_paths.append(np.asarray(observations.head_position))
        commands.append(np.asarray(observations.command))
        concentrations.append(np.asarray(observations.concentration))
    minimum = np.asarray(minimum_distances)
    return {
        "minimum_distance": minimum,
        "final_distance": np.asarray(final_distances),
        "success": minimum < 0.12,
        "head_position": np.asarray(head_paths),
        "commands": np.asarray(commands),
        "concentration": np.asarray(concentrations),
    }


def neural_validation(
    raw_policy,
    sources,
    headings,
    body_params,
    gait_params,
    neural_coefficients,
    neuromuscular_params,
    *,
    steps: int,
) -> dict[str, np.ndarray]:
    """Run held-out episodes through neural voltages and anatomical NMJs."""

    minimum_distances = []
    final_distances = []
    representative = None
    for index, (source, heading) in enumerate(zip(sources, headings)):
        _, _, observations, voltage = simulate_neural_petri_dish(
            raw_policy,
            default_petri_dish_params(source),
            body_params,
            gait_params,
            neural_coefficients,
            neuromuscular_params,
            heading=heading,
            steps=steps,
        )
        distances = np.asarray(observations.distance_to_source)
        minimum_distances.append(distances.min())
        final_distances.append(distances[-1])
        if index == 0:
            representative = {
                "head_position": np.asarray(observations.head_position),
                "commands": np.asarray(observations.command),
                "voltage": np.asarray(voltage),
                "muscle_activation": np.asarray(
                    observations.muscle_activation
                ),
            }
    minimum = np.asarray(minimum_distances)
    return {
        "minimum_distance": minimum,
        "final_distance": np.asarray(final_distances),
        "success": minimum < 0.12,
        **{f"representative_{key}": value for key, value in representative.items()},
    }


def print_metrics(label: str, metrics: dict[str, np.ndarray]) -> None:
    print(
        f"{label}: success={np.mean(metrics['success']):.1%} "
        f"mean_min_distance={np.mean(metrics['minimum_distance']):.4f} "
        f"mean_final_distance={np.mean(metrics['final_distance']):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--motion-library",
        default="results/body_fits/commanded_body_fit_v1.npz",
    )
    parser.add_argument(
        "--neural-fit",
        default="results/neural_fits/neural_motor_fit_v1.npz",
    )
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--train-episodes", type=int, default=16)
    parser.add_argument("--test-episodes", type=int, default=24)
    parser.add_argument(
        "--output", default="results/behavior/petri_chemotaxis_v1.npz"
    )
    args = parser.parse_args()

    with np.load(args.data_root / args.motion_library, allow_pickle=False) as data:
        gait_params = decode_commanded_controller(
            jnp.asarray(data["raw_controller"])
        )
    body_params = default_muscle_body_params(12)
    train_sources, train_headings = make_trials(7, args.train_episodes)
    test_sources, test_headings = make_trials(19, args.test_episodes)

    initial_raw = jnp.asarray([0.0, 0.05, -0.05, 0.05, -0.05, 0.0, -2.0])
    initial_test = evaluate_direct(
        initial_raw,
        test_sources,
        test_headings,
        body_params,
        gait_params,
        steps=args.steps,
    )
    raw_policy, losses = fit(
        gait_params,
        body_params,
        train_sources,
        train_headings,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        steps=args.steps,
    )
    train_metrics = evaluate_direct(
        raw_policy,
        train_sources,
        train_headings,
        body_params,
        gait_params,
        steps=args.steps,
    )
    test_metrics = evaluate_direct(
        raw_policy,
        test_sources,
        test_headings,
        body_params,
        gait_params,
        steps=args.steps,
    )

    connectome = load_neuromuscular_connectome(root=args.data_root)
    with np.load(args.data_root / args.neural_fit, allow_pickle=False) as neural:
        neural_coefficients = effective_neural_motor_coefficients(
            jnp.asarray(neural["coefficients"]),
            jnp.asarray(neural["trainable_neurons"]),
        )
    neuromuscular_params = NeuromuscularParams(
        synapse_weights=connectome.chemical_counts,
        synapse_signs=connectome.synapse_signs,
        neuron_threshold=jnp.full((302,), -20.0),
        neuron_slope=jnp.full((302,), 5.0),
        muscle_threshold=jnp.full((95,), 0.05),
        muscle_slope=jnp.full((95,), 0.1),
    )
    neural_metrics = neural_validation(
        raw_policy,
        test_sources,
        test_headings,
        body_params,
        gait_params,
        neural_coefficients,
        neuromuscular_params,
        steps=args.steps,
    )

    print(f"policy={decode_sensory_policy(raw_policy)}")
    print_metrics("initial held-out", initial_test)
    print_metrics("fitted train", train_metrics)
    print_metrics("fitted held-out", test_metrics)
    print_metrics("neural held-out", neural_metrics)

    output = args.data_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        raw_policy=np.asarray(raw_policy),
        losses=np.asarray(losses),
        train_sources=np.asarray(train_sources),
        train_headings=np.asarray(train_headings),
        test_sources=np.asarray(test_sources),
        test_headings=np.asarray(test_headings),
        **{f"initial_test_{key}": value for key, value in initial_test.items()},
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
        **{f"neural_test_{key}": value for key, value in neural_metrics.items()},
    )
    print(f"output={output}")


if __name__ == "__main__":
    main()
