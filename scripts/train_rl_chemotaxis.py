"""Train the seven-parameter sensory controller with reinforcement learning.

Unlike ``scripts/train_petri_chemotaxis.py``, which fits the same seven
numbers by differentiating directly through the muscle/body simulator, this
script treats the Petri dish as a black-box Gymnax environment
(:mod:`r_elegans.envs.gymnax_petri_dish`) and optimizes the controller with
model-free reinforcement learning (:mod:`r_elegans.rl`): PPO's clipped
surrogate objective by default, or a simpler A2C policy gradient via
``--algorithm a2c``. Only the body-direct path is covered; the supervised
motor teacher and neuromuscular projection are untouched.

Requires the optional ``env`` extra: ``pip install -e ".[env]"``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from r_elegans.body import default_muscle_body_params
from r_elegans.envs import (
    decode_sensory_policy,
    default_petri_dish_params,
    simulate_petri_dish,
)
from r_elegans.envs.gymnax_petri_dish import PetriDishGymnaxEnv
from r_elegans.model import load_builtin_model
from r_elegans.rl import TrainingConfig, train

INITIAL_RAW_POLICY = jnp.asarray([0.0, 0.05, -0.05, 0.05, -0.05, 0.0, -2.0])


def make_trials(seed: int, count: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample source locations and initial headings reproducibly."""

    generator = np.random.default_rng(seed)
    angles = generator.uniform(-np.pi, np.pi, count)
    radii = generator.uniform(0.65, 0.85, count)
    sources = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    headings = generator.uniform(-np.pi, np.pi, count)
    return jnp.asarray(sources, dtype=jnp.float32), jnp.asarray(
        headings, dtype=jnp.float32
    )


def evaluate_direct(
    raw_policy, sources, headings, body_params, gait_params, *, steps: int
) -> dict[str, np.ndarray]:
    """Deterministically roll out the trained controller through the body."""

    minimum_distances = []
    final_distances = []
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
    minimum = np.asarray(minimum_distances)
    return {
        "minimum_distance": minimum,
        "final_distance": np.asarray(final_distances),
        "success": minimum < 0.12,
    }


def print_metrics(label: str, metrics: dict[str, np.ndarray]) -> None:
    print(
        f"{label}: success={np.mean(metrics['success']):.1%} "
        f"mean_min_distance={np.mean(metrics['minimum_distance']):.4f} "
        f"mean_final_distance={np.mean(metrics['final_distance']):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm", choices=("ppo", "a2c"), default="ppo",
        help="RL update rule: PPO's clipped surrogate objective (default) or plain A2C",
    )
    parser.add_argument("--episode-steps", type=int, default=250)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-updates", type=int, default=300)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument(
        "--num-minibatches", type=int, default=4, help="PPO minibatches per epoch"
    )
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-episodes", type=int, default=24)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to results/behavior/petri_chemotaxis_<algorithm>_v1.npz",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(f"results/behavior/petri_chemotaxis_{args.algorithm}_v1.npz")

    model = load_builtin_model()
    body_params = default_muscle_body_params(12)
    env = PetriDishGymnaxEnv(body_params, model.gait_params)
    env_params = env.default_params.replace(max_steps_in_episode=args.episode_steps)

    config = TrainingConfig(
        num_envs=args.num_envs,
        num_steps=args.episode_steps,
        num_updates=args.num_updates,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        clip_eps=args.clip_eps,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
    )
    agent, history = train(
        env,
        env_params,
        config,
        algorithm=args.algorithm,
        seed=args.seed,
        log_every=args.log_every,
    )
    raw_policy = agent.actor.raw_sensory_policy

    test_sources, test_headings = make_trials(19, args.test_episodes)
    initial_metrics = evaluate_direct(
        INITIAL_RAW_POLICY,
        test_sources,
        test_headings,
        body_params,
        model.gait_params,
        steps=args.eval_steps,
    )
    trained_metrics = evaluate_direct(
        raw_policy,
        test_sources,
        test_headings,
        body_params,
        model.gait_params,
        steps=args.eval_steps,
    )

    print(f"policy={decode_sensory_policy(raw_policy)}")
    print_metrics("initial held-out", initial_metrics)
    print_metrics("RL-trained held-out", trained_metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        algorithm=args.algorithm,
        raw_policy=np.asarray(raw_policy),
        log_std=np.asarray(agent.actor.log_std),
        history_update=np.asarray([m["update"] for m in history]),
        history_reward=np.asarray([m["mean_reward"] for m in history]),
        history_success=np.asarray([m["success_rate"] for m in history]),
        test_sources=np.asarray(test_sources),
        test_headings=np.asarray(test_headings),
        **{f"initial_test_{key}": value for key, value in initial_metrics.items()},
        **{f"trained_test_{key}": value for key, value in trained_metrics.items()},
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
