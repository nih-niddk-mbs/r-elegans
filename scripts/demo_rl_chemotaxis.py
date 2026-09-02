"""Train the sensory controller with RL, then animate the worm finding food.

This is the end-to-end demo for the body-direct reinforcement-learning path:
it trains the seven-parameter sensory controller against the Gymnax-compatible
Petri dish (:mod:`r_elegans.envs.gymnax_petri_dish`, :mod:`r_elegans.rl`) with
PPO's clipped surrogate objective by default (``--algorithm a2c`` selects a
simpler policy-gradient update instead), then rolls the trained controller out
deterministically through the same differentiable body/gait simulator used
elsewhere in this repository and renders the full 12-segment body moving
through the dish toward the food source.

Requires the optional ``env`` and ``demo`` extras:

    pip install -e ".[env,demo]"
    python scripts/demo_rl_chemotaxis.py

Pass ``--load PATH`` to animate a controller already saved by
``scripts/train_rl_chemotaxis.py`` instead of training a new one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm", choices=("ppo", "a2c"), default="ppo",
        help="RL update rule: PPO's clipped surrogate objective (default) or plain A2C",
    )
    parser.add_argument("--episode-steps", type=int, default=250)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-updates", type=int, default=200)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument(
        "--num-minibatches", type=int, default=4, help="PPO minibatches per epoch"
    )
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--load", type=Path, help="Skip training; animate a saved raw_policy npz instead"
    )
    parser.add_argument("--demo-steps", type=int, default=400)
    parser.add_argument("--source-x", type=float, default=None)
    parser.add_argument("--source-y", type=float, default=None)
    parser.add_argument("--heading", type=float, default=None)
    parser.add_argument("--demo-seed", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=2, help="Render every Nth step")
    parser.add_argument("--save", type=Path, help="Write the animation to this file")
    parser.add_argument(
        "--no-show", action="store_true", help="Skip the interactive window (use with --save)"
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    from r_elegans.body import (
        BodyState,
        default_muscle_body_params,
        world_segment_centers,
    )
    from r_elegans.envs import (
        decode_sensory_policy,
        default_petri_dish_params,
        food_concentration,
        simulate_petri_dish,
    )
    from r_elegans.model import load_builtin_model

    model = load_builtin_model()
    body_params = default_muscle_body_params(12)

    if args.load is not None:
        with np.load(args.load, allow_pickle=False) as data:
            raw_policy = jnp.asarray(data["raw_policy"])
        print(f"loaded_policy={args.load}")
    else:
        from r_elegans.envs.gymnax_petri_dish import PetriDishGymnaxEnv
        from r_elegans.rl import TrainingConfig, train

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
        print(f"training the seven-parameter sensory controller with {args.algorithm.upper()} ...")
        agent, _ = train(
            env,
            env_params,
            config,
            algorithm=args.algorithm,
            seed=args.seed,
            log_every=args.log_every,
        )
        raw_policy = agent.actor.raw_sensory_policy

    print(f"trained_policy={decode_sensory_policy(raw_policy)}")

    generator = np.random.default_rng(args.demo_seed)
    if args.source_x is not None and args.source_y is not None:
        source = jnp.asarray([args.source_x, args.source_y])
    else:
        angle = generator.uniform(-np.pi, np.pi)
        radius = generator.uniform(0.65, 0.85)
        source = jnp.asarray([radius * np.cos(angle), radius * np.sin(angle)])
    heading = (
        jnp.asarray(args.heading)
        if args.heading is not None
        else jnp.asarray(generator.uniform(-np.pi, np.pi))
    )

    dish = default_petri_dish_params(source)
    _, trajectory, observations = simulate_petri_dish(
        raw_policy, dish, body_params, model.gait_params, heading=heading, steps=args.demo_steps
    )

    def centers_at(position, heading_value, joint_angles):
        state = BodyState(
            position=position,
            heading=heading_value,
            joint_angles=joint_angles,
            time=jnp.asarray(0.0),
        )
        return world_segment_centers(state, body_params.mechanics)

    segment_history = jax.vmap(centers_at)(
        trajectory.body.position, trajectory.body.heading, trajectory.body.joint_angles
    )
    segment_history = np.asarray(segment_history)  # [steps, num_segments, 2]
    head_path = np.asarray(observations.head_position)
    distance = np.asarray(observations.distance_to_source)

    minimum_distance = float(distance.min())
    success = minimum_distance < 0.12
    print(f"demo_source={np.asarray(source)} demo_heading={float(heading):.4f}")
    print(f"minimum_distance={minimum_distance:.4f} reached_food={success}")

    grid_size = 220
    extent = float(dish.radius)
    axis = np.linspace(-extent, extent, grid_size)
    grid_x, grid_y = np.meshgrid(axis, axis)
    grid_points = jnp.asarray(np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1))
    grid_concentration, _ = jax.vmap(food_concentration, in_axes=(0, None, None))(
        grid_points, jnp.asarray(0.0), dish
    )
    grid_concentration = np.asarray(grid_concentration).reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.set_facecolor("#fffdf6")
    ax.contourf(grid_x, grid_y, grid_concentration, levels=24, cmap="YlGn", alpha=0.85)
    dish_boundary = plt.Circle(
        (0, 0), extent, fill=False, color="#555a50", linewidth=2.0
    )
    ax.add_patch(dish_boundary)
    ax.plot(*np.asarray(source), marker="*", color="#c44536", markersize=16, zorder=5)
    target_circle = plt.Circle(
        np.asarray(source), 0.12, fill=False, color="#c44536", linestyle="--", linewidth=1.2
    )
    ax.add_patch(target_circle)

    (trail_line,) = ax.plot([], [], color="#277da1", linewidth=1.4, alpha=0.8)
    (body_line,) = ax.plot([], [], color="#242424", linewidth=3.0, solid_capstyle="round")
    (head_marker,) = ax.plot([], [], marker="o", color="#242424", markersize=7)
    status_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10, family="monospace"
    )

    ax.set_xlim(-extent - 0.1, extent + 0.1)
    ax.set_ylim(-extent - 0.1, extent + 0.1)
    ax.set_aspect("equal")
    ax.set_title("Body-direct RL-trained C. elegans chemotaxis")

    frame_indices = list(range(0, len(head_path), max(1, args.stride)))

    def update(frame_number: int):
        step = frame_indices[frame_number]
        body_line.set_data(segment_history[step, :, 0], segment_history[step, :, 1])
        head_marker.set_data([head_path[step, 0]], [head_path[step, 1]])
        trail_line.set_data(head_path[: step + 1, 0], head_path[: step + 1, 1])
        reached = distance[: step + 1].min() < 0.12
        status_text.set_text(
            f"t={step * 0.02:5.2f}s  distance={distance[step]:.3f}"
            + ("  FOOD REACHED" if reached else "")
        )
        return body_line, head_marker, trail_line, status_text

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices), interval=1000.0 / args.fps, blit=True
    )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        writer = "pillow" if args.save.suffix.lower() == ".gif" else "ffmpeg"
        anim.save(args.save, writer=writer, fps=args.fps)
        print(f"saved_animation={args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
