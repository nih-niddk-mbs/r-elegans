"""Train a sensory controller with RL, then animate the worm finding food.

This is the end-to-end demo for reinforcement learning against the Petri dish
(:mod:`r_elegans.envs.gymnax_petri_dish`, :mod:`r_elegans.rl`): PPO's clipped
surrogate objective by default (``--algorithm a2c`` selects a simpler
policy-gradient update instead), then a deterministic rollout of the trained
controller rendering the full 12-segment body moving through the dish toward
the food source.

``--actor analytic`` (default) trains the seven-parameter analytic
controller. ``--actor connectome`` instead trains the real 14-neuron
chemotaxis subcircuit (:mod:`r_elegans.brain.circuit`) -- pass
``--pretrained-checkpoint`` with a checkpoint from
``scripts/pretrain_connectome_circuit.py`` to fine-tune from a supervised
starting point. For the connectome actor, the animation adds a second panel
plotting each of the 14 neurons' voltage over time alongside the worm's
movement, so the sensory/interneuron/motor activity driving each turn is
visible as it happens.

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
        "--actor", choices=("analytic", "connectome"), default="analytic",
        help=(
            "Controller architecture: the seven-parameter analytic controller "
            "(default) or the real 14-neuron connectome subcircuit"
        ),
    )
    parser.add_argument(
        "--pretrained-checkpoint", type=Path, default=None,
        help=(
            "--actor connectome only: seed from a "
            "scripts/pretrain_connectome_circuit.py checkpoint instead of a "
            "random subcircuit initialization"
        ),
    )
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
        "--load", type=Path, help="Skip training; animate a saved checkpoint npz instead"
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
    from r_elegans.data.connectome import load_connectome
    from r_elegans.envs import (
        decode_sensory_policy,
        default_petri_dish_params,
        food_concentration,
        simulate_petri_dish,
    )
    from r_elegans.envs.gymnax_petri_dish import PetriDishGymnaxEnv
    from r_elegans.model import load_builtin_model
    from r_elegans.rl import (
        TrainingConfig,
        actor_from_arrays,
        deterministic_rollout,
        init_connectome_actor_params,
        make_connectome_actor_interface,
        train,
    )

    model = load_builtin_model()
    body_params = default_muscle_body_params(12)
    env = PetriDishGymnaxEnv(body_params, model.gait_params)
    env_params = env.default_params.replace(max_steps_in_episode=args.episode_steps)

    if args.actor == "connectome":
        actor_interface = make_connectome_actor_interface(dt=env_params.dt)
        connectome = load_connectome()
        if args.load is not None:
            with np.load(args.load, allow_pickle=False) as data:
                actor_params = actor_from_arrays(data)
            print(f"loaded_checkpoint={args.load}")
        else:
            init_actor = lambda: init_connectome_actor_params(
                connectome, jax.random.PRNGKey(args.seed)
            )
            initial_actor_params = None
            if args.pretrained_checkpoint is not None:
                with np.load(args.pretrained_checkpoint, allow_pickle=False) as data:
                    initial_actor_params = actor_from_arrays(data)
                print(f"loaded_pretrained_checkpoint={args.pretrained_checkpoint}")
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
            print(f"training the connectome subcircuit with {args.algorithm.upper()} ...")
            agent, _ = train(
                env,
                env_params,
                config,
                algorithm=args.algorithm,
                actor_interface=actor_interface,
                init_actor=init_actor,
                initial_actor_params=initial_actor_params,
                seed=args.seed,
                log_every=args.log_every,
            )
            actor_params = agent.actor
        print(f"trained_actor=connectome ({len(connectome.neuron_ids)}-neuron connectome, 14-neuron subcircuit)")
    else:
        if args.load is not None:
            with np.load(args.load, allow_pickle=False) as data:
                raw_policy = jnp.asarray(data["raw_policy"])
            print(f"loaded_policy={args.load}")
        else:
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

    def centers_at(position, heading_value, joint_angles):
        state = BodyState(
            position=position,
            heading=heading_value,
            joint_angles=joint_angles,
            time=jnp.asarray(0.0),
        )
        return world_segment_centers(state, body_params.mechanics)

    voltage_history = None
    if args.actor == "connectome":
        body_trajectory, distance, voltage_trajectory = deterministic_rollout(
            actor_params, env, env_params, source, heading,
            actor_interface=actor_interface, steps=args.demo_steps,
        )
        distance = np.asarray(distance)
        voltage_history = np.asarray(voltage_trajectory)  # [steps, 14]
    else:
        _, trajectory, observations = simulate_petri_dish(
            raw_policy, dish, body_params, model.gait_params, heading=heading, steps=args.demo_steps
        )
        body_trajectory = trajectory.body
        distance = np.asarray(observations.distance_to_source)

    segment_history = jax.vmap(centers_at)(
        body_trajectory.position, body_trajectory.heading, body_trajectory.joint_angles
    )
    segment_history = np.asarray(segment_history)  # [steps, num_segments, 2]
    head_path = segment_history[:, -1, :]  # the anterior-most segment, matching head_position

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

    if voltage_history is not None:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.6, 6.4))
    else:
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        ax2 = None
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
    title = (
        "Connectome-subcircuit RL-trained C. elegans chemotaxis"
        if args.actor == "connectome"
        else "Body-direct RL-trained C. elegans chemotaxis"
    )
    ax.set_title(title)

    voltage_lines = []
    now_line = None
    if ax2 is not None:
        from r_elegans.brain import SUBCIRCUIT_NEURON_NAMES

        neuron_group_colors = {
            "AWC": "#277da1",
            "ASE": "#43aa8b",
            "AIY": "#90be6d",
            "AIZ": "#f9c74f",
            "RIA": "#f8961e",
            "RMDD": "#f3722c",
            "RMDV": "#c44536",
        }
        time_axis = np.arange(len(head_path)) * 0.02
        v_min, v_max = float(voltage_history.min()), float(voltage_history.max())
        v_pad = 0.1 * max(v_max - v_min, 1e-6)

        ax2.set_facecolor("#fffdf6")
        ax2.set_xlim(0.0, max(time_axis[-1], 1e-6))
        ax2.set_ylim(v_min - v_pad, v_max + v_pad)
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("voltage (normalized)")
        ax2.set_title("Subcircuit neuron activity (solid=L, dashed=R)")

        for name in SUBCIRCUIT_NEURON_NAMES:
            color = neuron_group_colors[name[:-1]]
            linestyle = "-" if name.endswith("L") else "--"
            (line,) = ax2.plot([], [], color=color, linestyle=linestyle, linewidth=1.3)
            voltage_lines.append(line)
        now_line = ax2.axvline(0.0, color="#242424", linewidth=1.0, alpha=0.6)

        legend_handles = [
            plt.Line2D([0], [0], color=color, linewidth=2.0, label=label)
            for label, color in neuron_group_colors.items()
        ]
        ax2.legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=2, framealpha=0.85)

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
        artists = [body_line, head_marker, trail_line, status_text]
        if ax2 is not None:
            for index, line in enumerate(voltage_lines):
                line.set_data(time_axis[: step + 1], voltage_history[: step + 1, index])
            now_line.set_xdata([time_axis[step], time_axis[step]])
            artists.extend(voltage_lines)
            artists.append(now_line)
        return artists

    fig.tight_layout()
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
