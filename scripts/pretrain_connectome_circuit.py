"""Supervised-pretrain the connectome subcircuit against an existing controller.

Cold-starting reinforcement learning on an untrained ~100-parameter recurrent
network is a much harder exploration problem than the seven-parameter
analytic controller's. This script instead first fits the subcircuit
(:mod:`r_elegans.brain.circuit`, :mod:`r_elegans.rl.connectome_actor`) to
imitate an already-competent teacher -- by default the bundled
differentiable-fit controller -- via :mod:`r_elegans.rl.pretrain`. The
resulting checkpoint is meant to seed RL fine-tuning
(``scripts/train_rl_chemotaxis.py --actor connectome --pretrained-checkpoint``).

Requires the optional ``env`` extra: ``pip install -e ".[env]"``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.body import default_muscle_body_params
from r_elegans.data.connectome import load_connectome
from r_elegans.envs.gymnax_petri_dish import PetriDishGymnaxEnv
from r_elegans.model import load_builtin_model
from r_elegans.rl import (
    actor_to_arrays,
    collect_teacher_trajectories,
    init_connectome_actor_params,
    pretrain_fit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=None,
        help="npz with a raw_policy array; defaults to the bundled differentiable-fit controller",
    )
    parser.add_argument("--num-episodes", type=int, default=64)
    parser.add_argument("--episode-steps", type=int, default=250)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--output", type=Path, default=Path("results/behavior/connectome_pretrain_v1.npz")
    )
    args = parser.parse_args()

    model = load_builtin_model()
    body_params = default_muscle_body_params(12)
    env = PetriDishGymnaxEnv(body_params, model.gait_params)
    env_params = env.default_params.replace(max_steps_in_episode=args.episode_steps)

    if args.teacher_checkpoint is not None:
        with np.load(args.teacher_checkpoint, allow_pickle=False) as data:
            teacher_raw_policy = jnp.asarray(data["raw_policy"])
    else:
        teacher_raw_policy = model.raw_sensory_policy

    rng = jax.random.PRNGKey(args.seed)
    data_rng, init_rng = jax.random.split(rng)
    obs_batch, action_batch = collect_teacher_trajectories(
        env, env_params, teacher_raw_policy, data_rng, args.num_episodes, args.episode_steps
    )
    print(f"teacher_trajectories: obs={obs_batch.shape} action={action_batch.shape}")

    connectome = load_connectome()
    actor = init_connectome_actor_params(connectome, init_rng)

    trained_actor, losses = pretrain_fit(
        actor,
        obs_batch,
        action_batch,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        weight_l2=args.weight_l2,
        log_every=args.log_every,
        dt=env_params.dt,
    )
    print(f"final_loss={losses[-1]:.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        losses=np.asarray(losses),
        **{key: np.asarray(value) for key, value in actor_to_arrays(trained_actor).items()},
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
