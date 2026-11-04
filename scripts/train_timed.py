"""Wall-clock-bounded PPO training with checkpoints and independent evaluation.

Uses the normalized dish with either the legacy 14-neuron steering actor or
the full 302-neuron/95-muscle actor. run_cpu_training.py launches independent
CPU seeds; this script can also run on one GPU. Artifacts stay outside the
repository. These are exploratory, biologically uncalibrated experiments.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import time

from flax import serialization
import jax
import numpy as np

from r_elegans.body import default_muscle_body_params
from r_elegans.envs.gymnax_petri_dish import PetriDishGymnaxEnv
from r_elegans.model import load_builtin_model
from r_elegans.rl import (
    AgentParams, TrainingConfig, actor_to_arrays, init_connectome_actor_params,
    init_critic_params, make_connectome_actor_interface, make_ppo_train_step,
)

if __package__:
    from .train_rl_chemotaxis import evaluate_via_env, make_trials
else:
    from train_rl_chemotaxis import evaluate_via_env, make_trials


def finite_tree(tree) -> bool:
    return all(np.all(np.isfinite(np.asarray(leaf)))
               for leaf in jax.tree_util.tree_leaves(tree))


def atomic_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def save_json(path: Path, content: dict) -> None:
    atomic_bytes(path, (json.dumps(content, indent=2, allow_nan=False) + "\n").encode())


def save_actor(path: Path, actor) -> None:
    from r_elegans.rl.full_network_actor import FullNetworkActorParams, actor_to_arrays as full_arrays
    full = isinstance(actor, FullNetworkActorParams)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, actor="full-network" if full else "connectome", **{
            name: np.asarray(value) for name, value in (full_arrays(actor) if full else actor_to_arrays(actor)).items()
        })
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def checkpoint(path: Path, agent, opt_state, rng, updates: int) -> None:
    state = dict(agent=agent, opt_state=opt_state, rng=rng,
                 updates=np.asarray(updates, dtype=np.int64))
    if not finite_tree(state):
        raise RuntimeError("Refusing to overwrite a checkpoint with nonfinite state")
    atomic_bytes(path, serialization.to_bytes(jax.device_get(state)))


def summarize_evaluation(metrics: dict) -> dict:
    arrays = {name: np.asarray(value) for name, value in metrics.items()}
    if not finite_tree(arrays):
        raise RuntimeError("Nonfinite evaluation result")
    return {
        "success_rate": float(np.mean(arrays["success"])),
        "mean_minimum_distance": float(np.mean(arrays["minimum_distance"])),
        "mean_final_distance": float(np.mean(arrays["final_distance"])),
        "trials": {name: value.tolist() for name, value in arrays.items()},
    }


def validation_rank(metrics: dict) -> tuple[float, float]:
    """Prefer success, then minimum distance; final test trials never select weights."""
    return metrics["success_rate"], -metrics["mean_minimum_distance"]


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actor", choices=("connectome", "full-network"), default="full-network")
    parser.add_argument("--train-group", choices=("all", "neural", "nmj", "sensory"), default="all")
    parser.add_argument("--train-neurons", nargs="+", help="Freeze parameters outside these neurons; simulate all 302")
    parser.add_argument("--initial-actor", type=Path, help="Full-network actor NPZ to start another training stage; optimizer resets")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--checkpoint-seconds", type=float, default=300.0)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--max-updates", type=int, default=None,
                        help="Optional update cap for smoke tests; normally use the time budget")
    args = parser.parse_args()
    if args.num_envs is None:
        args.num_envs = 32 if args.actor == "full-network" else 128
    if args.episode_steps is None:
        args.episode_steps = 250
    if args.learning_rate is None:
        args.learning_rate = 1e-4 if args.actor == "full-network" else 1e-3
    if args.actor != "full-network" and (args.train_neurons or args.train_group != "all" or args.initial_actor):
        parser.error("Subset controls require --actor full-network")
    if not math.isfinite(args.minutes) or args.minutes <= 0:
        parser.error("--minutes must be finite and positive")
    if not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds <= 0:
        parser.error("--checkpoint-seconds must be finite and positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if min(args.eval_episodes, args.validation_episodes, args.eval_steps) <= 0:
        parser.error("Evaluation counts and steps must be positive")
    if args.max_updates is not None and args.max_updates <= 0:
        parser.error("--max-updates must be positive")
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.expanduser().resolve()
    if output == repo or repo in output.parents:
        parser.error("--output-dir must be outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        parser.error("Output already contains a run; use a fresh output directory")

    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    budget = args.minutes * 60.0
    # Reserve up to one minute for test evaluation and final writes. A single
    # compiled operation cannot be interrupted at an exact wall-clock deadline.
    training_deadline = started + budget - min(60.0, budget * 0.1)
    model = load_builtin_model()
    if args.actor == "full-network":
        from r_elegans.envs.full_network_petri_dish import FullNetworkPetriDishEnv
        env = FullNetworkPetriDishEnv(default_muscle_body_params(12))
    else:
        env = PetriDishGymnaxEnv(default_muscle_body_params(12), model.gait_params)
    params = env.default_params.replace(max_steps_in_episode=args.episode_steps)
    config = TrainingConfig(
        num_envs=args.num_envs, num_steps=args.episode_steps, num_updates=1,
        update_epochs=args.update_epochs, num_minibatches=args.num_minibatches,
        learning_rate=args.learning_rate,
    )
    rng, init_key, critic_key = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    if args.actor == "full-network":
        from r_elegans.rl.full_network_actor import init_full_network_actor, make_full_network_actor_interface
        initial_actor = init_full_network_actor(model, init_key)
        if args.initial_actor:
            from r_elegans.rl.full_network_actor import actor_from_arrays
            with np.load(args.initial_actor, allow_pickle=False) as arrays:
                if str(arrays["actor"]) != "full-network":
                    parser.error("--initial-actor must be a full-network checkpoint")
                initial_actor = actor_from_arrays(arrays)
            expected = init_full_network_actor(model, init_key).neural
            for name in ("chemical_mask", "gap_mask"):
                if not np.array_equal(getattr(initial_actor.neural, name), getattr(expected, name)):
                    parser.error("Checkpoint topology differs from bundled connectome")
        interface = make_full_network_actor_interface(model, dt=params.dt,
            train_group=args.train_group, train_neurons=args.train_neurons)
    else:
        initial_actor = init_connectome_actor_params(model.connectome, init_key)
        interface = make_connectome_actor_interface(dt=params.dt)
    agent = AgentParams(actor=initial_actor, critic=init_critic_params(critic_key, observation_dim=env.obs_shape[0]))
    train_step, optimizer = make_ppo_train_step(env, config, actor_interface=interface)
    opt_state = optimizer.init(agent)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True
    )
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "seed": args.seed, "minutes": args.minutes,
        "environment": env.name, "actor": args.actor,
        "actor_neurons": 302 if args.actor == "full-network" else 14,
        "train_group": args.train_group, "train_neurons": args.train_neurons,
        "initial_actor": str(args.initial_actor) if args.initial_actor else None,
        "neuron_ids": list(model.neuron_ids) if args.actor == "full-network" else None,
        "body_units": "legacy_normalized_uncalibrated",
        "chemical_edges": int(np.count_nonzero(model.connectome.chemical_counts)),
        "gap_pairs": int(np.count_nonzero(np.triu(np.asarray(model.connectome.gap_counts), 1))),
        "nmj_edges": int(np.count_nonzero(model.neuromuscular_params.synapse_weights)),
        "full_network_integrator": "frozen_conductance_exponential_euler_4_substeps" if args.actor == "full-network" else None,
        "nmj_sign_constraints": "known signs fixed, unknown signs trainable" if args.actor == "full-network" else None,
        "backend": jax.default_backend(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "git_revision": revision.stdout.strip(),
        "packages": {name: version(name) for name in ("jax", "jaxlib", "gymnax", "optax", "flax", "numpy", "diffrax")},
        "training_config": config._asdict() | {"num_updates": "wall_clock_bounded"},
        "matmul_precision": str(jax.config.jax_default_matmul_precision),
        "initialization": "anatomical priors with seed-dependent sensory gains",
        "validation_seed": 37, "test_seed": 19,
        "evaluation_steps": args.eval_steps,
        "validation_episodes": args.validation_episodes, "test_episodes": args.eval_episodes,
    }
    save_json(output / "manifest.json", manifest)
    save_actor(output / "actor_initial.npz", initial_actor)
    checkpoint(output / "latest.msgpack", agent, opt_state, rng, 0)

    def evaluate(actor, trial_seed, count):
        sources, headings = make_trials(trial_seed, count)
        result = evaluate_via_env(
            actor, env, params, sources, headings,
            actor_interface=interface, steps=args.eval_steps,
        )
        return summarize_evaluation(result)

    updates = 0
    best_actor = initial_actor
    error = None
    status = "completed"
    try:
        initial_test = evaluate(initial_actor, 19, args.eval_episodes)
        save_json(output / "initial_test.json", initial_test)
        best_validation = evaluate(initial_actor, 37, args.validation_episodes)
        save_actor(output / "actor_best.npz", best_actor)
        save_json(output / "best_validation.json", best_validation | {"update": 0})
        print(f"seed={args.seed} initial_test_success={initial_test['success_rate']:.1%}", flush=True)
        next_checkpoint = time.perf_counter() + args.checkpoint_seconds
        with (output / "metrics.jsonl").open("a", buffering=1) as history:
            while (time.perf_counter() < training_deadline and not stop_requested
                   and (args.max_updates is None or updates < args.max_updates)):
                candidate, candidate_opt, candidate_rng, metrics = jax.block_until_ready(
                    train_step(agent, opt_state, params, rng)
                )
                if not finite_tree((candidate, candidate_opt, metrics)):
                    raise RuntimeError("Nonfinite PPO update; previous finite state retained")
                agent, opt_state, rng = candidate, candidate_opt, candidate_rng
                updates += 1
                numeric = {name: float(value) for name, value in metrics.items()}
                numeric.update(update=updates, elapsed_seconds=time.perf_counter() - started)
                history.write(json.dumps(numeric, allow_nan=False) + "\n")
                if updates == 1 or updates % 20 == 0:
                    print(f"seed={args.seed} update={updates} elapsed_min={numeric['elapsed_seconds']/60:.1f} reward={numeric['mean_reward']:.4f} train_success={numeric['success_rate']:.1%}", flush=True)
                if time.perf_counter() >= next_checkpoint:
                    checkpoint(output / "latest.msgpack", agent, opt_state, rng, updates)
                    save_actor(output / "actor_latest.npz", agent.actor)
                    validation = evaluate(agent.actor, 37, args.validation_episodes)
                    save_json(output / "validation_latest.json", validation | {"update": updates})
                    if validation_rank(validation) > validation_rank(best_validation):
                        best_validation, best_actor = validation, agent.actor
                        save_actor(output / "actor_best.npz", best_actor)
                        save_json(output / "best_validation.json", validation | {"update": updates})
                    print(f"seed={args.seed} checkpoint update={updates} validation_success={validation['success_rate']:.1%}", flush=True)
                    next_checkpoint = time.perf_counter() + args.checkpoint_seconds
        if stop_requested:
            status = "interrupted"
        if updates == 0 and not stop_requested:
            raise RuntimeError("Time budget expired before any PPO update completed")
        # Select the final candidate using validation trials only.
        validation = evaluate(agent.actor, 37, args.validation_episodes)
        if validation_rank(validation) > validation_rank(best_validation):
            best_actor = agent.actor
            save_actor(output / "actor_best.npz", best_actor)
            save_json(output / "best_validation.json", validation | {"update": updates})
        save_json(output / "final_test.json", evaluate(agent.actor, 19, args.eval_episodes))
        save_json(output / "best_test.json", evaluate(best_actor, 19, args.eval_episodes))
    except Exception as exception:
        status, error = "failed", repr(exception)
        print(f"seed={args.seed} ERROR: {error}", flush=True)
    finally:
        checkpoint(output / "latest.msgpack", agent, opt_state, rng, updates)
        save_actor(output / "actor_final.npz", agent.actor)
        summary = {
            "status": status, "error": error, "updates": updates,
            "environment_steps": updates * args.num_envs * args.episode_steps,
            "elapsed_seconds": time.perf_counter() - started,
            "requested_minutes": args.minutes,
            "stop_reason": (
                "failure" if status == "failed" else
                "signal" if stop_requested else
                "update_cap" if args.max_updates is not None and updates >= args.max_updates
                else "time_budget"
            ),
        }
        for name in ("initial_test", "final_test", "best_test"):
            path = output / f"{name}.json"
            if path.exists():
                metrics = json.loads(path.read_text())
                summary[name] = {key: metrics[key] for key in (
                    "success_rate", "mean_minimum_distance", "mean_final_distance"
                )}
        save_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
