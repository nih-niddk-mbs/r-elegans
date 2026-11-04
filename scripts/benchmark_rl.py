"""Measure PPO compilation and synchronized training throughput.

Uses the legacy normalized dish and either the analytic controller or the
14-neuron actor. This is a compute benchmark, not biological validation. Select
one GPU with CUDA_VISIBLE_DEVICES and pass --require-gpu to forbid CPU fallback.
The JSON report goes to stdout; progress goes to stderr. No fits are saved.
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
import statistics
import subprocess
import sys
import time

import jax
import numpy as np

from r_elegans.body import default_muscle_body_params
from r_elegans.envs.gymnax_petri_dish import PetriDishGymnaxEnv
from r_elegans.model import load_builtin_model
from r_elegans.rl import (
    ANALYTIC_ACTOR_INTERFACE,
    AgentParams,
    TrainingConfig,
    init_actor_params,
    init_connectome_actor_params,
    init_critic_params,
    make_connectome_actor_interface,
    make_ppo_train_step,
)


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", choices=("analytic", "connectome"), default="connectome")
    parser.add_argument("--num-envs", type=positive_int, default=32)
    parser.add_argument("--num-steps", type=positive_int, default=128)
    parser.add_argument("--updates", type=positive_int, default=5)
    parser.add_argument("--warmup-updates", type=positive_int, default=1)
    parser.add_argument("--update-epochs", type=positive_int, default=2)
    parser.add_argument("--num-minibatches", type=positive_int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    backend = jax.default_backend()
    if args.require_gpu and backend != "gpu":
        parser.error(f"GPU required, but JAX selected {backend!r}")
    if args.num_envs % args.num_minibatches:
        parser.error("--num-envs must be divisible by --num-minibatches")

    model = load_builtin_model()
    env = PetriDishGymnaxEnv(default_muscle_body_params(12), model.gait_params)
    env_params = env.default_params.replace(max_steps_in_episode=args.num_steps)
    config = TrainingConfig(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_updates=args.updates,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
    )
    rng, actor_key, critic_key = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    if args.actor == "connectome":
        actor = init_connectome_actor_params(model.connectome, actor_key)
        interface = make_connectome_actor_interface(dt=env_params.dt)
    else:
        actor = init_actor_params()
        interface = ANALYTIC_ACTOR_INTERFACE
    agent = AgentParams(actor=actor, critic=init_critic_params(critic_key))
    train_step, optimizer = make_ppo_train_step(env, config, actor_interface=interface)
    opt_state = optimizer.init(agent)
    # Finish setup work before timing compilation or a training update.
    jax.block_until_ready((agent, opt_state, env_params, rng))

    print(f"Compiling {args.actor} PPO on {backend}...", file=sys.stderr, flush=True)
    started = time.perf_counter()
    executable = train_step.lower(agent, opt_state, env_params, rng).compile()
    compile_seconds = time.perf_counter() - started

    def update():
        nonlocal agent, opt_state, rng
        result = executable(agent, opt_state, env_params, rng)
        # GPU dispatch is asynchronous: synchronize all outputs before stopping
        # the timer, rather than measuring only Python launch latency.
        agent, opt_state, rng, metrics = jax.block_until_ready(result)
        return metrics

    for _ in range(args.warmup_updates):
        update()

    durations = []
    for index in range(args.updates):
        started = time.perf_counter()
        metrics = update()
        durations.append(time.perf_counter() - started)
        print(
            f"update={index + 1}/{args.updates} seconds={durations[-1]:.4f}",
            file=sys.stderr,
            flush=True,
        )

    # Validation and host copies occur outside the timed updates. A fast run
    # with nonfinite parameters must never be reported as a successful result.
    for leaf in jax.tree_util.tree_leaves((agent, opt_state, metrics)):
        if not np.all(np.isfinite(np.asarray(leaf))):
            raise RuntimeError("Nonfinite training state; benchmark rejected")

    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    )
    working_tree = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    )
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(working_tree.stdout) if working_tree.returncode == 0 else None,
        "hostname": platform.node(),
        "architecture": platform.machine(),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("jax", "jaxlib", "gymnax", "optax", "diffrax", "numpy")
        },
        "backend": backend,
        "devices": [str(device) for device in jax.devices()],
        "device_kinds": [device.device_kind for device in jax.devices()],
        "matmul_precision": str(jax.config.jax_default_matmul_precision),
        "environment": "legacy_normalized_petri_dish",
        "actor": args.actor,
        "actor_neurons": 14 if args.actor == "connectome" else None,
        "seed": args.seed,
        "training_config": config._asdict(),
        "warmup_updates": args.warmup_updates,
        "compile_seconds": compile_seconds,
        "update_seconds": durations,
        "median_update_seconds": statistics.median(durations),
        "environment_steps_per_second": (
            args.num_envs * args.num_steps * args.updates / math.fsum(durations)
        ),
        "final_metrics": {name: float(value) for name, value in metrics.items()},
        "finite_state": True,
    }
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
