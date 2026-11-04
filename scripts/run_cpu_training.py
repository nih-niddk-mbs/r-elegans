"""Launch a timed four-seed PPO pilot on disjoint Linux physical cores.

Run with nohup in the r-elegans environment. Training artifacts remain outside
the repo; each seed has its own log, checkpoints and evaluation reports.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

if __package__:
    from .benchmark_cpu import physical_core_cpus, positive_int
    from .benchmark_cpu_parallel import worker_cpu_groups
else:
    from benchmark_cpu import physical_core_cpus, positive_int
    from benchmark_cpu_parallel import worker_cpu_groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--cores-per-worker", type=positive_int, default=4)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-envs", type=positive_int, default=None)
    parser.add_argument("--episode-steps", type=positive_int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--actor", choices=("connectome", "full-network"), default="full-network")
    parser.add_argument("--train-group", choices=("all", "neural", "nmj", "sensory"), default="all")
    parser.add_argument("--train-neurons", nargs="+")
    parser.add_argument("--initial-actor", type=Path)
    args = parser.parse_args()
    if args.num_envs is None:
        args.num_envs = 32 if args.actor == "full-network" else 128
    if args.episode_steps is None:
        args.episode_steps = 250
    if args.learning_rate is None:
        args.learning_rate = 1e-4 if args.actor == "full-network" else 1e-3
    if args.actor != "full-network" and (args.train_group != "all" or args.train_neurons or args.initial_actor):
        parser.error("Subset controls and initial-actor require --actor full-network")
    if not sys.platform.startswith("linux") or not hasattr(os, "sched_getaffinity"):
        parser.error("Run on a Linux workstation such as mbslnx2")
    if not math.isfinite(args.minutes) or args.minutes <= 0:
        parser.error("--minutes must be finite and positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if args.num_envs % 4:
        parser.error("--num-envs must be divisible by four PPO minibatches")
    candidates, known = physical_core_cpus(sorted(os.sched_getaffinity(0)))
    if not known:
        parser.error("Physical topology is required for disjoint core allocations")
    try:
        groups = worker_cpu_groups(candidates, args.workers, args.cores_per_worker)
    except ValueError as error:
        parser.error(str(error))
    repo = Path(__file__).resolve().parents[1]
    output_root = args.output_dir.expanduser().resolve()
    if output_root == repo or repo in output_root.parents:
        parser.error("--output-dir must be outside the repository")
    run_dir = output_root / datetime.now(timezone.utc).strftime("training-%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Exploratory {args.actor} normalized-dish PPO pilot; {args.workers} independent seeds", flush=True)
    stopped = False

    def stop_workers(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop_workers)
    signal.signal(signal.SIGTERM, stop_workers)
    jobs = []
    with ExitStack() as files:
        try:
            for index, cpus in enumerate(groups):
                seed = args.seed_start + index
                output = run_dir / f"seed-{seed}"
                output.mkdir()
                child_env = os.environ.copy()
                child_env.update(
                    JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
                    JAX_DEFAULT_MATMUL_PRECISION="highest",
                    OMP_NUM_THREADS=str(len(cpus)), OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                )
                command = [
                    sys.executable, "-u", str(repo / "scripts/train_timed.py"),
                    "--output-dir", str(output), "--seed", str(seed),
                    "--minutes", str(args.minutes), "--num-envs", str(args.num_envs),
                    "--episode-steps", str(args.episode_steps),
                    "--learning-rate", str(args.learning_rate),
                    "--actor", args.actor, "--train-group", args.train_group,
                ]
                if args.train_neurons:
                    command.extend(["--train-neurons", *args.train_neurons])
                if args.initial_actor:
                    command.extend(["--initial-actor", str(args.initial_actor.expanduser().resolve())])
                log = files.enter_context((output / "worker.log").open("w"))
                process = subprocess.Popen(
                    command, cwd=repo, env=child_env, stdout=log,
                    stderr=subprocess.STDOUT,
                    preexec_fn=lambda selected=cpus: os.sched_setaffinity(0, selected),
                )
                jobs.append(dict(seed=seed, process=process, output=output,
                                 cpus=cpus, started=time.perf_counter(), stop_sent=None))
                print(f"seed={seed} pid={process.pid} CPUs={cpus} log={output / 'worker.log'}", flush=True)
            next_status = time.perf_counter() + 60
            while any(job["process"].poll() is None for job in jobs):
                now = time.perf_counter()
                for job in jobs:
                    process = job["process"]
                    if process.poll() is not None:
                        continue
                    overrun = now - job["started"] > args.minutes * 60 + 300
                    if (stopped or overrun) and job["stop_sent"] is None:
                        process.terminate()
                        job["stop_sent"] = now
                    elif job["stop_sent"] is not None and now - job["stop_sent"] > 30:
                        process.kill()
                if now >= next_status:
                    running = [j["seed"] for j in jobs if j["process"].poll() is None]
                    print(f"Still running seeds: {running}", flush=True)
                    next_status = now + 60
                time.sleep(2)
        finally:
            for job in jobs:
                process = job["process"]
                if process.poll() is None:
                    process.terminate()
            for job in jobs:
                process = job["process"]
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    summary = {"requested_minutes": args.minutes, "workers": [], "interrupted": stopped}
    failed = stopped or len(jobs) != args.workers
    for job in jobs:
        path = job["output"] / "summary.json"
        result = json.loads(path.read_text()) if path.exists() else {"status": "missing_summary"}
        entry = dict(seed=job["seed"], cpu_affinity=job["cpus"],
                     exit_code=job["process"].returncode, **result)
        summary["workers"].append(entry)
        failed = failed or entry["exit_code"] != 0 or entry["status"] != "completed"
        print(json.dumps(entry), flush=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"CPU_TRAINING_EXIT_CODE={int(bool(failed))}", flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
