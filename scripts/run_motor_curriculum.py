"""Unattended mbslnx2 CPU motor stages, with resume into slow traveling bends.

Only a seed that passes its bend gate advances. No RL or food task is launched.
This supervisor imports no JAX; affinity is set before each worker imports it.
--resume-latest or --resume-run continues each passed alternation seed into
wave fitting, with fresh optimizers and the previous actor weights.
"""
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


def stage_command(repo, output, seed, stage, minutes, iterations):
    if stage == "bend":
        return [sys.executable, "-u", str(repo / "scripts/train_motor_bend.py"),
                "--seed", str(seed), "--iterations", "500", "--output-dir", str(output / "bend")]
    if stage != "alternation":
        raise ValueError("Unknown motor stage")
    return [sys.executable, "-u", str(repo / "scripts/train_motor_alternation.py"),
            "--seed", str(seed), "--minutes", str(minutes), "--iterations", str(iterations),
            "--initial-actor", str(output / "bend/actor_best.npz"),
            "--output-dir", str(output / "alternation")]


def stage_passed(result):
    """A usable checkpoint cannot turn a failed/interrupted run into success."""
    return (result.get("passed") is True and result.get("exit_code") == 0
            and result.get("status", "completed") == "completed")


def resume_checkpoints(root, seeds):
    """Require the matching seed's successful alternation AND its checkpoint."""
    summary = json.loads((root / "summary.json").read_text())
    workers = {w["seed"]: w for w in summary["workers"]}
    paths = {}
    for seed in seeds:
        result = workers.get(seed, {}).get("stages", {}).get("alternation", {})
        path = root / f"seed-{seed}/alternation/actor_best.npz"
        if not stage_passed(result) or not path.is_file():
            raise ValueError(f"Seed {seed} lacks a passed alternation stage/checkpoint in {root}")
        paths[seed] = path
    return paths


def find_resume_run(output_root, seeds):
    for root in sorted(output_root.glob("motor-*"), reverse=True):
        try:
            return root, resume_checkpoints(root, seeds)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    raise ValueError("No completed motor run with passed alternation checkpoints for all requested seeds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=60.)
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--cores-per-worker", type=positive_int, default=4)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--alternation-iterations", type=positive_int, default=100000)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-run", type=Path, help="Start wave stage from this passed motor run")
    resume.add_argument("--resume-latest", action="store_true", help="Start waves from latest passed motor run under output-dir")
    args = parser.parse_args()
    if not sys.platform.startswith("linux") or not hasattr(os, "sched_getaffinity"):
        parser.error("Run on Linux, such as mbslnx2")
    if not math.isfinite(args.minutes) or args.minutes <= 0:
        parser.error("Minutes must be finite and positive")
    dependencies = subprocess.run([sys.executable, "-c", "import jax, optax, flax"], capture_output=True, text=True)
    if dependencies.returncode:
        parser.error(f"Training dependencies unavailable in {sys.executable}; activate conda r-elegans first.\n{dependencies.stderr}")
    candidates, known = physical_core_cpus(sorted(os.sched_getaffinity(0)))
    if not known:
        parser.error("Physical CPU topology is required")
    try:
        groups = worker_cpu_groups(candidates, args.workers, args.cores_per_worker)
    except ValueError as error:
        parser.error(str(error))
    repo = Path(__file__).resolve().parents[1]
    output_root = args.output_dir.expanduser().resolve()
    if output_root == repo or repo in output_root.parents:
        parser.error("Results must stay outside Git")
    seeds = list(range(args.seed_start, args.seed_start + args.workers))
    resume_root, initial_actors = None, {}
    try:
        if args.resume_run:
            resume_root = args.resume_run.expanduser().resolve()
            initial_actors = resume_checkpoints(resume_root, seeds)
        elif args.resume_latest:
            resume_root, initial_actors = find_resume_run(output_root, seeds)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    prefix = "wave-" if initial_actors else "motor-"
    root = output_root / (prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    root.mkdir(parents=True, exist_ok=False)
    deadline = time.perf_counter() + args.minutes * 60
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"Run directory: {root}\nSupervised motor curriculum; no RL; {args.workers} independent seeds", flush=True)
    if resume_root:
        print(f"Resuming each seed's passed alternation checkpoint from: {resume_root}", flush=True)
    jobs = []
    with ExitStack() as files:
        def launch(job, stage):
            remaining = max(.01, (deadline - time.perf_counter()) / 60.)
            if stage == "wave":
                command = [sys.executable, "-u", str(repo / "scripts/train_motor_alternation.py"),
                    "--stage", "wave", "--seed", str(job["seed"]), "--minutes", str(remaining),
                    "--iterations", str(args.alternation_iterations),
                    "--initial-actor", str(initial_actors[job["seed"]]),
                    "--output-dir", str(job["output"] / "wave")]
            else:
                command = stage_command(repo, job["output"], job["seed"], stage, remaining, args.alternation_iterations)
            child_env = os.environ.copy()
            child_env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
                JAX_DEFAULT_MATMUL_PRECISION="highest", OMP_NUM_THREADS=str(len(job["cpus"])),
                OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
            log_path = job["output"] / (stage + ".log")
            log = files.enter_context(log_path.open("w"))
            job.update(stage=stage, stop_sent=None, process=subprocess.Popen(command,
                cwd=repo, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                preexec_fn=lambda cpus=job["cpus"]: os.sched_setaffinity(0, cpus)))
            print(f"seed={job['seed']} stage={stage} pid={job['process'].pid} CPUs={job['cpus']} log={log_path}", flush=True)
        try:
            for index, cpus in enumerate(groups):
                seed = args.seed_start + index
                output = root / f"seed-{seed}"
                output.mkdir()
                job = dict(seed=seed, cpus=cpus, output=output, done=False, stages={})
                launch(job, "wave" if initial_actors else "bend")
                jobs.append(job)
            next_status = time.perf_counter() + 60
            while any(not job["done"] for job in jobs):
                now = time.perf_counter()
                for job in jobs:
                    if job["done"]:
                        continue
                    process = job["process"]
                    if process.poll() is not None:
                        path = job["output"] / job["stage"] / "summary.json"
                        result = json.loads(path.read_text()) if path.exists() else {"passed": False, "status": "missing_summary"}
                        job["stages"][job["stage"]] = dict(exit_code=process.returncode, **result)
                        print(f"seed={job['seed']} stage={job['stage']} passed={result['passed']} exit_code={process.returncode}", flush=True)
                        if (job["stage"] == "bend" and stage_passed(job["stages"]["bend"])
                                and not stopped and now < deadline):
                            launch(job, "alternation")
                        else:
                            job["done"] = True
                        continue
                    if (stopped or now > deadline + 180) and job["stop_sent"] is None:
                        process.terminate()
                        job["stop_sent"] = now
                    elif job["stop_sent"] is not None and now - job["stop_sent"] > 30:
                        process.kill()
                if now >= next_status:
                    print("Running: " + str([(j["seed"], j["stage"]) for j in jobs if not j["done"]]), flush=True)
                    next_status = now + 60
                time.sleep(1)
        finally:
            for job in jobs:
                if job["process"].poll() is None:
                    job["process"].terminate()
            for job in jobs:
                try:
                    job["process"].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    job["process"].kill()
                    job["process"].wait()
    summary = dict(requested_minutes=args.minutes, stopped=stopped, resume_run=str(resume_root) if resume_root else None,
        workers=[dict(seed=j["seed"], cpus=j["cpus"], stages=j["stages"]) for j in jobs])
    final_stage = "wave" if initial_actors else "alternation"
    passed = all(stage_passed(j["stages"].get(final_stage, {})) for j in jobs)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for worker in summary["workers"]:
        print(json.dumps(worker), flush=True)
    print(f"MOTOR_CURRICULUM_EXIT_CODE={0 if passed and not stopped else 2}", flush=True)
    return 0 if passed and not stopped else 2


if __name__ == "__main__":
    raise SystemExit(main())
