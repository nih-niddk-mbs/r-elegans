"""Unattended mbslnx2 CPU motor stages, with resume into slow traveling bends.

Only a seed that passes its bend gate advances. No RL or food task is launched.
This supervisor imports no JAX; affinity is set before each worker imports it.
--resume-latest or --resume-run continues each passed alternation seed into
wave fitting, with fresh optimizers and the previous actor weights. Assessment
and amplitude-continuation modes instead read passed wave checkpoints.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import hashlib
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

if __package__:
    from .experiment_records import capture_run_record
    from .benchmark_cpu import physical_core_cpus, positive_int
    from .benchmark_cpu_parallel import worker_cpu_groups
else:
    from experiment_records import capture_run_record
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


def baseline_valid(result):
    """Weak but real propulsion can justify a small amplitude step, not promotion."""
    return (result.get("status") == "completed" and result.get("exit_code") in (0, 2)
            and result.get("error") is None and len(result.get("periods", [])) >= 3
            and all(p.get("detectable_translation") is True and p.get("approximately_straight") is True
                    for p in result["periods"]))


def resume_checkpoints(root, seeds, stage="alternation"):
    """Require the matching seed's successful source stage AND its checkpoint."""
    summary = json.loads((root / "summary.json").read_text())
    workers = {w["seed"]: w for w in summary["workers"]}
    paths = {}
    for seed in seeds:
        result = workers.get(seed, {}).get("stages", {}).get(stage, {})
        path = root / f"seed-{seed}/{stage}/actor_best.npz"
        if not stage_passed(result) or not path.is_file():
            raise ValueError(f"Seed {seed} lacks a passed {stage} stage/checkpoint in {root}")
        paths[seed] = path
    return paths


def find_resume_run(output_root, seeds, stage="alternation"):
    pattern = "wave-*" if stage == "wave" else "motor-*"
    for root in sorted(output_root.glob(pattern), reverse=True):
        try:
            return root, resume_checkpoints(root, seeds, stage)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    raise ValueError(f"No completed run with passed {stage} checkpoints for all requested seeds")


def refinement_checkpoints(root, seeds):
    """Completed candidates with both control assays may be refined, not promoted."""
    workers = {w["seed"]: w for w in json.loads((root / "summary.json").read_text())["workers"]}
    paths = {}
    for seed in seeds:
        stages = workers.get(seed, {}).get("stages", {})
        for name in ("reverse", "forward_retention", "reverse_locomotion"):
            result = stages.get(name, {})
            if (result.get("status") != "completed" or result.get("error") is not None
                    or result.get("exit_code") not in (0, 2)
                    or (name != "reverse" and result.get("control_passed") is not True)):
                raise ValueError(f"Seed {seed} lacks completed, controlled forward/reverse results in {root}")
        path = root / f"seed-{seed}/reverse/actor_best.npz"
        if not path.is_file():
            raise ValueError(f"Missing reverse checkpoint: {path}")
        manifest = json.loads((path.parent / "manifest.json").read_text())
        field = "forward_reference_actor" if "forward_reference_actor" in manifest else "initial_actor"
        digest = "forward_reference_sha256" if field == "forward_reference_actor" else "initial_actor_sha256"
        reference = Path(manifest[field]).expanduser().resolve()
        if hashlib.sha256(reference.read_bytes()).hexdigest() != manifest.get(digest):
            raise ValueError(f"Seed {seed}: original forward reference checksum mismatch")
        paths[seed] = path
    return paths


def find_refinement_run(output_root, seeds):
    for root in sorted(output_root.glob("reverse-*"), reverse=True):
        try:
            return root, refinement_checkpoints(root, seeds)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    raise ValueError("No completed reverse cohort with valid control assays and original forward references")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=60.)
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--cores-per-worker", type=positive_int, default=4)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--alternation-iterations", type=positive_int, default=100000)
    parser.add_argument("--assess-locomotion", action="store_true", help="Diagnose passed wave checkpoints without training")
    parser.add_argument("--stop-restart", action="store_true", help="Test continuous go/stop/restart with frozen wave checkpoints")
    parser.add_argument("--reverse", action="store_true", help="Fit reverse waves with forward rehearsal, then assess both directions")
    parser.add_argument("--refine-reverse", action="store_true", help="Continue a completed reverse cohort with broader commands and stronger original-forward retention")
    parser.add_argument("--refinement-iterations", type=positive_int, default=1000000)
    parser.add_argument("--increase-amplitude", type=float, help="Resume passed waves, fit a slightly larger target, then assess locomotion")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-run", type=Path, help="Read passed source checkpoints from this run")
    resume.add_argument("--resume-latest", action="store_true", help="Select latest passed source run under output-dir for the requested mode")
    args = parser.parse_args()
    if (args.assess_locomotion or args.stop_restart or args.reverse or args.refine_reverse or args.increase_amplitude is not None) and not (args.resume_run or args.resume_latest):
        parser.error("Assessment or amplitude continuation requires --resume-latest or --resume-run")
    if sum((args.assess_locomotion, args.stop_restart, args.reverse, args.refine_reverse, args.increase_amplitude is not None)) > 1:
        parser.error("Choose one of locomotion assessment, stop/restart, reverse fitting, or amplitude continuation")
    if args.increase_amplitude is not None and not (0 < args.increase_amplitude <= .1):
        parser.error("Amplitude must be finite and in (0, 0.1]")
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
    source_stage = "reverse" if args.refine_reverse else "wave" if args.assess_locomotion or args.stop_restart or args.reverse or args.increase_amplitude is not None else "alternation"
    try:
        if args.resume_run:
            resume_root = args.resume_run.expanduser().resolve()
            initial_actors = refinement_checkpoints(resume_root, seeds) if args.refine_reverse else resume_checkpoints(resume_root, seeds, source_stage)
        elif args.resume_latest:
            resume_root, initial_actors = find_refinement_run(output_root, seeds) if args.refine_reverse else find_resume_run(output_root, seeds, source_stage)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if args.increase_amplitude is not None:
        for seed, checkpoint_path in initial_actors.items():
            source_manifest = json.loads((checkpoint_path.parent / "manifest.json").read_text())
            old_amplitude = source_manifest.get("amplitude", .04)
            if not old_amplitude < args.increase_amplitude <= old_amplitude * 1.3 + 1e-8:
                parser.error(f"Seed {seed}: increase amplitude from {old_amplitude} by at most 30% per stage")
    prefix = "reverse-" if args.reverse or args.refine_reverse else "stop-restart-" if args.stop_restart else "locomotion-" if args.assess_locomotion else "wave-" if initial_actors else "motor-"
    root = output_root / (prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    root.mkdir(parents=True, exist_ok=False)
    capture_run_record(root, repo, args)
    deadline = time.perf_counter() + args.minutes * 60
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"Run directory: {root}\nMotor curriculum; no RL; {args.workers} independent seeds", flush=True)
    if resume_root:
        qualifier = "completed" if args.refine_reverse else "passed"
        print(f"Using each seed's {qualifier} {source_stage} checkpoint from: {resume_root}", flush=True)
    jobs = []
    with ExitStack() as files:
        def launch(job, stage):
            remaining = max(.01, (deadline - time.perf_counter()) / 60.)
            if stage in ("locomotion", "baseline_locomotion", "stop_restart", "forward_retention", "reverse_locomotion"):
                checkpoint_path = (job["output"] / "wave/actor_best.npz"
                    if args.increase_amplitude is not None and stage == "locomotion" else initial_actors[job["seed"]])
                if stage in ("forward_retention", "reverse_locomotion"):
                    checkpoint_path = job["output"] / "reverse/actor_best.npz"
                script = "assess_stop_restart.py" if stage in ("stop_restart", "forward_retention", "reverse_locomotion") else "assess_locomotion.py"
                command = [sys.executable, "-u", str(repo / "scripts" / script),
                    "--seed", str(job["seed"]), "--initial-actor", str(checkpoint_path),
                    "--output-dir", str(job["output"] / stage)]
                if stage == "reverse_locomotion":
                    command.extend(["--direction", "reverse"])
            elif stage == "reverse" and args.refine_reverse:
                command = [sys.executable, "-u", str(repo / "scripts/refine_reverse.py"),
                    "--seed", str(job["seed"]), "--minutes", str(max(.01, remaining - 1)),
                    "--iterations", str(args.refinement_iterations),
                    "--initial-actor", str(initial_actors[job["seed"]]),
                    "--output-dir", str(job["output"] / "reverse")]
            elif stage in ("wave", "reverse"):
                command = [sys.executable, "-u", str(repo / "scripts/train_motor_alternation.py"),
                    "--stage", stage, "--seed", str(job["seed"]), "--minutes", str(max(.01, remaining - 1) if stage == "reverse" else remaining),
                    "--iterations", str(args.alternation_iterations),
                    "--initial-actor", str(initial_actors[job["seed"]]),
                    "--output-dir", str(job["output"] / stage)]
                if args.increase_amplitude is not None:
                    command.extend(["--amplitude", str(args.increase_amplitude)])
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
                launch(job, "baseline_locomotion" if args.increase_amplitude is not None else
                    "reverse" if args.reverse or args.refine_reverse else
                    "stop_restart" if args.stop_restart else
                    "locomotion" if args.assess_locomotion else "wave" if initial_actors else "bend")
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
                        elif (job["stage"] == "baseline_locomotion" and not stopped and now < deadline
                                and baseline_valid(job["stages"]["baseline_locomotion"])):
                            print(f"seed={job['seed']} baseline propulsion is real and stable; fitting amplitude={args.increase_amplitude}", flush=True)
                            launch(job, "wave")
                        elif (job["stage"] == "wave" and args.increase_amplitude is not None
                                and stage_passed(job["stages"]["wave"]) and not stopped):
                            launch(job, "locomotion")
                        elif (job["stage"] == "reverse" and result.get("status") == "completed"
                                and result.get("error") is None and process.returncode in (0, 2) and not stopped):
                            # Diagnose the saved candidate even when its imitation gate fails.
                            launch(job, "forward_retention")
                        elif job["stage"] == "forward_retention" and not stopped:
                            launch(job, "reverse_locomotion")
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
    final_stage = "reverse_locomotion" if args.reverse or args.refine_reverse else "stop_restart" if args.stop_restart else "locomotion" if args.assess_locomotion or args.increase_amplitude is not None else "wave" if initial_actors else "alternation"
    passed = all(stage_passed(j["stages"].get(final_stage, {})) for j in jobs)
    if args.reverse or args.refine_reverse:
        passed = passed and all(stage_passed(j["stages"].get(stage, {}))
            for j in jobs for stage in ("reverse", "forward_retention"))
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for worker in summary["workers"]:
        print(json.dumps(worker), flush=True)
    print(f"MOTOR_CURRICULUM_EXIT_CODE={0 if passed and not stopped else 2}", flush=True)
    return 0 if passed and not stopped else 2


if __name__ == "__main__":
    raise SystemExit(main())
