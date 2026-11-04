"""Compare one CPU PPO worker with concurrent workers on disjoint cores.

Each worker compiles and warms up before a shared start signal. Aggregate
throughput uses the span covering all workers' measured updates, excluding
compilation and warmup. This benchmarks the legacy 14-neuron actor only.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

if __package__:
    from .benchmark_cpu import physical_core_cpus, positive_int
else:
    from benchmark_cpu import physical_core_cpus, positive_int


def worker_cpu_groups(cpus: list[int], workers: int, cores: int) -> list[list[int]]:
    if len(cpus) != len(set(cpus)):
        raise ValueError("CPU candidates must be unique")
    if workers <= 0 or cores <= 0 or workers * cores > len(cpus):
        raise ValueError("Insufficient cores for disjoint worker allocations")
    return [cpus[index * cores:(index + 1) * cores] for index in range(workers)]


def aggregate_reports(reports: list[dict]) -> dict:
    if not reports:
        raise ValueError("No worker reports")
    for report in reports:
        if report["backend"] != "cpu" or report["finite_state"] is not True:
            raise ValueError("A worker failed CPU/finite-state validation")
        start = report["measurement_start_monotonic_seconds"]
        end = report["measurement_end_monotonic_seconds"]
        steps = report["measured_environment_steps"]
        if not all(math.isfinite(value) for value in (start, end, steps)):
            raise ValueError("Nonfinite measurement metadata")
        if end <= start or steps <= 0:
            raise ValueError("Invalid measurement interval or step count")
    start = min(r["measurement_start_monotonic_seconds"] for r in reports)
    end = max(r["measurement_end_monotonic_seconds"] for r in reports)
    steps = sum(r["measured_environment_steps"] for r in reports)
    return {
        "measurement_span_seconds": end - start,
        "total_measured_environment_steps": steps,
        "aggregate_environment_steps_per_second": steps / (end - start),
        "worker_environment_steps_per_second": [
            r["environment_steps_per_second"] for r in reports
        ],
    }


def run_cohort(args, groups: list[list[int]], run_dir: Path, repo: Path) -> dict:
    cohort_dir = run_dir / f"workers-{len(groups)}"
    cohort_dir.mkdir()
    start_file = cohort_dir / "start"
    processes = []
    report_paths = []
    ready_paths = []
    deadline = time.perf_counter() + args.timeout_seconds
    with ExitStack() as files:
        try:
            for index, cpus in enumerate(groups):
                ready = cohort_dir / f"worker-{index}.ready"
                report = cohort_dir / f"worker-{index}.json"
                log = cohort_dir / f"worker-{index}.log"
                child_env = os.environ.copy()
                child_env.update(
                    JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
                    JAX_DEFAULT_MATMUL_PRECISION="highest",
                    OMP_NUM_THREADS=str(len(cpus)), OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                )
                command = [
                    sys.executable, "-u", str(repo / "scripts/benchmark_rl.py"),
                    "--actor", "connectome", "--num-envs", str(args.num_envs),
                    "--num-steps", str(args.num_steps), "--updates", str(args.updates),
                    "--seed", str(index), "--ready-file", str(ready),
                    "--start-file", str(start_file),
                    "--start-timeout-seconds", str(args.timeout_seconds),
                ]
                print(f"Starting worker {index} on CPUs {cpus}; log={log}", flush=True)
                processes.append(subprocess.Popen(
                    command, cwd=repo, env=child_env,
                    stdout=files.enter_context(report.open("w")),
                    stderr=files.enter_context(log.open("w")),
                    # The supervisor imports no numerical libraries or threads.
                    preexec_fn=lambda selected=cpus: os.sched_setaffinity(0, selected),
                ))
                report_paths.append(report)
                ready_paths.append(ready)

            while not all(path.exists() for path in ready_paths):
                if any(process.poll() is not None for process in processes):
                    raise RuntimeError("Worker exited before the start signal; inspect its log")
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Workers did not finish preparation before the timeout")
                time.sleep(0.1)
            print("All workers ready; starting synchronized measurements", flush=True)
            start_file.write_text("start\n")
            for index, process in enumerate(processes):
                code = process.wait(timeout=max(0.01, deadline - time.perf_counter()))
                if code:
                    raise RuntimeError(f"Worker {index} failed with exit code {code}")
            reports = [json.loads(path.read_text()) for path in report_paths]
            for expected, report in zip(groups, reports):
                if sorted(report["cpu_affinity"]) != sorted(expected):
                    raise ValueError("A worker's reported affinity differs from its allocation")
            return {
                "status": "passed", "workers": len(groups),
                "cpu_groups": groups, "worker_reports": [str(p) for p in report_paths],
                **aggregate_reports(reports),
            }
        finally:
            # Only clean up processes created by this supervisor, including on
            # preparation failures, timeouts or interruption.
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--cores-per-worker", type=positive_int, default=4)
    parser.add_argument("--num-envs", type=positive_int, default=128)
    parser.add_argument("--num-steps", type=positive_int, default=128)
    parser.add_argument("--updates", type=positive_int, default=20)
    parser.add_argument("--timeout-seconds", type=positive_int, default=600)
    args = parser.parse_args()
    if not sys.platform.startswith("linux") or not hasattr(os, "sched_getaffinity"):
        parser.error("Run this benchmark on a Linux workstation such as mbslnx2")
    if args.workers < 2 or args.num_envs % 4:
        parser.error("At least two workers and a batch size divisible by four are required")
    allowed = sorted(os.sched_getaffinity(0))
    candidates, known = physical_core_cpus(allowed)
    if not known:
        parser.error("Physical CPU topology is required to avoid sharing sibling cores")
    try:
        groups = worker_cpu_groups(candidates, args.workers, args.cores_per_worker)
    except ValueError as error:
        parser.error(str(error))
    repo = Path(__file__).resolve().parents[1]
    output_root = args.output_dir.expanduser().resolve()
    if output_root == repo or repo in output_root.parents:
        parser.error("--output-dir must be outside the repository")
    run_dir = output_root / datetime.now(timezone.utc).strftime("cpu-parallel-%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "hostname": platform.node(), "configuration": vars(args) | {"output_dir": str(output_root)},
        "allowed_logical_cpus": allowed, "physical_core_candidates": candidates,
        "cohorts": [],
    }
    print(f"Reports: {run_dir}", flush=True)
    exit_code = 0
    for cohort_groups in ([groups[0]], groups):
        print(f"Benchmarking {len(cohort_groups)} concurrent worker(s)", flush=True)
        try:
            result = run_cohort(args, cohort_groups, run_dir, repo)
            print(
                f"workers={len(cohort_groups)} aggregate_steps_per_second="
                f"{result['aggregate_environment_steps_per_second']:.0f}", flush=True,
            )
        except (OSError, ValueError, KeyError, RuntimeError,
                subprocess.SubprocessError) as error:
            result = {"status": "failed", "workers": len(cohort_groups), "error": str(error)}
            print(f"Cohort failed: {error}", flush=True)
            exit_code = 1
        summary["cohorts"].append(result)
        if len(summary["cohorts"]) == 2 and not exit_code:
            baseline, concurrent = summary["cohorts"]
            speedup = concurrent["aggregate_environment_steps_per_second"] / baseline["aggregate_environment_steps_per_second"]
            summary.update(aggregate_speedup=speedup, parallel_efficiency=speedup / args.workers)
            print(f"Aggregate speedup={speedup:.2f}x; efficiency={speedup / args.workers:.1%}", flush=True)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if exit_code:
            break
    print(f"PARALLEL_BENCHMARK_EXIT_CODE={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
