"""Run sequential CPU-affinity/batch-size benchmarks on a Linux workstation.

This driver imports no numerical libraries. It restricts each child process
before Python imports JAX, selects one logical CPU per physical core where
Linux topology is available, and writes reports outside the repository.
Run after other benchmarks finish for an uncontended comparison.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def physical_core_cpus(allowed: list[int]) -> tuple[list[int], bool]:
    """Choose one allowed sibling per (socket, core); never expand affinity."""

    representatives = {}
    for cpu in sorted(allowed):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            socket = int((topology / "physical_package_id").read_text().strip())
            core = int((topology / "core_id").read_text().strip())
            if socket < 0 or core < 0:
                raise ValueError("Unknown physical topology")
        except (OSError, ValueError):
            return sorted(allowed), False
        representatives.setdefault((socket, core), cpu)
    return list(representatives.values()), True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-counts", type=positive_int, nargs="+", default=[4, 16, 32])
    parser.add_argument("--batches", type=positive_int, nargs="+", default=[32, 128])
    parser.add_argument("--num-steps", type=positive_int, default=128)
    parser.add_argument("--updates", type=positive_int, default=5)
    parser.add_argument("--timeout-seconds", type=positive_int, default=600)
    args = parser.parse_args()
    if not sys.platform.startswith("linux") or not hasattr(os, "sched_getaffinity"):
        parser.error("This runner requires Linux CPU affinity (run on mbslnx2)")
    if any(batch % 4 for batch in args.batches):
        parser.error("Batch sizes must be divisible by four PPO minibatches")
    allowed = sorted(os.sched_getaffinity(0))
    candidates, topology_known = physical_core_cpus(allowed)
    repo = Path(__file__).resolve().parents[1]
    output_root = args.output_dir.expanduser().resolve()
    if output_root == repo or repo in output_root.parents:
        parser.error("--output-dir must be outside the repository")
    # A unique directory prevents overwriting earlier benchmark results.
    run_dir = output_root / datetime.now(timezone.utc).strftime("cpu-%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "hostname": platform.node(),
        "allowed_logical_cpus": allowed,
        "physical_topology_known": topology_known,
        "candidate_cpus": candidates,
        "selection": "one_thread_per_core" if topology_known else "logical_cpu_fallback",
        "results": [],
    }
    print(f"Reports: {run_dir}", flush=True)
    print(
        f"Allowed logical CPUs={len(allowed)}; selected candidates={len(candidates)}; "
        f"physical topology known={topology_known}",
        flush=True,
    )
    failures = 0
    for count in dict.fromkeys(args.cpu_counts):
        for batch in dict.fromkeys(args.batches):
            record = {"cpu_count": count, "num_envs": batch}
            if count > len(candidates):
                record.update(status="skipped", reason="insufficient allowed cores")
            else:
                cpus = candidates[:count]
                record["cpu_affinity"] = cpus
                label = f"cores-{count}-envs-{batch}"
                report_file = run_dir / f"{label}.json"
                log_file = run_dir / f"{label}.log"
                record.update(report=str(report_file), log=str(log_file))
                child_env = os.environ.copy()
                child_env.update(
                    JAX_PLATFORMS="cpu",
                    CUDA_VISIBLE_DEVICES="",
                    JAX_DEFAULT_MATMUL_PRECISION="highest",
                    OMP_NUM_THREADS=str(count),
                    OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    NUMEXPR_NUM_THREADS="1",
                )
                command = [
                    sys.executable, "-u", str(repo / "scripts/benchmark_rl.py"),
                    "--actor", "connectome", "--num-envs", str(batch),
                    "--num-steps", str(args.num_steps), "--updates", str(args.updates),
                ]
                print(f"Running {label} on CPUs {cpus}", flush=True)
                try:
                    with report_file.open("w") as report, log_file.open("w") as log:
                        # This supervisor is single-threaded and imports no JAX,
                        # so affinity is safely set before the child execs Python.
                        result = subprocess.run(
                            command, cwd=repo, env=child_env, stdout=report, stderr=log,
                            preexec_fn=lambda: os.sched_setaffinity(0, cpus),
                            timeout=args.timeout_seconds,
                        )
                    if result.returncode:
                        raise RuntimeError(f"benchmark exit code {result.returncode}")
                    data = json.loads(report_file.read_text())
                    if data["backend"] != "cpu" or data["finite_state"] is not True:
                        raise RuntimeError("benchmark did not validate finite CPU training")
                    record.update(
                        status="passed",
                        compile_seconds=data["compile_seconds"],
                        median_update_seconds=data["median_update_seconds"],
                        environment_steps_per_second=data["environment_steps_per_second"],
                    )
                    print(
                        f"{label}: {record['environment_steps_per_second']:.0f} "
                        "environment steps/s", flush=True,
                    )
                except (OSError, ValueError, KeyError, RuntimeError,
                        subprocess.SubprocessError) as error:
                    failures += 1
                    record.update(status="failed", error=str(error))
                    print(f"{label} failed: {error}; see {log_file}", flush=True)
            summary["results"].append(record)
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not any(row["status"] == "passed" for row in summary["results"]):
        failures += 1
    print(f"CPU_SWEEP_EXIT_CODE={int(bool(failures))}", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
