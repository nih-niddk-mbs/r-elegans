"""E065 parallel controlled simulations; supervisor never imports JAX."""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from r_elegans.rhythm_comparison import comparison_cases
from scripts.benchmark_cpu import physical_core_cpus
from scripts.benchmark_cpu_parallel import worker_cpu_groups
from scripts.experiment_records import capture_run_record, file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-reference', type=Path, required=True)
    p.add_argument('--new-run', type=Path, required=True)
    p.add_argument('--expected-new', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--duration', type=float, default=60.)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--cores-per-worker', type=int, default=4)
    p.add_argument('--no-affinity', action='store_true', help='Explicit local smoke only; no CPU scaling claim')
    args = p.parse_args()
    if not math.isfinite(args.duration) or args.duration < 12 or args.workers < 1 or args.cores_per_worker < 1:
        p.error('Duration>=12s and positive worker/core counts required')
    repo = Path(__file__).resolve().parents[1]; out = args.output_dir.resolve()
    for source in (repo, args.old_reference.resolve(), args.new_run.resolve()):
        if out == source or source in out.parents or out in source.parents:
            p.error('Output must be external and separate from source inputs')
    if args.no_affinity:
        groups = [None] * args.workers
    else:
        if not hasattr(os, 'sched_getaffinity') or not shutil.which('taskset'):
            p.error('Linux taskset/affinity required; local verification must explicitly use --no-affinity')
        cpus, known = physical_core_cpus(sorted(os.sched_getaffinity(0)))
        if not known:
            p.error('Verified physical topology required')
        groups = worker_cpu_groups(cpus, args.workers, args.cores_per_worker)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'logs').mkdir()
    capture_run_record(out, repo, args)
    cases = comparison_cases()
    (out / 'planned_cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    env = os.environ.copy()
    env.update(JAX_PLATFORMS='cpu', JAX_DEFAULT_MATMUL_PRECISION='highest', JAX_ENABLE_X64='false',
               OMP_NUM_THREADS=str(args.cores_per_worker), OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    prep = [sys.executable, '-u', '-m', 'scripts.prepare_rhythm_comparison',
            '--old-reference', str(args.old_reference.resolve()), '--new-run', str(args.new_run.resolve()),
            '--expected-new', args.expected_new, '--output-dir', str(out / 'inputs')]
    subprocess.run(prep, cwd=repo, env=env, check=True)
    manifest_sha = file_sha256(out / 'inputs/manifest.json')
    pending = deque(cases); free = deque(range(args.workers)); running = {}; finished = []
    started = time.monotonic(); heartbeat = started
    try:
        while pending or running:
            while pending and free:
                case = pending.popleft(); slot = free.popleft()
                command = [sys.executable, '-u', '-m', 'scripts.compare_rhythm_worker',
                           '--inputs', str(out / 'inputs'), '--case', case['name'],
                           '--duration', str(args.duration), '--output-dir', str(out / case['name'])]
                if groups[slot] is not None:
                    command = ['taskset', '-c', ','.join(map(str, groups[slot])), *command]
                log = (out / 'logs' / (case['name'] + '.log')).open('w')
                try:
                    process = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close(); raise
                running[case['name']] = (process, log, slot, case)
                print(json.dumps(dict(starting=case['name'], pid=process.pid, cpus=groups[slot])), flush=True)
            for name, (process, log, slot, case) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close(); del running[name]; free.append(slot)
                path = out / name / 'summary.json'
                result = json.loads(path.read_text()) if path.is_file() else None
                finished.append(dict(case=case, exit_code=code, cpus=groups[slot], result=result))
                print(json.dumps(dict(finished=name, exit_code=code,
                                      late_repetitions=result['late_repeating_by_stage'] if result else None)), flush=True)
            if time.monotonic() - heartbeat >= 30:
                print(json.dumps(dict(elapsed_minutes=(time.monotonic()-started)/60,
                                      running=list(running), pending=len(pending))), flush=True)
                heartbeat = time.monotonic()
            time.sleep(.2)
    finally:
        for process, log, _, _ in running.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            log.close()
    fingerprints = {row['result']['base_initial_state_sha256'] for row in finished
                    if row['result'] and row['case']['precision'] == 'float64'}
    same_initial = len(fingerprints) == 1
    unchanged = file_sha256(out / 'inputs/manifest.json') == manifest_sha
    valid = (len(finished) == len(cases) and all(row['exit_code'] == 0 and row['result']
             and row['result']['execution_valid'] for row in finished) and same_initial and unchanged)
    summary = dict(experiment='E065', status='completed', execution_valid=valid,
                   duration_s=args.duration, workers=args.workers, cpu_groups=groups,
                   no_affinity=args.no_affinity, elapsed_seconds=time.monotonic()-started,
                   common_float64_initial_state=same_initial, input_manifest_unchanged=unchanged,
                   finished=finished, no_optimization=True, no_rng_draws=True,
                   scope='Controlled family substitutions and finite-window rhythm observations; no biological or locomotion pass')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    for row in sorted(finished, key=lambda r: r['case']['name']):
        result = row['result']
        print(json.dumps(dict(case=row['case']['name'], exit_code=row['exit_code'],
                              late_repetitions=result['late_repeating_by_stage'] if result else None,
                              repeating_motor_neurons={k: v['repeating_neurons'] for k, v in result['motor_named_groups'].items()}
                              if result else None,
                              early_reference_match=result['early_reference_match'] if result else None)), flush=True)
    print(json.dumps(dict(execution_valid=valid, common_float64_initial_state=same_initial,
                          report=str(out / 'summary.json'))), flush=True)
    return 0 if valid else 2


if __name__ == '__main__':
    raise SystemExit(main())
