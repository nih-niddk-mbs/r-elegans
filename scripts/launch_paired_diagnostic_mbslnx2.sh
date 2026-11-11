#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
if [ "${1:-}" = --worker-process ]; then
    trap 'worm_status=$?; echo "PAIRED_DIAGNOSTIC_EXIT_CODE=$worm_status"' 0
    "$worm_python" -u -m scripts.diagnose_paired_motion --run "$2" --output-dir "$3" --duration "$4"
    exit 0
fi
worm_duration=${1:-40}
worm_run=$("$worm_python" -c 'from scripts.paired_resume import latest_completed; print(latest_completed("/home/carsonc/r-elegans-results"))')
worm_cpus=$("$worm_python" -c 'import os; from scripts.benchmark_cpu import physical_core_cpus; cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0))); assert known and len(cpus)>=4; print(",".join(map(str,cpus[:4])))')
worm_out="/home/carsonc/r-elegans-results/paired-diagnostic-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export JAX_PLATFORMS=cpu JAX_DEFAULT_MATMUL_PRECISION=highest
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
nohup taskset -c "$worm_cpus" sh "$worm_repo/scripts/launch_paired_diagnostic_mbslnx2.sh" \
    --worker-process "$worm_run" "$worm_out" "$worm_duration" > "$worm_out.log" 2>&1 < /dev/null &
echo "Read-only diagnostic; checkpoint unchanged: $worm_run"
echo "PID: $!; CPUs: $worm_cpus"
echo "Three cases of $worm_duration simulated seconds: active, refined, held voltage"
echo "Bounds: $worm_out/bounds.json"
echo "Report: $worm_out/summary.json"
echo "Motion: $worm_out/active/motion.html"
echo "Log: $worm_out.log"
