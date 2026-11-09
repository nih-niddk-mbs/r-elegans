#!/bin/sh
# E040 diagnostic only: finite bounded screen, no training or injected input.
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
if [ "${1:-}" = --worker-process ]; then
    trap 'worm_status=$?; echo "POSTERIOR_DIAGNOSTIC_EXIT_CODE=$worm_status"' 0
    "$worm_python" -u -m scripts.assay_posterior_excitability \
        --worker "$2" --output-dir "$3/network-audit" --random-profiles 8 --seed 4000
    "$worm_python" -u -m scripts.assay_posterior_basins \
        --audit-dir "$3/network-audit" --output-dir "$3/initial-basins"
    exit 0
fi
worm_parent=${1:-/home/carsonc/r-elegans-results/autonomous-tail-amplitude-20261109T200140261630Z/seed-2000}
"$worm_python" -c 'import sys; from scripts.autonomous_warm_start import read_warm_start; r=read_warm_start(sys.argv[1]); print("Reviewed parent:",r["parent_path"],"bound factor:",r["maximum_factor"])' "$worm_parent"
worm_cpus=$("$worm_python" -c 'import os; from scripts.benchmark_cpu import physical_core_cpus; cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0))); assert known and len(cpus)>=4; print(",".join(map(str,cpus[:4])))')
worm_results=/home/carsonc/r-elegans-results
worm_run="$worm_results/posterior-excitability-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$worm_run"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' JAX_DEFAULT_MATMUL_PRECISION=highest
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
nohup taskset -c "$worm_cpus" sh "$worm_repo/scripts/launch_posterior_excitability_mbslnx2.sh" \
    --worker-process "$worm_parent" "$worm_run" > "$worm_run.log" 2>&1 < /dev/null &
echo "Diagnostic only; zero injected current; no model promotion"
echo "PID: $!; CPUs: $worm_cpus"
echo "Results: $worm_run"
echo "Log: $worm_run.log"
