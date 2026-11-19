#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
"$worm_python" -c 'import jax; assert jax.config.x64_enabled; print("Dedicated interpreter ready; CPU float64")'
worm_out="/home/carsonc/r-elegans-results/node-probe-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.total_job_budget \
    --pipeline node-probe --total-minutes 30 \
    --source-run /home/carsonc/r-elegans-results/relay-experiments-20261118T035453Z-398433 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "E096: maximum 30 minutes TOTAL including parent check, probes, saved-data audit and cleanup"
echo "Imposed conductance dose-response (0.01-0.1 nS, excitatory and inhibitory) on 10 node groups; no fitting"
echo "Up to 21 CPU workers x4 physical cores; scan metrics-only, 0.1 nS cases keep raw traces"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Audit: $worm_out-audit/summary.json"
echo "Log: $worm_out.log"
