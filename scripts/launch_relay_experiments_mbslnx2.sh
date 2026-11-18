#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
"$worm_python" -c 'import jax, cma; from importlib.metadata import version; assert version("cma")=="4.4.1"; assert jax.config.x64_enabled; print("Dedicated interpreter ready; CPU float64")'
worm_out="/home/carsonc/r-elegans-results/relay-experiments-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.total_job_budget \
    --pipeline relay-experiments --total-minutes 480 \
    --source-run /home/carsonc/r-elegans-results/relay-range-20261117T232652Z-388058 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "E092: maximum 8 hours TOTAL, including validation and automatic saved-data audit"
echo "Seven single-edge tests; two paired optimizer seeds per cap; 8 CPU workers x4 physical cores if available"
echo "Original prior versus explicit AIYR->AIYL-only extended prior; no automatic behavior qualification"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Audit: $worm_out-audit/summary.json"
echo "Log: $worm_out.log"
