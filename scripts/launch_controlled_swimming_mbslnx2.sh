#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
worm_out="/home/carsonc/r-elegans-results/controlled-swimming-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.total_job_budget --pipeline state-control --total-minutes 240 \
    --source-run /home/carsonc/r-elegans-results/swim-continuation-20261113T150202Z-1002622 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "Four hours TOTAL: sensory stop/restart first; swimming blocked until full control validation passes"
echo "Four CPU search workers; selected-checkpoint validation cases run in parallel"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Log: $worm_out.log"
