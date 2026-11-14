#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
worm_out="/home/carsonc/r-elegans-results/control-access-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup /home/carsonc/miniconda3/envs/r-elegans/bin/python -u -m scripts.total_job_budget \
    --pipeline control-access --total-minutes 45 \
    --source-run /home/carsonc/r-elegans-results/controlled-swimming-20261114T034629Z-30860 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "Diagnostic interneuron probes only; no training or learned-stop qualification"
echo "Four CPU workers;45min maximum TOTAL including refinement/cleanup"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Log: $worm_out.log"
