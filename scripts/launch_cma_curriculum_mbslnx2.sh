#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
worm_minutes=${1:-240}
worm_out="/home/carsonc/r-elegans-results/cma-cadence-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup /home/carsonc/miniconda3/envs/r-elegans/bin/python -u -m scripts.total_job_budget \
    --pipeline cma-curriculum --total-minutes "$worm_minutes" \
    --source-run /home/carsonc/r-elegans-results/controlled-swimming-20261114T034629Z-30860 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "CMA-ES sensory cadence curriculum:5/10/20/35/50% slowing with retained bends and START recovery"
echo "Maximum $worm_minutes minutes TOTAL, including validation/cleanup; no swimming promotion"
echo "Four persistent CPU workers; pinned cma installed in r-elegans env if missing"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Log: $worm_out.log"
