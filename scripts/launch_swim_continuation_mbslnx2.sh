#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
worm_out="/home/carsonc/r-elegans-results/swim-continuation-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.total_job_budget --total-minutes 240 --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "FOUR HOURS TOTAL: setup, baseline, search, full validation and cleanup inside one deadline"
echo "Verified E070 seed6802; four fresh tail-focused continuations, hard swimming timing guard"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Budget receipt: $worm_out/budget_receipt.json"
echo "Log: $worm_out.log"
