#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
worm_out="/home/carsonc/r-elegans-results/rial-control-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup /home/carsonc/miniconda3/envs/r-elegans/bin/python -u -m scripts.total_job_budget \
    --pipeline rial-control --total-minutes 60 \
    --source-run /home/carsonc/r-elegans-results/avb-training-20261115T041708Z-122855 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "RIAL causal test: membrane inhibition versus existing RIAL-to-SMD chemical release holds"
echo "Maximum60min TOTAL, four CPU workers, fixed weights, repeated/long-hold recovery and timestep checks"
echo "No training, new connections or learned-stop qualification"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Log: $worm_out.log"
