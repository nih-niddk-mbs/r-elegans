#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
worm_out="/home/carsonc/r-elegans-results/relay-range-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup /home/carsonc/miniconda3/envs/r-elegans/bin/python -u -m scripts.total_job_budget \
    --pipeline relay-range --continue-relay --total-minutes 180 \
    --source-run /home/carsonc/r-elegans-results/relay-range-20261116T194725Z-293645 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "E088: maximum3h TOTAL including validation, automatic audit and cleanup"
echo "Pinned E087 selected weights; fresh CMA, not exact optimizer resume; original fitting bounds unchanged"
echo "Same AIYL range/cue/recovery objective;4CPUworkers x4cores; float64; no clamps during fitting"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Audit: $worm_out-audit/summary.json"
echo "Log: $worm_out.log"
