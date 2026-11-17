#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
worm_out="/home/carsonc/r-elegans-results/relay-range-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup /home/carsonc/miniconda3/envs/r-elegans/bin/python -u -m scripts.total_job_budget \
    --pipeline relay-range --continue-relay --optimization-minutes 480 --total-minutes 660 \
    --source-run /home/carsonc/r-elegans-results/relay-range-20261116T222253Z-298495 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "E089:8h OPTIMIZATION after baseline; then up to2h validation and automatic audit"
echo "11h overall watchdog includes preparation/cleanup; curriculum success or disk/error limits may end earlier"
echo "Pinned latest E088 selected weights; fresh CMA, original bounds/gait criteria unchanged"
echo "4CPUworkers x4cores, float64; all weights/optimizer/RNG retained; intermediate traces in memory"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Audit: $worm_out-audit/summary.json"
echo "Log: $worm_out.log"
