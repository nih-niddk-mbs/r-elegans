#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
worm_out="/home/carsonc/r-elegans-results/relay-range-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup /home/carsonc/miniconda3/envs/r-elegans/bin/python -u -m scripts.total_job_budget \
    --pipeline relay-range --total-minutes 180 \
    --source-run /home/carsonc/r-elegans-results/rial-release-20261116T163801Z-287910 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "Maximum3h TOTAL including final validation and automatic full saved-data audit"
echo "AIYL release headroom2/5/10% plus cue responsiveness; uncued gait/recovery preserved"
echo "Four CPU workers x4cores; no imposed neural drive/clamp in fitting; all models/RNG retained"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Audit: $worm_out-audit/summary.json"
echo "Log: $worm_out.log"
