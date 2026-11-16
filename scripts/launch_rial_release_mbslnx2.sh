#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
worm_out="/home/carsonc/r-elegans-results/rial-release-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup /home/carsonc/miniconda3/envs/r-elegans/bin/python -u -m scripts.total_job_budget \
    --pipeline rial-release --total-minutes 180 \
    --source-run /home/carsonc/r-elegans-results/rial-sensory-20261116T041450Z-211013 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "Maximum3h TOTAL including final validation and automatic full saved-data audit"
echo "Sensory-cued RIAL release-variation reduction1/2/5%; uncued gait and recovery preserved"
echo "Four CPU workers x4cores; no imposed neural drive/clamp in fitting; all models/RNG retained"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Audit: $worm_out-audit/summary.json"
echo "Log: $worm_out.log"
