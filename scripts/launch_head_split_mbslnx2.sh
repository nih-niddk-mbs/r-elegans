#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
worm_out="/home/carsonc/r-elegans-results/head-split-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup /home/carsonc/miniconda3/envs/r-elegans/bin/python -u -m scripts.total_job_budget \
    --pipeline head-split --total-minutes 45 \
    --source-run /home/carsonc/r-elegans-results/avb-training-20261115T041708Z-122855 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "RMD/SMD/SMB NMJ split; frozen checkpoint; maximum45min TOTAL including refinement"
echo "Six screens: sham, RMD, SMD, SMB, combined head, all-output control; four CPU workers"
echo "No training or checkpoint changes"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Log: $worm_out.log"
