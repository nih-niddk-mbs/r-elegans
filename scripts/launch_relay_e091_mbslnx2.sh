#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
worm_out="/home/carsonc/r-elegans-results/relay-range-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup /home/carsonc/miniconda3/envs/r-elegans/bin/python -u -m scripts.total_job_budget \
    --pipeline relay-range --continue-relay --optimization-minutes 180 --total-minutes 360 \
    --training-variant redundant \
    --source-run /home/carsonc/r-elegans-results/relay-range-20261117T184013Z-380160 \
    --parent-sha256 18eadad77c048a86aa305d59ec37defa3fa2e34ba2da8cf72139d9e6965c00cc \
    --experiment-id E091 --seed 9100 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "E091:3h optimization of first and close-repeat cue response; validation and automatic audit afterwards"
echo "Pinned E090 selected weights; fresh CMA/local recentering; original bounds and final acceptance criteria unchanged"
echo "Weakest cue response drives fitting;6h overall watchdog;4CPUworkers x4cores,float64"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Audit: $worm_out-audit/summary.json"
echo "Log: $worm_out.log"
