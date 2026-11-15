#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
worm_out="/home/carsonc/r-elegans-results/smd-control-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup /home/carsonc/miniconda3/envs/r-elegans/bin/python -u -m scripts.total_job_budget \
    --pipeline smd-control --total-minutes 240 \
    --source-run /home/carsonc/r-elegans-results/avb-training-20261115T041708Z-122855 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "SMD neural inhibition/release: maximum4h TOTAL including all confirmations/refinements"
echo "4 CPU workers x4 physical cores; fixed weights; no training or automatic promotion"
echo "Dose/individual/pair screen, shifted/repeated/long-hold recovery, neural/body-half checks"
echo "May finish earlier when planned trials complete; no existing data deleted"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Log: $worm_out.log"
