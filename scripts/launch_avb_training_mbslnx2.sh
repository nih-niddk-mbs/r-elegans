#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
worm_minutes=${1:-480}
"$worm_python" -c 'import jax,numpy,scipy; print("Interpreter ready; JAX",jax.__version__)'
mkdir -p /home/carsonc/r-elegans-results
worm_out="/home/carsonc/r-elegans-results/avb-training-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.total_job_budget \
    --pipeline avb-training --total-minutes "$worm_minutes" \
    --source-run /home/carsonc/r-elegans-results/controlled-swimming-20261114T034629Z-30860 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "AVB-dependent motor-circuit fitting; maximum $worm_minutes minutes TOTAL including validation/cleanup"
echo "4 CPU workers x 4 physical cores; float64; fixed water physics; no swimming promotion"
echo "Candidate metrics and all weights retained; full trajectories only for parent/final validation"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Log: $worm_out.log"
