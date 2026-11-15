#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
worm_out="/home/carsonc/r-elegans-results/avb-audit-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup timeout 20m /home/carsonc/miniconda3/envs/r-elegans/bin/python -u \
    -m scripts.audit_avb_influence \
    --run /home/carsonc/r-elegans-results/avb-training-20261115T041708Z-122855 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "Read-only parent/selected AVB influence audit; saved data only; maximum20min"
echo "No JAX/GPU, training, parameter changes or new large trajectories"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Log: $worm_out.log"
