#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
worm_out="/home/carsonc/r-elegans-results/control-signal-audit-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup timeout 20m taskset -c 0,1,2,3 /home/carsonc/miniconda3/envs/r-elegans/bin/python -u \
    -m scripts.audit_control_signal \
    --run /home/carsonc/r-elegans-results/controlled-swimming-20261114T034629Z-30860 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "Read-only saved-trajectory audit; no simulation, training or parameter changes. Maximum20min."
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Log: $worm_out.log"
