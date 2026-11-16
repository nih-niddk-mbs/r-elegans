#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
worm_out="/home/carsonc/r-elegans-results/smd-drive-audit-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export MPLCONFIGDIR="$worm_out/mplconfig"
nohup timeout 20m /home/carsonc/miniconda3/envs/r-elegans/bin/python -u \
    -m scripts.audit_smd_drive \
    --run /home/carsonc/r-elegans-results/smd-control-20261115T191006Z-193279 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "Read-only SMD input/release/residual-command audit; maximum20min; no new simulation/training"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Figure: $worm_out/signal_chain.png"
echo "Log: $worm_out.log"
