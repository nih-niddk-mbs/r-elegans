#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [ "${1:-}" = --worker-process ]; then
    trap 'worm_status=$?; echo "BODY_TIMING_EXIT_CODE=$worm_status"' 0
    "$worm_python" -u -m scripts.run_body_timing \
        --source-run /home/carsonc/r-elegans-results/hybrid-validation-20261112T202746Z-868392 \
        --output-dir "$2"
    exit 0
fi
worm_out="/home/carsonc/r-elegans-results/body-timing-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup sh "$worm_repo/scripts/launch_body_timing_mbslnx2.sh" --worker-process "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "Body timing diagnostic: baseline replay check, then four independent CPU workers; no training"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Log: $worm_out.log"
