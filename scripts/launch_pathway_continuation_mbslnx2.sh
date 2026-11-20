#!/bin/sh
set -eu
minutes=${1:-180}
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
"$worm_python" -c 'import jax,cma; from importlib.metadata import version; assert version("cma")=="4.4.1"; assert jax.config.x64_enabled; print("Dedicated interpreter ready; CPU float64")'
worm_out="/home/carsonc/r-elegans-results/pathway-continuation-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.total_job_budget \
    --pipeline pathway-continuation --total-minutes "$minutes" \
    --source-run /home/carsonc/r-elegans-results/pathway-training-20261119T143748Z-561835 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "E099: maximum $minutes minutes TOTAL including training, validation, saved-data audit and cleanup"
echo "Continuation from the audited E098 checkpoint; same 53 pathway groups (edges up to 4x above the checkpoint), weaker-side RIA node term"
echo "21 CPU workers x4 physical cores; population 21; gait preservation anchored to the E092 parent"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Audit: $worm_out-audit/summary.json"
echo "Log: $worm_out.log"
