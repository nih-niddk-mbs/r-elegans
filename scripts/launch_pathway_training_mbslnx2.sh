#!/bin/sh
set -eu
minutes=${1:-420}
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
"$worm_python" -c 'import jax,cma; from importlib.metadata import version; assert version("cma")=="4.4.1"; assert jax.config.x64_enabled; print("Dedicated interpreter ready; CPU float64")'
worm_out="/home/carsonc/r-elegans-results/pathway-training-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.total_job_budget \
    --pipeline pathway-training --total-minutes "$minutes" \
    --source-run /home/carsonc/r-elegans-results/relay-experiments-20261118T035453Z-398433 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "E098: maximum $minutes minutes TOTAL including training, validation, saved-data audit and cleanup"
echo "Joint cue-to-RIA-to-motor pathway training (53 groups, edges widened up to 4x), RIA node-level target; population 21"
echo "21 CPU workers x4 physical cores; all weights/RNG retained"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Audit: $worm_out-audit/summary.json"
echo "Log: $worm_out.log"
