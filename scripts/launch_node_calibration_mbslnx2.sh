#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
"$worm_python" -c 'import jax; assert jax.config.x64_enabled; print("Dedicated interpreter ready; CPU float64")'
worm_out="/home/carsonc/r-elegans-results/node-calibration-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.total_job_budget \
    --pipeline node-probe-calibration --total-minutes 60 \
    --source-run /home/carsonc/r-elegans-results/relay-experiments-20261118T035453Z-398433 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "E097: maximum 60 minutes TOTAL including parent check, probes, saved-data audit and cleanup"
echo "Low-dose (0.0003-0.01 nS) imposed conductance on AIZ, AIY, RIA, SMD, RIB, RIM; cue run; AIZ grid controls; no fitting"
echo "Up to 21 CPU workers x4 physical cores; every trace retained"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Audit: $worm_out-audit/summary.json"
echo "Log: $worm_out.log"
echo "Then: python -m scripts.report_node_probe --run $worm_out ; python -m scripts.analyze_node_calibration --run $worm_out"
