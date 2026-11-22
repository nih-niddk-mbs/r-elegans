#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 "$worm_python" -c 'import jax,scipy; assert jax.config.x64_enabled; assert len(jax.devices("gpu"))>=2; print("Dedicated environment ready; two GPUs, float64")'
export JAX_PLATFORMS=cpu
worm_out="/home/carsonc/r-elegans-results/atlas-target-continuation-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.run_atlas_target_refinement \
  --continue-refinement \
  --source-run /home/carsonc/r-elegans-results/atlas-target-refinement-20261122T025343Z-789212 \
  --output-dir "$worm_out" --minutes 120 > "$worm_out.log" 2>&1 < /dev/null &
echo "E111: same12 targets, both unchanged models, exact prior states and criteria"
echo "Replay0.0078125ms baseline, then0.00390625/0.001953125ms; no fitting"
echo "Two A100s under nohup; maximum120 minutes TOTAL including automatic audit"
echo "PID: $!"
echo "Log: $worm_out.log"
echo "Report: $worm_python -m scripts.run_atlas_target_refinement --output-dir $worm_out --report"
