#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 "$worm_python" -c 'import jax,scipy; assert jax.config.x64_enabled; assert len(jax.devices("gpu"))>=2; print("Dedicated environment ready; two GPUs, float64")'
export JAX_PLATFORMS=cpu
worm_out="/home/carsonc/r-elegans-results/atlas-target-refinement-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.run_atlas_target_refinement \
  --source-run /home/carsonc/r-elegans-results/atlas-shared-state-20261122T014924Z-785140 \
  --output-dir "$worm_out" --minutes 120 > "$worm_out.log" 2>&1 < /dev/null &
echo "E110: target union from ALL remaining E109 pair failures, both unchanged models and exact states"
echo "Same-step baseline reproduction plus two finer grids; no fitting or threshold changes"
echo "Two A100s, nohup, maximum120 minutes TOTAL including automatic audit"
echo "Exact target/pair selection prints in log before simulation; subset cannot qualify full atlas"
echo "PID: $!"
echo "Log: $worm_out.log"
echo "Report: $worm_python -m scripts.run_atlas_target_refinement --output-dir $worm_out --report"
