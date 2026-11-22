#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 "$worm_python" -c 'import jax,scipy; assert jax.config.x64_enabled; assert len(jax.devices("gpu"))>=2; print("Dedicated r-elegans environment ready; two A100s, float64")'
export JAX_PLATFORMS=cpu
worm_out="/home/carsonc/r-elegans-results/stable-atlas-fit-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.run_stable_atlas_fit --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "E113: bounded supervised neuron/connection fit; fixed stereotypical properties and RMD channels"
echo "Maximum 480 minutes TOTAL: up to 330 optimization, remaining time for fresh-state validation/audit"
echo "Two persistent GPU workers; compact responses, all proposed weights/optimizer states retained"
echo "PID: $!; safe to disconnect"
echo "Log: $worm_out.log"
echo "Report: $worm_python -m scripts.run_stable_atlas_fit --output-dir $worm_out --report"
