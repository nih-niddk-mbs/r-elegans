#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 "$worm_python" -c 'import jax,scipy; assert jax.config.x64_enabled; assert len(jax.devices("gpu"))>=2; print("Dedicated environment ready; two GPUs, float64")'
export JAX_PLATFORMS=cpu
worm_out="/home/carsonc/r-elegans-results/atlas-shared-state-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.run_atlas_shared_state \
  --source-run /home/carsonc/r-elegans-results/common-state-20261122T010707Z-782722 \
  --output-dir "$worm_out" --minutes 120 > "$worm_out.log" 2>&1 < /dev/null &
echo "E109: all cached atlas targets and observed downstream pairs; exact E108 states, no fitting"
echo "Two A100s, batches8, four reference phases and four neural grids; self responses scored separately"
echo "Maximum120 minutes TOTAL including automatic audit; may finish earlier"
echo "Response means/control differences retained; no large target-wise neural trajectory archive"
echo "PID: $!"
echo "Log: $worm_out.log"
echo "Report: $worm_python -m scripts.run_atlas_shared_state --output-dir $worm_out --report"
