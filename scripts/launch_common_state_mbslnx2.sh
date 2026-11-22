#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 "$worm_python" -c 'import jax,scipy; assert jax.config.x64_enabled; assert len(jax.devices("gpu"))>=2; print("Dedicated environment ready; two GPUs, float64")'
export JAX_PLATFORMS=cpu
worm_out="/home/carsonc/r-elegans-results/common-state-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.run_common_state \
  --source-run /home/carsonc/r-elegans-results/neural-convergence-20261121T232040Z-778363 \
  --output-dir "$worm_out" --minutes 120 > "$worm_out.log" 2>&1 < /dev/null &
echo "E108: unchanged models; common complete states, four reference phases, three pulse targets"
echo "Two A100 workers; no fitting. Duplicate shams and automatic post-run audit."
echo "Maximum120 minutes TOTAL including audit. May finish earlier; incomplete never qualifies."
echo "PID: $!"
echo "Log: $worm_out.log"
echo "Report: $worm_python -m scripts.run_common_state --output-dir $worm_out --report"
