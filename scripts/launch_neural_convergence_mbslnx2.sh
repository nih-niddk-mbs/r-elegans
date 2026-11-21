#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 "$worm_python" -c 'import jax,scipy; assert jax.config.x64_enabled; assert len(jax.devices("gpu"))>=2; print("Dedicated environment ready; two GPUs, float64")'
export JAX_PLATFORMS=cpu
worm_out="/home/carsonc/r-elegans-results/neural-convergence-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.run_neural_convergence \
  --source-audit /home/carsonc/r-elegans-results/selected-atlas-audit-20261121T221402Z-771144 \
  --output-dir "$worm_out" --minutes 120 > "$worm_out.log" 2>&1 < /dev/null &
echo "E107: unchanged selected network control and stored parent, one model per A100"
echo "Five refinement levels each; no fitting; automatic complete post-run audit"
echo "Maximum120 minutes TOTAL, including audit. May finish earlier; timeout never qualifies."
echo "Compact atlas responses/RMD traces retained; no full-brain trajectory archive"
echo "PID: $!"
echo "Log: $worm_out.log"
echo "Audit: $worm_out/audit.json"
echo "Report: $worm_python -m scripts.run_neural_convergence --output-dir $worm_out --report"
