#!/bin/sh
# Starts fixed-weight solver validation; status/report never launch work.
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
"$worm_python" -m scripts.run_midpoint_validation --check
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 "$worm_python" -c 'import jax; assert jax.config.x64_enabled; assert len(jax.devices("gpu"))==2; print("Two GPUs ready; fixed-weight solver validation")'
export JAX_PLATFORMS=cpu
worm_out="/home/carsonc/r-elegans-results/midpoint-validation-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup timeout --signal=TERM --kill-after=5s 14395s "$worm_python" -u -m scripts.run_midpoint_validation --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "E126 LAUNCH requested; duplicate check/shared lock required"
echo "Two A100s; opt-in midpoint benchmark then gated all-target clean-parent validation"
echo "Four cycles x four phases; no fitting or weight changes; original solver/gates preserved"
echo "Maximum240min TOTAL including preparation, validation, audit and cleanup; may finish earlier"
echo "PID: $!"
echo "Log (confirms launch or refusal): $worm_out.log"
echo "READ-ONLY status: $worm_python -m scripts.run_midpoint_validation --status"
echo "READ-ONLY report: $worm_python -m scripts.run_midpoint_validation --output-dir $worm_out --report"
