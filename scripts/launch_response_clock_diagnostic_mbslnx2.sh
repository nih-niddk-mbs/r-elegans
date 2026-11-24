#!/bin/sh
# Starts a fixed-weight forward diagnostic. --status/--report never start it.
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
"$worm_python" -m scripts.run_response_clock_diagnostic --check
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 "$worm_python" -c 'import jax; assert jax.config.x64_enabled; assert len(jax.devices("gpu"))==2; print("Two GPUs ready; fixed-weight diagnostic")'
export JAX_PLATFORMS=cpu
worm_out="/home/carsonc/r-elegans-results/response-clock-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup timeout --signal=TERM --kill-after=5s 7195s "$worm_python" -u -m scripts.run_response_clock_diagnostic --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "E125 LAUNCH requested; duplicate check and shared lock required"
echo "Two fixed models, two A100s; three stimuli plus shams; no fitting or weight changes"
echo "Maximum120min TOTAL including preparation, forward checks, audit and cleanup; may finish earlier"
echo "PID: $!"
echo "Log (confirms launch or refusal): $worm_out.log"
echo "READ-ONLY status: $worm_python -m scripts.run_response_clock_diagnostic --status"
echo "READ-ONLY report: $worm_python -m scripts.run_response_clock_diagnostic --output-dir $worm_out --report"
