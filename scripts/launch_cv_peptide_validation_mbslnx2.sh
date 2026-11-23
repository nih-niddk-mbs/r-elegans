#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
"$worm_python" -m scripts.cv_validation_job --check
export JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4
export JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 "$worm_python" -c 'import jax; assert jax.config.x64_enabled; assert len(jax.devices("gpu"))==2; print("Two GPUs ready, dedicated environment, float64")'
export JAX_PLATFORMS=cpu
worm_out="/home/carsonc/r-elegans-results/cv-peptide-validation-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.cv_validation_job --module scripts.run_cv_peptide_validation --minutes 60 -- --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "E123: fixed E122 endpoints; no fitting. Three grids,300s dynamics, fresh-process continuation, batch4/8/16"
echo "Both A100s; maximum60min including post-run audit/cleanup. Missing work never qualifies."
echo "PID: $!"
echo "Log: $worm_out.log"
echo "Report: $worm_python -m scripts.run_cv_peptide_validation --output-dir $worm_out --report"
