#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
# Check happens before GPU initialization and again under an exclusive lock.
"$worm_python" -m scripts.cv_validation_job --check
worm_source=${1:-/home/carsonc/r-elegans-results/cv-peptide-validation-20261123T220426Z-974332}
worm_minutes=${2:-240}
export JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4
export JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 "$worm_python" -c 'import jax; assert jax.config.x64_enabled; assert len(jax.devices("gpu"))==2; print("Two GPUs ready")'
export JAX_PLATFORMS=cpu
worm_out="/home/carsonc/r-elegans-results/cv-peptide-continuation-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.cv_validation_job --module scripts.continue_cv_peptide_validation --minutes "$worm_minutes" -- --output-dir "$worm_out" --source "$worm_source" --minutes "$worm_minutes" > "$worm_out.log" 2>&1 < /dev/null &
echo "CONTINUATION LAUNCH requested; source preserved: $worm_source"
echo "Completed audited stages reused; interrupted computation/JIT restarts. Both GPUs. No fitting."
echo "Maximum $worm_minutes minutes TOTAL including post-run audit and cleanup; may finish earlier."
echo "PID: $!"
echo "Log (confirms lock/launch or refusal): $worm_out.log"
echo "READ-ONLY report: $worm_python -m scripts.continue_cv_peptide_validation --output-dir $worm_out --report"
