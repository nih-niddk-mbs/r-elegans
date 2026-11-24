#!/bin/sh
# Continue saved validation, allowing it to finish; no training/total timeout.
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
worm_source=${1:-/home/carsonc/r-elegans-results/midpoint-validation-20261124T165552Z-1120675}
"$worm_python" -m scripts.continue_midpoint_validation --check
worm_out=$("$worm_python" -m scripts.continue_midpoint_validation --source-run "$worm_source" --select-output)
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 "$worm_python" -c 'import jax; assert jax.config.x64_enabled; assert len(jax.devices("gpu"))==2; print("Two GPUs ready; remaining batches divided between them")'
export JAX_PLATFORMS=cpu
nohup "$worm_python" -u -m scripts.continue_midpoint_validation --source-run "$worm_source" --output-dir "$worm_out" >> "$worm_out.log" 2>&1 < /dev/null &
echo "CONTINUATION requested; source preserved: $worm_source"
echo "Reuses completed levels/benchmark; resumes verified saved batches if already started"
echo "Validation allowed to finish; no training and no validation wall-clock cutoff"
echo "Two GPUs share remaining batches; each batch saved atomically and verified"
echo "An already completed continuation starts no new work; duplicate active jobs refused"
echo "PID: $!"
echo "Log (confirms launch/resume or refusal): $worm_out.log"
echo "READ-ONLY status: $worm_python -m scripts.continue_midpoint_validation --status"
echo "READ-ONLY report: $worm_python -m scripts.continue_midpoint_validation --output-dir $worm_out --report"
