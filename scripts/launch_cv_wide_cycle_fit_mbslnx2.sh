#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
"$worm_python" -m scripts.run_cv_wide_cycle_fit --check
worm_out=$("$worm_python" -m scripts.run_cv_wide_cycle_fit --select-output)
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 "$worm_python" -c 'import jax; assert jax.config.x64_enabled; assert len(jax.devices("gpu"))==2; print("Two GPUs ready; float64 wider-panel CV fit")'
export JAX_PLATFORMS=cpu
nohup "$worm_python" -u -m scripts.run_cv_wide_cycle_fit --output-dir "$worm_out" --minutes 480 >> "$worm_out.log" 2>&1 < /dev/null &
echo "E130: benchmark4/8/16-target GPU batches, then eight hours of optimization"
echo "Up to16 training targets, five independent animal folds; same validated clean parent"
echo "Calibration/baseline and final validation/audit are additional time; no validation cutoff"
echo "Parameters, optimizer/RNG and detailed records kept; temporary arrays pruned after clean audit"
echo "PID: $!"
echo "Log (confirms launch or duplicate refusal): $worm_out.log"
echo "READ-ONLY status: $worm_python -m scripts.run_cv_wide_cycle_fit --status"
echo "READ-ONLY report: $worm_python -m scripts.run_cv_wide_cycle_fit --output-dir $worm_out --report"
