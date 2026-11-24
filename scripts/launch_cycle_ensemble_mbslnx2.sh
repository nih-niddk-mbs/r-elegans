#!/bin/sh
# Three hours of additional sampling; checkpointed stages and validation finish.
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
"$worm_python" -m scripts.run_cycle_ensemble --check
worm_out=$("$worm_python" -m scripts.run_cycle_ensemble --select-output)
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 "$worm_python" -c 'import jax; assert jax.config.x64_enabled; assert len(jax.devices("gpu"))==2; print("Two GPUs ready; nested cycle sampling")'
export JAX_PLATFORMS=cpu
nohup "$worm_python" -u -m scripts.run_cycle_ensemble --output-dir "$worm_out" "$@" >> "$worm_out.log" 2>&1 < /dev/null &
echo "E127 SAMPLING requested; shared exclusive lock and duplicate checks required"
echo "Three hours for additional sampling; admitted stages, validation and audit finish without a cutoff"
echo "Nested8/16/32cycles on affected targets; original four-cycle results reused; no fitting or weight changes"
echo "Each extension and response batch checkpointed atomically; launcher resumes existing unfinished work"
echo "PID: $!"
echo "Log (confirms launch/resume or refusal): $worm_out.log"
echo "READ-ONLY status: $worm_python -m scripts.run_cycle_ensemble --status"
echo "READ-ONLY report: $worm_python -m scripts.run_cycle_ensemble --output-dir $worm_out --report"
