#!/bin/sh
# Eight-hour plan; full validation and audit finish without a cutoff.
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
"$worm_python" -m scripts.run_full_cycle_validation --check
worm_out=$("$worm_python" -m scripts.run_full_cycle_validation --select-output)
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 "$worm_python" -c 'import jax; assert jax.config.x64_enabled; assert len(jax.devices("gpu"))==2; print("Two GPUs ready; float64 full-target peptide validation")'
export JAX_PLATFORMS=cpu
nohup "$worm_python" -u -m scripts.run_full_cycle_validation --output-dir "$worm_out" --minutes 480 "$@" >> "$worm_out.log" 2>&1 < /dev/null &
echo "E128 requested; active-job checks and shared exclusive GPU lock required"
echo "Eight-hour planning allowance; full validation and audit finish, even if longer"
echo "Peptide only; all 135 targets, 32 cycles, two clocks; saved panel/states reused; no fitting"
echo "Interrupted work resumes saved batches; all weights frozen; no automatic promotion"
echo "PID: $!"
echo "Log (confirms launch/resume or refusal): $worm_out.log"
echo "READ-ONLY status: $worm_python -m scripts.run_full_cycle_validation --status"
echo "READ-ONLY report: $worm_python -m scripts.run_full_cycle_validation --output-dir $worm_out --report"
