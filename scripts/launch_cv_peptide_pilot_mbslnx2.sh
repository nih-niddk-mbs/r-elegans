#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4
export JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 "$worm_python" -c 'import jax,h5py,scipy; assert jax.config.x64_enabled; assert len(jax.devices("gpu"))==2; print("Dedicated environment ready; two GPUs, float64")'
export JAX_PLATFORMS=cpu
worm_out="/home/carsonc/r-elegans-results/cv-peptide-pilot-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup timeout --signal=TERM --kill-after=5s 3595s "$worm_python" -u -m scripts.run_cv_peptide_pilot --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "E122: one animal-held-out fold, matched network-only/active peptide arms, one per A100"
echo "Maximum60 minutes TOTAL including numerical checks, audit and cleanup; incomplete work reported"
echo "Four training targets; bounded channel/synapse group search; peptide gains/timescale fixed priors"
echo "PID: $!"
echo "Log: $worm_out.log"
echo "Report: $worm_python -m scripts.run_cv_peptide_pilot --output-dir $worm_out --report"
