#!/bin/sh
# Starts fitting. For read-only status use status_cv_peptide_fit_mbslnx2.sh.
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
"$worm_python" -m scripts.run_cv_peptide_fit --check
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 "$worm_python" -c 'import jax,h5py,scipy; assert jax.config.x64_enabled; assert len(jax.devices("gpu"))==2; print("Two GPUs ready; dedicated r-elegans environment")'
export JAX_PLATFORMS=cpu
worm_out="/home/carsonc/r-elegans-results/cv-peptide-fit-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup timeout --signal=TERM --kill-after=5s 28795s "$worm_python" -u -m scripts.run_cv_peptide_fit --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "E124 FIT LAUNCH requested; duplicate process check and shared exclusive lock required"
echo "Five independent animal folds; network-only/peptide arms; two A100s; batch16"
echo "Maximum480min TOTAL, including preparation, fitting, validation, audit and cleanup"
echo "Optional probes stop early to preserve measured validation reserve; incomplete work never qualifies"
echo "PID: $!"
echo "Log (confirms launch or refusal): $worm_out.log"
echo "READ-ONLY status: sh scripts/status_cv_peptide_fit_mbslnx2.sh"
echo "READ-ONLY report: $worm_python -m scripts.run_cv_peptide_fit --output-dir $worm_out --report"
