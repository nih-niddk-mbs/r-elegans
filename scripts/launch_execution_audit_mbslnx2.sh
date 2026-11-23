#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2
"$worm_python" -c 'import h5py,numpy,scipy; print("Audit environment ready; CPU only, no simulation or fitting")'
worm_out="/home/carsonc/r-elegans-results/execution-audit-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.run_execution_audit --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "E121: checkpoint/ancestry audit, raw animal/genotype/filter inventory, draft fold identities and implementation gaps"
echo "Original run audits/checkpoints unchanged; no fitting; CPU; maximum60 minutes"
echo "PID: $!"
echo "Log: $worm_out.log"
echo "Report: $worm_python -m scripts.run_execution_audit --output-dir $worm_out --report"
