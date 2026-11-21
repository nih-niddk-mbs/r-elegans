#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 "$worm_python" -c 'import jax,h5py,scipy; assert jax.config.x64_enabled; assert len(jax.devices("gpu"))>=2; print("Dedicated interpreter ready; two GPUs, float64")'
export JAX_PLATFORMS=cpu
worm_out="/home/carsonc/r-elegans-results/selected-atlas-audit-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.selected_atlas_audit_job \
    --e104-sweep /home/carsonc/r-elegans-results/allparams-20261121T023851Z-685266 \
    --e105-sweep /home/carsonc/r-elegans-results/peptides-20261121T164252Z-759054 \
    --data-root "${R_ELEGANS_DATA_DIR:-/home/carsonc/r-elegans-data}" \
    --minutes 120 --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "E106: read-only audit of all8 selected E104/E105 checkpoints; no fitting or reselection"
echo "One case per GPU; copied model/peptide pairs, independent data metrics, 40-vs-80 response tests, parent-relative rhythm"
echo "Maximum2hours TOTAL; may finish earlier. Missing/incomplete/failed checks never qualify."
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Log: $worm_out.log"
