#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4
/home/carsonc/miniconda3/envs/r-elegans/bin/python -u -m scripts.audit_atlas_composite \
  --coarse-run /home/carsonc/r-elegans-results/atlas-shared-state-20261122T014924Z-785140 \
  --fine-run /home/carsonc/r-elegans-results/atlas-target-continuation-20261122T032855Z-792463
