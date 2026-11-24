#!/bin/sh
# Read-only; no fitting, simulation, resume or cancellation.
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
exec /home/carsonc/miniconda3/envs/r-elegans/bin/python -m scripts.run_cv_peptide_fit --status
