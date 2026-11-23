#!/bin/sh
# Read-only. This script never launches, stops, or resumes a simulation.
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
exec /home/carsonc/miniconda3/envs/r-elegans/bin/python -m scripts.cv_validation_job --status
