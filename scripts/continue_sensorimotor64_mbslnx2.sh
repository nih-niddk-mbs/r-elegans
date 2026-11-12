#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
worm_run=$("$worm_python" -c 'from scripts.sensorimotor_resume import latest_completed; print(latest_completed("/home/carsonc/r-elegans-results","sensorimotor64-*"))')
exec sh "$worm_repo/scripts/launch_sensorimotor64_mbslnx2.sh" "${1:-60}" "$worm_run" resume
