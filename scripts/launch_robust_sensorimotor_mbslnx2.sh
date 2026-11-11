#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
worm_minutes=${1:-60}
worm_run=$("$worm_python" -c 'from scripts.sensorimotor_resume import latest_completed; print(latest_completed("/home/carsonc/r-elegans-results"))')
echo "New robust-cue objective; inherit neural/sensory parameters, fresh seed5700: $worm_run"
exec sh "$worm_repo/scripts/launch_sensorimotor_mbslnx2.sh" "$worm_minutes" "$worm_run" robust
