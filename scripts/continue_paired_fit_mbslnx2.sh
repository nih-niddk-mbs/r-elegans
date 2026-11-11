#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
worm_minutes=${1:-180}
worm_resume=$("$worm_python" -c 'from scripts.paired_resume import latest_completed; print(latest_completed("/home/carsonc/r-elegans-results"))')
echo "Continuing completed search: $worm_resume"
sh scripts/launch_paired_fit_mbslnx2.sh "$worm_minutes" \
    /home/carsonc/r-elegans-results/autonomous-posterior-motor-20261110T140125786528Z/seed-7001 "$worm_resume"
