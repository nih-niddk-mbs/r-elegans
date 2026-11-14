#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_log="/tmp/r-elegans-prune-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
nohup /home/carsonc/miniconda3/envs/r-elegans/bin/python -u -m scripts.prune_intermediate_trajectories \
    --base /home/carsonc/r-elegans-results --apply > "$worm_log" 2>&1 < /dev/null &
echo "Deletes excess intermediate trajectories from completed quiet fits; no /data or /proc required"
echo "Keeps all weights/checkpoints/RNG/optimizer, best/final/baseline and validation trajectories, and referenced input files"
echo "PID: $!"
echo "Log: $worm_log"
