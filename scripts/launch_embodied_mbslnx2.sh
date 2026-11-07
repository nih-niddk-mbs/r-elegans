#!/bin/sh
# Launch from the checked-out repository; all inputs/results remain external.
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
worm_results=/home/carsonc/r-elegans-results
cd "$worm_repo"
"$worm_python" -c 'from scripts.bootstrap_embodied_inputs import find_successful_parent; print("Parent:", find_successful_parent("/home/carsonc/r-elegans-results"))'
mkdir -p "$worm_results"
worm_log="$worm_results/embodied-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
nohup "$worm_python" -u -m scripts.run_embodied_diagnostics \
    --output-dir "$worm_results" --minutes 45 \
    > "$worm_log" 2>&1 < /dev/null &
echo "Supervisor PID: $!"
echo "Log: $worm_log"
