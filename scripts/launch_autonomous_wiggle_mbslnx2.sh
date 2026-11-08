#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
worm_results=/home/carsonc/r-elegans-results
cd "$worm_repo"
"$worm_python" -c 'from scripts.run_embodied_diagnostics import select_input_bundle,verify_input_bundle; verify_input_bundle(select_input_bundle())'
mkdir -p "$worm_results"
worm_log="$worm_results/autonomous-wiggle-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
nohup "$worm_python" -u -m scripts.run_autonomous_wiggle \
    --output-dir "$worm_results" --minutes 60 \
    > "$worm_log" 2>&1 < /dev/null &
echo "Supervisor PID: $!"
echo "Log: $worm_log"
