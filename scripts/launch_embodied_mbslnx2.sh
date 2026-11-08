#!/bin/sh
# Launch the bundled Mac reference; generated results remain external.
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
worm_results=/home/carsonc/r-elegans-results
cd "$worm_repo"
"$worm_python" -c 'from scripts.run_embodied_diagnostics import select_input_bundle, verify_input_bundle; root=select_input_bundle(); verify_input_bundle(root); print("Reference:", root)'
mkdir -p "$worm_results"
worm_log="$worm_results/embodied-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
nohup "$worm_python" -u -m scripts.run_embodied_diagnostics \
    --output-dir "$worm_results" --minutes 45 "$@" \
    > "$worm_log" 2>&1 < /dev/null &
echo "Supervisor PID: $!"
echo "Log: $worm_log"
