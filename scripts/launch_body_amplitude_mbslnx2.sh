#!/bin/sh
# Fresh amplitude-only RNG forks, learned parameters and original bounds kept.
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
worm_results=/home/carsonc/r-elegans-results
worm_previous="$worm_results/autonomous-wave-20261109T032054849657Z"
worm_parent_a=${1:-"$worm_previous/seed-2"}
worm_parent_b=${2:-"$worm_previous/seed-3"}
cd "$worm_repo"
"$worm_python" -c 'import sys; from scripts.autonomous_warm_start import read_warm_start; rows=[read_warm_start(p) for p in sys.argv[1:]]; [print("Parent:", r["parent_path"], "at original bounds:", r["parameters_at_bound"], "/", len(r["state"]["best"])) for r in rows]' "$worm_parent_a" "$worm_parent_b"
mkdir -p "$worm_results"
worm_log="$worm_results/autonomous-body-amplitude-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
nohup "$worm_python" -u -m scripts.run_autonomous_wiggle \
    --objective body-amplitude --fit-scope neurons-connections \
    --warm-start-workers "$worm_parent_a" "$worm_parent_b" --seed-start 1000 \
    --output-dir "$worm_results" --minutes 240 \
    > "$worm_log" 2>&1 < /dev/null &
echo "Stage: body amplitude only; no phase, steering or propulsion objective"
echo "Supervisor PID: $!"
echo "Log: $worm_log"
