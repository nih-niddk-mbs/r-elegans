#!/bin/sh
# E039: short, autonomous tail recruitment from reviewed learned parents.
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
worm_results=/home/carsonc/r-elegans-results
worm_previous="$worm_results/autonomous-body-amplitude-20261109T152102003705Z"
worm_parent_a=${1:-"$worm_previous/seed-1001"}
worm_parent_b=${2:-"$worm_previous/seed-1003"}
cd "$worm_repo"
"$worm_python" -c 'import sys; from scripts.autonomous_warm_start import read_warm_start; rows=[read_warm_start(p) for p in sys.argv[1:]]; [print("Parent:",r["parent_path"],"at original bounds:",r["parameters_at_bound"],"/",len(r["state"]["best"])) for r in rows]' "$worm_parent_a" "$worm_parent_b"
mkdir -p "$worm_results"
worm_log="$worm_results/autonomous-tail-amplitude-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
nohup "$worm_python" -u -m scripts.run_autonomous_wiggle \
    --objective body-amplitude --tail-focus --fit-scope neurons-connections \
    --warm-start-workers "$worm_parent_a" "$worm_parent_b" --seed-start 2000 \
    --output-dir "$worm_results" --minutes 20 --fit-seconds 20 --validation-seconds 60 \
    > "$worm_log" 2>&1 < /dev/null &
echo "Stage: tail amplitude; zero external drive;20-minute budget including validation"
echo "Supervisor PID: $!"
echo "Log: $worm_log"
