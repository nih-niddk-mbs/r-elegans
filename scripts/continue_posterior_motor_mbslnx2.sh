#!/bin/sh
# E043: three-hour continuation of E041; three best-parent forks, one alternate.
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
worm_results=/home/carsonc/r-elegans-results
worm_previous="$worm_results/autonomous-posterior-motor-20261109T232143970960Z"
worm_parent_best=${1:-"$worm_previous/seed-6002"}
worm_parent_alternate=${2:-"$worm_previous/seed-6003"}
cd "$worm_repo"
"$worm_python" -c '
import sys
from scripts.autonomous_warm_start import read_warm_start
for path in sys.argv[1:]:
    row=read_warm_start(path)
    if row["summary"].get("posterior_motor_focus") is not True:
        raise SystemExit("Continuation requires a completed posterior-motor worker: "+path)
    print("Parent:",row["parent_path"],"best SHA256:",row["summary"]["network_best_sha256"],
          "original factor:",row["maximum_factor"],"parameters at bound:",row["parameters_at_bound"])
' "$worm_parent_best" "$worm_parent_alternate"
mkdir -p "$worm_results"
worm_log="$worm_results/autonomous-posterior-motor-continue-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
nohup "$worm_python" -u -m scripts.run_autonomous_wiggle \
    --objective body-amplitude --posterior-motor-focus --fit-scope neurons-connections \
    --warm-start-workers "$worm_parent_best" "$worm_parent_best" "$worm_parent_best" "$worm_parent_alternate" \
    --workers 4 --cores-per-worker 4 --seed-start 7000 \
    --output-dir "$worm_results" --minutes 180 --fit-seconds 20 --validation-seconds 300 \
    > "$worm_log" 2>&1 < /dev/null &
echo "Three-hour posterior motor continuation; zero external drive; original bounds retained"
echo "Seeds7000/7001/7002 from $worm_parent_best;seed7003 from $worm_parent_alternate"
echo "Budget:180 minutes including setup and validation;long validation300 simulated seconds"
echo "Supervisor PID: $!"
echo "Log: $worm_log"
