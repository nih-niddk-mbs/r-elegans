#!/bin/sh
# E044: read-only full-body assessment of the pinned qualified tail winner.
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
if [ "${1:-}" = --worker-process ]; then
    trap 'worm_status=$?; echo "FULL_BODY_ASSESSMENT_EXIT_CODE=$worm_status"' 0
    "$worm_python" -u -m scripts.assess_full_body --worker "$2" --output-dir "$3" --body-length .001
    exit 0
fi
worm_parent=${1:-/home/carsonc/r-elegans-results/autonomous-posterior-motor-20261110T140125786528Z/seed-7001}
"$worm_python" -c '
import importlib.util,sys
from scripts.autonomous_warm_start import read_warm_start
row=read_warm_start(sys.argv[1])
if row["summary"].get("passed") is not True or row["summary"].get("posterior_motor_focus") is not True:
    raise SystemExit("A completed qualified posterior-motor winner is required")
if importlib.util.find_spec("matplotlib") is None:
    raise SystemExit("The dedicated environment needs matplotlib for scientific figures")
print("Qualified winner:",row["parent_path"],"SHA256:",row["summary"]["network_best_sha256"])
' "$worm_parent"
worm_cpus=$("$worm_python" -c 'import os; from scripts.benchmark_cpu import physical_core_cpus; cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0))); assert known and len(cpus)>=4; print(",".join(map(str,cpus[:4])))')
worm_results=/home/carsonc/r-elegans-results
mkdir -p "$worm_results"
worm_out="$worm_results/full-body-assessment-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="$worm_results/.matplotlib-cache"
nohup taskset -c "$worm_cpus" sh "$worm_repo/scripts/assess_winner_mbslnx2.sh" \
    --worker-process "$worm_parent" "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "Read-only assessment;no training or model changes"
echo "PID: $!; CPUs: $worm_cpus"
echo "Results: $worm_out"
echo "Motion preview: $worm_out/motion.html"
echo "Long-window figure: $worm_out/long/trajectory.png"
echo "Log: $worm_out.log"
