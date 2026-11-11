#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_DEFAULT_MATMUL_PRECISION=highest
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [ "${1:-}" = --worker-process ]; then
    trap 'worm_status=$?; echo "SENSORIMOTOR_EXIT_CODE=$worm_status"' 0
    if [ "${5:-fresh}" = resume ]; then
        worm_cues=$("$worm_python" -c 'import json,sys; from pathlib import Path; print(json.loads((Path(sys.argv[1])/"configuration.json").read_text())["arguments"].get("training_cues","nominal"))' "$2")
        "$worm_python" -u -m scripts.train_sensorimotor --resume-run "$2" --training-cues "$worm_cues" --output-dir "$3" --minutes "$4"
    elif [ "${5:-fresh}" = robust ]; then
        "$worm_python" -u -m scripts.train_sensorimotor --warm-start-run "$2" --training-cues nominal-low-dose --seed 5700 --output-dir "$3" --minutes "$4"
    else
        "$worm_python" -u -m scripts.train_sensorimotor --parent-run "$2" --output-dir "$3" --minutes "$4"
    fi
    exit 0
fi
worm_minutes=${1:-20}
"$worm_python" -c 'import math,sys; n=float(sys.argv[1]); assert math.isfinite(n) and n>0, "Minutes must be positive"; import jax,numpy,scipy' "$worm_minutes"
worm_mode=fresh
if [ -n "${2:-}" ]; then
    worm_mode=${3:-resume}
    case "$worm_mode" in resume|robust) ;; *) echo "Unknown saved-search mode: $worm_mode" >&2; exit 1 ;; esac
    worm_run=$2
else
    worm_run=$("$worm_python" -c 'from scripts.paired_resume import latest_completed; print(latest_completed("/home/carsonc/r-elegans-results"))')
fi
worm_cpus=$("$worm_python" -c 'import os; from scripts.benchmark_cpu import physical_core_cpus; cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0))); assert known and len(cpus)>=4; print(",".join(map(str,cpus[:4])))')
worm_out="/home/carsonc/r-elegans-results/sensorimotor-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup taskset -c "$worm_cpus" sh "$worm_repo/scripts/launch_sensorimotor_mbslnx2.sh" \
    --worker-process "$worm_run" "$worm_out" "$worm_minutes" "$worm_mode" > "$worm_out.log" 2>&1 < /dev/null &
echo "Stage1: finite odor cue -> reversible bend; full302-neuron network; mode=$worm_mode"
echo "Parent: $worm_run"
echo "Budget: $worm_minutes minutes including baseline/search, plus subsequent validation"
echo "PID: $!; CPUs: $worm_cpus"
echo "Results: $worm_out"
echo "Report: $worm_out/summary.json"
echo "Motion: $worm_out/active/motion.html"
echo "Log: $worm_out.log"
