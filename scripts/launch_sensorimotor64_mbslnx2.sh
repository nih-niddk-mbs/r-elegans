#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_DEFAULT_MATMUL_PRECISION=highest
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [ "${1:-}" = --worker-process ]; then
    trap 'worm_status=$?; echo "SENSORIMOTOR64_EXIT_CODE=$worm_status"' 0
    if [ "$5" = resume ]; then
        "$worm_python" -u -m scripts.train_sensorimotor --resume-run "$2" --precision float64 --training-margin 1.02 --training-cues nominal-low-dose --seed 6200 --output-dir "$3" --minutes "$4"
    else
        "$worm_python" -u -m scripts.train_sensorimotor --warm-start-run "$2" --precision-reference-assay /home/carsonc/r-elegans-results/cue-precision-20261112T004158Z-781583 --precision float64 --training-margin 1.02 --training-cues nominal-low-dose --seed 6200 --output-dir "$3" --minutes "$4"
    fi
    exit 0
fi
worm_minutes=${1:-60}
worm_run=${2:-/home/carsonc/r-elegans-results/sensorimotor-20261111T193810Z-773573}
worm_mode=${3:-warm}
case "$worm_mode" in warm|resume) ;; *) echo "Expected warm or resume" >&2; exit 1 ;; esac
"$worm_python" -c 'import math,sys; assert math.isfinite(float(sys.argv[1])) and float(sys.argv[1])>0' "$worm_minutes"
if [ "$worm_mode" = warm ]; then
    "$worm_python" -c 'import hashlib,sys; from pathlib import Path; assert hashlib.sha256((Path(sys.argv[1])/"network_best.npz").read_bytes()).hexdigest()=="6cd7b0fa95c545c354f83334fe32aaf0551b8dd4e3d4e7f98babc944849669bf"' "$worm_run"
fi
worm_cpus=$("$worm_python" -c 'import os; from scripts.benchmark_cpu import physical_core_cpus; cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0))); assert known and len(cpus)>=4; print(",".join(map(str,cpus[:4])))')
worm_out="/home/carsonc/r-elegans-results/sensorimotor64-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup taskset -c "$worm_cpus" sh "$worm_repo/scripts/launch_sensorimotor64_mbslnx2.sh" \
    --worker-process "$worm_run" "$worm_out" "$worm_minutes" "$worm_mode" > "$worm_out.log" 2>&1 < /dev/null &
echo "Float64 robust sensory-motor fitting; mode=$worm_mode; original validation target and coordinate bounds"
echo "Parent: $worm_run"
echo "Budget: $worm_minutes minutes including baseline/search, plus validation"
echo "PID: $!; CPUs: $worm_cpus"
echo "Results: $worm_out"
echo "Log: $worm_out.log"
