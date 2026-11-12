#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_DEFAULT_MATMUL_PRECISION=highest
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [ "${1:-}" = --worker-process ]; then
    trap 'worm_status=$?; echo "ALTERNATION64_EXIT_CODE=$worm_status"' 0
    "$worm_python" -u -m scripts.train_sensorimotor --warm-start-run "$2" --stage alternation --precision float64 \
        --training-cues nominal-low-dose --training-margin 1.02 --alternation-p2p-rad 0.00001 --seed 6300 --minutes "$4" --output-dir "$3"
    exit 0
fi
worm_minutes=${1:-60}
worm_run=/home/carsonc/r-elegans-results/sensorimotor64-20261112T020300Z-784477
"$worm_python" -c 'import hashlib,json,math,sys; from pathlib import Path; p=Path(sys.argv[1]); assert math.isfinite(float(sys.argv[2])) and float(sys.argv[2])>0; assert json.loads((p/"summary.json").read_text())["passed"] is True; assert hashlib.sha256((p/"network_best.npz").read_bytes()).hexdigest()=="16146cc539d47277cb2230e1cfced9a00282d9dd8e4e3b18112fde46515a042b"' "$worm_run" "$worm_minutes"
worm_cpus=$("$worm_python" -c 'import os; from scripts.benchmark_cpu import physical_core_cpus; cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0))); assert known and len(cpus)>=4; print(",".join(map(str,cpus[:4])))')
worm_out="/home/carsonc/r-elegans-results/alternation64-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup taskset -c "$worm_cpus" sh "$worm_repo/scripts/launch_alternation64_mbslnx2.sh" \
    --worker-process "$worm_run" "$worm_out" "$worm_minutes" > "$worm_out.log" 2>&1 < /dev/null &
echo "New independent alternation skill, float64; one cue2-3s, observe4-12s; no periodic pacing"
echo "Parent E062 preserved: $worm_run"
echo "Budget: $worm_minutes minutes including baseline/search, plus full validation"
echo "PID: $!; CPUs: $worm_cpus"
echo "Results: $worm_out"
echo "Log: $worm_out.log"
