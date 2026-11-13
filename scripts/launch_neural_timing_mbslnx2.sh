#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_DEFAULT_MATMUL_PRECISION=highest
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [ "${1:-}" = --worker-process ]; then
    trap 'worm_status=$?; echo "NEURAL_TIMING_EXIT_CODE=$worm_status"' 0
    "$worm_python" -u -m scripts.run_connection_amplitude \
        --source-run /home/carsonc/r-elegans-results/hybrid-validation-20261112T202746Z-868392 \
        --expected-parent f024bd14b75e07a6c9c48bcc2bc03aed4b8354510c1d46ee8235e7575638f56e \
        --timing-evidence-run /home/carsonc/r-elegans-results/body-timing-20261113T044423Z-940182 \
        --neural-timing --sigma .08 --minutes "$3" --workers 4 --cores-per-worker 4 \
        --seed-start 6800 --output-dir "$2"
    exit 0
fi
worm_minutes=${1:-480}
"$worm_python" -c 'import math,sys; x=float(sys.argv[1]); assert math.isfinite(x) and x>0' "$worm_minutes"
worm_out="/home/carsonc/r-elegans-results/neural-timing-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup sh "$worm_repo/scripts/launch_neural_timing_mbslnx2.sh" --worker-process "$worm_out" "$worm_minutes" > "$worm_out.log" 2>&1 < /dev/null &
echo "Four full-network density/connection searches; no external pacing or clock scaling"
echo "Baseline/search: $worm_minutes minutes per worker; full validation follows (extra time)"
echo "Swimming in water: body-rhythm target1.8 Hz; same physical clock; seeds6800..6803"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Log: $worm_out.log"
