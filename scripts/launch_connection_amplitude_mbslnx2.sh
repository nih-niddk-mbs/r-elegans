#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_DEFAULT_MATMUL_PRECISION=highest
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [ "${1:-}" = --worker-process ]; then
    trap 'worm_status=$?; echo "CONNECTION_AMPLITUDE_EXIT_CODE=$worm_status"' 0
    "$worm_python" -u -m scripts.run_connection_amplitude \
        --source-run /home/carsonc/r-elegans-results/hybrid-validation-20261112T202746Z-868392 \
        --expected-parent f024bd14b75e07a6c9c48bcc2bc03aed4b8354510c1d46ee8235e7575638f56e \
        --minutes "$3" --workers 4 --cores-per-worker 4 --seed-start 6700 --output-dir "$2"
    exit 0
fi
worm_minutes=${1:-240}
"$worm_python" -c 'import math,sys; x=float(sys.argv[1]); assert math.isfinite(x) and x>0' "$worm_minutes"
worm_out="/home/carsonc/r-elegans-results/connection-amplitude-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup sh "$worm_repo/scripts/launch_connection_amplitude_mbslnx2.sh" \
    --worker-process "$worm_out" "$worm_minutes" > "$worm_out.log" 2>&1 < /dev/null &
echo "Four independent connection-only searches; all channel scales frozen; no periodic drive"
echo "Search budget: $worm_minutes minutes per worker including baseline; validation follows"
echo "Four concurrent workers on disjoint four-physical-core groups; seeds6700..6703"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Log: $worm_out.log"
