#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_DEFAULT_MATMUL_PRECISION=highest
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [ "${1:-}" = --worker-process ]; then
    trap 'worm_status=$?; echo "HYBRID_VALIDATION_EXIT_CODE=$worm_status"' 0
    "$worm_python" -u -m scripts.run_rhythm_comparison --protocol hybrid \
        --comparison-run /home/carsonc/r-elegans-results/rhythm-comparison-20261112T185742Z-865067 \
        --expected-new 3fabef504731bc5cc904713752e6b3023c015e650b2e5318cc34f32cf7691a3b \
        --duration 60 --long-duration 300 --workers 4 --cores-per-worker 4 --output-dir "$2"
    exit 0
fi
worm_out="/home/carsonc/r-elegans-results/hybrid-validation-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup sh "$worm_repo/scripts/launch_hybrid_validation_mbslnx2.sh" \
    --worker-process "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "Candidate validation:7 simulations of60s plus one of300s; four concurrent workers"
echo "No fitting or periodic drive; source candidate hash verified and printed in log"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Log: $worm_out.log"
