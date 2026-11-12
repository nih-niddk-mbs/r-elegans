#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [ "${1:-}" = --worker-process ]; then
    trap 'worm_status=$?; echo "ALTERNATION_DIAGNOSTIC_EXIT_CODE=$worm_status"' 0
    "$worm_python" -u -m scripts.analyze_alternation_signals --run "$2" --output-dir "$3" \
        --expected-sha256 3fabef504731bc5cc904713752e6b3023c015e650b2e5318cc34f32cf7691a3b
    exit 0
fi
worm_run=/home/carsonc/r-elegans-results/alternation64-20261112T042107Z-791518
worm_out="/home/carsonc/r-elegans-results/alternation-signals-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup sh "$worm_repo/scripts/launch_alternation_diagnostic_mbslnx2.sh" \
    --worker-process "$worm_run" "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "Read-only saved-trajectory analysis; no simulation, training or GPU initialization"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Figures: $worm_out/baseline/active/raw-signals.png and $worm_out/best/active/raw-signals.png"
echo "Log: $worm_out.log"
