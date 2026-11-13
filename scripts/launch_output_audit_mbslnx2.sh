#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "${1:-}" = --worker-process ]; then
    trap 'worm_status=$?; echo "OUTPUT_AUDIT_EXIT_CODE=$worm_status"' 0
    "$worm_python" -u -m scripts.audit_output_path \
        --comparison-run /home/carsonc/r-elegans-results/rhythm-comparison-20261112T185742Z-865067 \
        --hybrid-run /home/carsonc/r-elegans-results/hybrid-validation-20261112T202746Z-868392 \
        --fit-run /home/carsonc/r-elegans-results/connection-amplitude-20261112T221938Z-871781 \
        --output-dir "$2"
    exit 0
fi
worm_out="/home/carsonc/r-elegans-results/output-path-audit-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup sh "$worm_repo/scripts/launch_output_audit_mbslnx2.sh" --worker-process "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "Read-only saved-trajectory audit; no simulations or weight changes"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Log: $worm_out.log"
