#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
worm_run=${1:-/home/carsonc/r-elegans-results/relay-experiments-20261118T035453Z-398433}
test -f "$worm_run/summary.json"
worm_audit="$worm_run-audit-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup /home/carsonc/miniconda3/envs/r-elegans/bin/python -u -m scripts.audit_relay_experiments \
    --run "$worm_run" --output-dir "$worm_audit" > "$worm_audit.log" 2>&1 < /dev/null &
echo "Read-only corrected audit; original results, audit and budget receipt preserved"
echo "No training or forward simulation"
echo "PID: $!"
echo "Audit: $worm_audit/summary.json"
echo "Log: $worm_audit.log"
