#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
worm_log="/tmp/r-elegans-storage-clean-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
nohup "$worm_python" -u -m scripts.clean_results_storage --apply "$@" > "$worm_log" 2>&1 < /dev/null &
echo "Cleanup PID: $!"
echo "Keeps parameters, optimizer/RNG, detailed records, active runs and recursive dependencies"
echo "Deletes only recognized simulation arrays from completed quiet runs; journal retained in results/.cleanup-records"
echo "Log: $worm_log"
echo "READ-ONLY compact report: $worm_python -m scripts.clean_results_storage --report"
