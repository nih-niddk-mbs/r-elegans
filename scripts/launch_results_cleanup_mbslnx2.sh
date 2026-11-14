#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_archive=${1:-/data/carsonc/r-elegans-results}
worm_log="/tmp/r-elegans-results-cleanup-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
nohup /home/carsonc/miniconda3/envs/r-elegans/bin/python -u -m scripts.archive_results_mbslnx2 \
    --source /home/carsonc/r-elegans-results --destination "$worm_archive" \
    --apply > "$worm_log" 2>&1 < /dev/null &
echo "Copies inactive results to /data; deletes /home duplicates only after checksum verification"
echo "Active/recent runs skipped; old paths preserved as symlinks. Originals retained on any verification error."
echo "PID: $!"
echo "Log: $worm_log"
