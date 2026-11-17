#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
worm_out="/home/carsonc/r-elegans-results/relay-range-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup /home/carsonc/miniconda3/envs/r-elegans/bin/python -u -m scripts.total_job_budget \
    --pipeline relay-range --continue-relay --optimization-minutes 180 --total-minutes 360 \
    --source-run /home/carsonc/r-elegans-results/relay-range-20261117T031207Z-308052 \
    --parent-sha256 a84145d65ff503e71b6e6f0bf54ec69d396eed29cc3dfc3010d1c13bb89f4b57 \
    --experiment-id E090 --seed 9000 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "E090:3h optimization from pinned E089 selected weights; final validation and automatic audit afterwards"
echo "Same objective and original bounds; fresh CMA/local recentering, not exact optimizer resume"
echo "6h overall watchdog;4CPUworkers x4cores; all models/RNG retained, intermediate raw traces in memory"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Audit: $worm_out-audit/summary.json"
echo "Log: $worm_out.log"
