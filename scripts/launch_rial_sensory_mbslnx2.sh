#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
worm_out="/home/carsonc/r-elegans-results/rial-sensory-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup /home/carsonc/miniconda3/envs/r-elegans/bin/python -u -m scripts.total_job_budget \
    --pipeline rial-sensory --total-minutes 480 \
    --source-run /home/carsonc/r-elegans-results/avb-training-20261115T041708Z-122855 \
    --output-dir "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "Sensory-triggered RIAL amplitude curriculum2/5/10%; maximum8h TOTAL including final validation/cleanup"
echo "Four persistent CPU workers x4cores; unknown control parameters fit, motor/NMJ/water parameters fixed"
echo "No diagnostic clamp in fitting; raw training traces stay in memory; all weights/optimizer/RNG retained"
echo "Early-response component only; full start/stop and swimming stay unqualified"
echo "PID: $!"
echo "Report: $worm_out/summary.json"
echo "Log: $worm_out.log"
