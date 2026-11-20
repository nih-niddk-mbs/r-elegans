#!/bin/sh
set -eu
minutes=${1:-180}
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$worm_python" -c 'import jax,h5py,scipy; assert jax.config.x64_enabled; print("Interpreter ready; CPU float64")'
worm_data=${R_ELEGANS_DATA_DIR:-/home/carsonc/r-elegans-data}
test -d "$worm_data/raw/functional/randi2023_dandi001075_segmentation" || { echo "Atlas traces missing under $worm_data"; exit 2; }
worm_out="/home/carsonc/r-elegans-results/atlas-baseline-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.run_atlas_baseline \
    --run /home/carsonc/r-elegans-results/pathway-continuation-20261120T001737Z-577769 \
    --checkpoint-sha256 57fc914e84050099e6be190d1efc418fc207962e81fc28520d3f4bfd180558a7 \
    --data-root "$worm_data" --output-dir "$worm_out" --budget-minutes "$minutes" > "$worm_out.log" 2>&1 < /dev/null &
echo "E100: maximum $minutes minutes (10 min reserve) on the audited E099 selected checkpoint; no training"
echo "PID: $!"
echo "Log: $worm_out.log"
echo "Report: python -m scripts.report_atlas_baseline --run $worm_out"
