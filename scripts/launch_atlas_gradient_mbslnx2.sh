#!/bin/sh
set -eu
minutes=${1:-60}
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_ENABLE_X64=true CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$worm_python" -c 'import jax,h5py,scipy,optax,flax; assert jax.config.x64_enabled; print("Interpreter ready;", jax.devices(), "float64")'
worm_data=${R_ELEGANS_DATA_DIR:-/home/carsonc/r-elegans-data}
test -d "$worm_data/raw/functional/randi2023_dandi001075_segmentation" || { echo "Atlas traces missing under $worm_data"; exit 2; }
worm_out="/home/carsonc/r-elegans-results/atlas-gradient-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.train_atlas_gradient \
    --network r_elegans/assets/mac_rhythm_v1_polarized/network.npz \
    --data-root "$worm_data" --output-dir "$worm_out" --minutes "$minutes" > "$worm_out.log" 2>&1 < /dev/null &
echo "E101: gradient fit to the Randi atlas, wall budget $minutes minutes (5 min final reserve)"
echo "PID: $!"
echo "Log: $worm_out.log"
echo "Report: python -m scripts.report_atlas_gradient --run $worm_out"
