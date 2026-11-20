#!/bin/sh
set -eu
minutes=${1:-180}
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_ENABLE_X64=true CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$worm_python" -c 'import jax,h5py,scipy,optax,flax; assert jax.config.x64_enabled; print("Interpreter ready;", jax.devices(), "float64")'
if pgrep -f scripts.train_atlas_gradient > /dev/null; then echo "Another atlas gradient run is active; stop it first"; exit 3; fi
worm_data=${R_ELEGANS_DATA_DIR:-/home/carsonc/r-elegans-data}
test -d "$worm_data/raw/functional/randi2023_dandi001075_segmentation" || { echo "Atlas traces missing under $worm_data"; exit 2; }
start_run=${E101_RUN:-/home/carsonc/r-elegans-results/atlas-gradient-20261120T182817Z-664349}
expected=$("$worm_python" -c "import json,sys; print(json.load(open(sys.argv[1]+'/summary.json'))['final_sha256'])" "$start_run")
actual=$(sha256sum "$start_run/network_final.npz" | cut -d' ' -f1)
test "$expected" = "$actual" || { echo "E101 final checkpoint hash mismatch: $actual vs $expected"; exit 4; }
echo "Start checkpoint (E101 final) SHA-256: $actual"
worm_out="/home/carsonc/r-elegans-results/atlas-gradient-continue-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.train_atlas_gradient --experiment-id E102 \
    --network "$start_run/network_final.npz" --anchor-network r_elegans/assets/mac_rhythm_v1_polarized/network.npz \
    --data-root "$worm_data" --output-dir "$worm_out" --minutes "$minutes" --final-reserve-minutes 10 \
    --substeps 40 --snapshot-every 250 > "$worm_out.log" 2>&1 < /dev/null &
echo "E102: continuation of E101 (fresh optimizer, substeps 40), wall budget $minutes minutes (10 min final reserve)"
echo "PID: $!"
echo "Log: $worm_out.log"
echo "Report: python -m scripts.report_atlas_gradient --run $worm_out"
