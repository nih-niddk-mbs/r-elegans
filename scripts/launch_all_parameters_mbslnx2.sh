#!/bin/sh
set -eu
minutes=${1:-180}
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-.45}
"$worm_python" -c 'import jax,h5py,scipy,optax,flax; assert jax.config.x64_enabled; print("Interpreter ready;", jax.devices(), "float64")'
worm_data=${R_ELEGANS_DATA_DIR:-/home/carsonc/r-elegans-data}
test -d "$worm_data/raw/functional/randi2023_dandi001075_segmentation" || { echo "Atlas traces missing under $worm_data"; exit 2; }
bundle=r_elegans/assets/mac_rhythm_v1_polarized/network.npz
e102_run=${E102_RUN:-/home/carsonc/r-elegans-results/atlas-gradient-continue-20261120T193418Z-668699}
expected=$("$worm_python" -c "import json,sys; print(json.load(open(sys.argv[1]+'/summary.json'))['final_sha256'])" "$e102_run")
actual=$(sha256sum "$e102_run/network_final.npz" | cut -d' ' -f1)
test "$expected" = "$actual" || { echo "E102 final checkpoint hash mismatch: $actual vs $expected"; exit 4; }
root="/home/carsonc/r-elegans-results/allparams-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$root"
launch() {
    name=$1; gpu=$2; shift 2
    CUDA_VISIBLE_DEVICES=$gpu nohup "$worm_python" -u -m scripts.train_all_parameters --data-root "$worm_data" \
        --output-dir "$root/$name" --minutes "$minutes" "$@" > "$root/$name.log" 2>&1 < /dev/null &
    echo "$name GPU $gpu PID $!"
}
launch all_lr005 0 --network "$bundle" --parameter-set all --learning-rate .005
launch all_lr002 1 --network "$bundle" --parameter-set all --learning-rate .002
launch all_from_e102 0 --network "$e102_run/network_final.npz" --anchor-network "$bundle" --parameter-set all --learning-rate .003
launch network_control 1 --network "$bundle" --parameter-set network --learning-rate .005
echo "E104: four variants, two per GPU (memory fraction $XLA_PYTHON_CLIENT_MEM_FRACTION each), $minutes minute wall budget, 10 min final reserve"
echo "Sweep folder: $root"
echo "Progress: tail -n 2 $root/*.log | cut -c1-300"
echo "Report: python -m scripts.report_all_parameters --sweep $root"
