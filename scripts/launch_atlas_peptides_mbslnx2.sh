#!/bin/sh
set -eu
minutes=${1:-120}
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-.45}
"$worm_python" -c 'import jax,h5py,scipy,optax,flax; assert jax.config.x64_enabled; print("Interpreter ready;", jax.devices(), "float64")'
worm_data=${R_ELEGANS_DATA_DIR:-/home/carsonc/r-elegans-data}
test -d "$worm_data/raw/functional/randi2023_dandi001075_segmentation" || { echo "Atlas traces missing under $worm_data"; exit 2; }
bundle=r_elegans/assets/mac_rhythm_v1_polarized/network.npz
e104_run=${E104_RUN:-/home/carsonc/r-elegans-results/allparams-20261121T023851Z-685266}
control="$e104_run/network_control/network_best_val.npz"
expected=$("$worm_python" -c "import json,sys; print(json.load(open(sys.argv[1]+'/network_control/summary.json'))['best_val_sha256'])" "$e104_run")
actual=$(sha256sum "$control" | cut -d' ' -f1)
test "$expected" = "$actual" || { echo "E104 control best-validation hash mismatch: $actual vs $expected"; exit 4; }
echo "Start for the control-based variants (E104 network_control best-validation) SHA-256: $actual"
root="/home/carsonc/r-elegans-results/peptides-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$root"
launch() {
    name=$1; gpu=$2; shift 2
    CUDA_VISIBLE_DEVICES=$gpu nohup "$worm_python" -u -m scripts.train_atlas_peptides --data-root "$worm_data" \
        --output-dir "$root/$name" --minutes "$minutes" "$@" > "$root/$name.log" 2>&1 < /dev/null &
    echo "$name GPU $gpu PID $!"
}
launch pep_joint_lr005 0 --network "$bundle" --train-network yes --learning-rate .005
launch pep_joint_lr002 1 --network "$bundle" --train-network yes --learning-rate .002
launch pep_only_from_ctrl 0 --network "$control" --anchor-network "$bundle" --train-network no --learning-rate .01
launch pep_joint_from_ctrl 1 --network "$control" --anchor-network "$bundle" --train-network yes --learning-rate .002
echo "E105: four variants, two per GPU (memory fraction $XLA_PYTHON_CLIENT_MEM_FRACTION each), $minutes minute wall budget, 10 min final reserve"
echo "Sweep folder: $root"
echo "Progress: tail -n 2 $root/*.log | cut -c1-300"
echo "Report: python -m scripts.report_atlas_peptides --sweep $root"
