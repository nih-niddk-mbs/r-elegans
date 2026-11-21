#!/bin/sh
set -eu
minutes=${1:-120}
cores_per_job=${CORES_PER_JOB:-4}
first_core=${FIRST_CORE:-8}
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
run=${E099_RUN:-/home/carsonc/r-elegans-results/pathway-continuation-20261120T001737Z-577769}
checkpoint=${E099_SHA256:-57fc914e84050099e6be190d1efc418fc207962e81fc28520d3f4bfd180558a7}
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=$cores_per_job PYTHONPATH=.
"$worm_python" -c 'import jax; assert jax.config.x64_enabled; print("Interpreter ready; CPU float64")'
command -v taskset > /dev/null || { echo "taskset required"; exit 2; }
sweep="/home/carsonc/r-elegans-results/polarity-sweep-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$sweep/gait"
"$worm_python" -m scripts.make_polarity_variants --run "$run" --checkpoint-sha256 "$checkpoint" --output-dir "$sweep/variants" > "$sweep/variants.log" 2>&1
names=$("$worm_python" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['variants']))" "$sweep/variants/variants.json")
i=0
for name in $names; do
    lo=$((first_core + i * cores_per_job)); hi=$((lo + cores_per_job - 1))
    nohup taskset -c "$lo-$hi" "$worm_python" -u -m scripts.assay_checkpoint_gait --run "$run" \
        --fixed-network "$sweep/variants/$name/network.npz" --output-dir "$sweep/gait/$name" --only polarized --minutes "$minutes" \
        > "$sweep/gait/$name.log" 2>&1 < /dev/null &
    echo "$name cores $lo-$hi PID $!"
    i=$((i + 1))
done
echo "Sweep folder: $sweep"
echo "Progress: grep -l '\"label\": \"polarized\"' $sweep/gait/*.log | wc -l   (of $i)"
echo "Report: python -m scripts.report_polarity_sweep --sweep $sweep --extra full=/home/carsonc/r-elegans-results/e099-polarized-gait-20261121T000743Z/comparison.json"
