#!/bin/sh
# Short E050 joint-data pilot, detached; same absolute conda interpreter throughout.
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
if [ "${1:-}" = --worker-process ]; then
    trap 'worm_status=$?; echo "PAIRED_FIT_EXIT_CODE=$worm_status"' 0
    "$worm_python" -u -m scripts.fetch_paired_recordings --data-root "$5"
    "$worm_python" -u -m scripts.fit_paired_recordings \
        --reference "$2" --output-dir "$3" --prepared-data "$5/processed/calcium/paired_E049_v1" \
        --uids 2022-06-14-01 2022-06-14-07 --frames 16 --iterations 100000 \
        --minutes "$4" --media wet-agar --maximum-factor 2 --refine
    exit 0
fi
worm_minutes=${1:-20}
worm_parent=${2:-/home/carsonc/r-elegans-results/autonomous-posterior-motor-20261110T140125786528Z/seed-7001}
worm_data=${R_ELEGANS_DATA_DIR:-/home/carsonc/r-elegans-data}
"$worm_python" -c '
import importlib.util,math,sys
from scripts.assess_wet_surface import verify_reference
assert math.isfinite(float(sys.argv[1])) and float(sys.argv[1])>0
verify_reference(sys.argv[2])
for name in ("jax","numpy","scipy","h5py","flax"):
    assert importlib.util.find_spec(name), "Missing package: "+name
' "$worm_minutes" "$worm_parent"
worm_cpus=$("$worm_python" -c 'import os; from scripts.benchmark_cpu import physical_core_cpus; cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0))); assert known and len(cpus)>=4; print(",".join(map(str,cpus[:4])))')
worm_results=/home/carsonc/r-elegans-results
mkdir -p "$worm_results"
worm_out="$worm_results/paired-fit-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export JAX_PLATFORMS=cpu JAX_DEFAULT_MATMUL_PRECISION=highest
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
nohup taskset -c "$worm_cpus" sh "$worm_repo/scripts/launch_paired_fit_mbslnx2.sh" \
    --worker-process "$worm_parent" "$worm_out" "$worm_minutes" "$worm_data" > "$worm_out.log" 2>&1 < /dev/null &
echo "PID: $!; CPUs: $worm_cpus"
echo "Fit budget: $worm_minutes minutes, plus initial download/preparation and final refinement"
echo "First download: about 569 MB; extracted and processed files stay outside Git"
echo "Exploratory paired calcium/bend statistics, not calibrated mounted-medium physics"
echo "Results: $worm_out"
echo "Log: $worm_out.log"
