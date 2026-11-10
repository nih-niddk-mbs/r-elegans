#!/bin/sh
# E048: genuinely closed-loop sensing on pinned full-network winner7001.
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
if [ "${1:-}" = --worker-process ]; then
    trap 'worm_status=$?; echo "EMBODIED_SENSES_RUN_EXIT_CODE=$worm_status"' 0
    "$worm_python" -u -m scripts.diagnose_embodied_senses --reference "$2" \
        --output-dir "$3" --duration "$4" --medium both --refine
    exit 0
fi
worm_parent=${1:-/home/carsonc/r-elegans-results/autonomous-posterior-motor-20261110T140125786528Z/seed-7001}
worm_duration=${2:-20}
"$worm_python" -c '
import importlib.util,math,sys
from scripts.assess_wet_surface import verify_reference
source,network,readout=verify_reference(sys.argv[1])
assert math.isfinite(float(sys.argv[2])) and 5<=float(sys.argv[2])<=600, "Duration must be 5..600 simulated seconds"
for name in ("jax","numpy","scipy"):
    assert importlib.util.find_spec(name), "Missing package: "+name
print("Pinned fixed winner:",source)
' "$worm_parent" "$worm_duration"
worm_cpus=$("$worm_python" -c 'import os; from scripts.benchmark_cpu import physical_core_cpus; cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0))); assert known and len(cpus)>=4; print(",".join(map(str,cpus[:4])))')
worm_results=/home/carsonc/r-elegans-results
mkdir -p "$worm_results"
worm_out="$worm_results/embodied-senses-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export JAX_PLATFORMS=cpu JAX_DEFAULT_MATMUL_PRECISION=highest CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
nohup taskset -c "$worm_cpus" sh "$worm_repo/scripts/launch_embodied_senses_mbslnx2.sh" \
    --worker-process "$worm_parent" "$worm_out" "$worm_duration" > "$worm_out.log" 2>&1 < /dev/null &
echo "Closed-loop sensory diagnostic; no fitting or gait teacher"
echo "PID: $!; CPUs: $worm_cpus; simulated seconds per case: $worm_duration"
echo "Ten cases: water and wet agar, input-block controls and timestep refinements"
echo "Results: $worm_out"
echo "Wet-surface preview: $worm_out/wet-agar-closed-loop/motion.html"
echo "Log: $worm_out.log"
