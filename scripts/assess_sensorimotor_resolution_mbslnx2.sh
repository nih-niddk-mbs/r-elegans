#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
cd "$worm_repo"
export JAX_PLATFORMS=cpu JAX_DEFAULT_MATMUL_PRECISION=highest
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [ "${1:-}" = --worker-process ]; then
    trap 'worm_status=$?; echo "SENSORIMOTOR_RESOLUTION_EXIT_CODE=$worm_status"' 0
    "$worm_python" -u -m scripts.assess_sensorimotor_resolution --run "$2" --output-dir "$3" \
        --expected-sha256 6cd7b0fa95c545c354f83334fe32aaf0551b8dd4e3d4e7f98babc944849669bf
    exit 0
fi
worm_run=/home/carsonc/r-elegans-results/sensorimotor-20261111T193810Z-773573
"$worm_python" -c 'import hashlib,json,sys; from pathlib import Path; p=Path(sys.argv[1]); assert json.loads((p/"summary.json").read_text())["passed"] is True; assert hashlib.sha256((p/"network_best.npz").read_bytes()).hexdigest()=="6cd7b0fa95c545c354f83334fe32aaf0551b8dd4e3d4e7f98babc944849669bf"' "$worm_run"
worm_cpus=$("$worm_python" -c 'import os; from scripts.benchmark_cpu import physical_core_cpus; cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0))); assert known and len(cpus)>=4; print(",".join(map(str,cpus[:4])))')
worm_out="/home/carsonc/r-elegans-results/cue-resolution-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup taskset -c "$worm_cpus" sh "$worm_repo/scripts/assess_sensorimotor_resolution_mbslnx2.sh" \
    --worker-process "$worm_run" "$worm_out" > "$worm_out.log" 2>&1 < /dev/null &
echo "Read-only low-dose resolution assay; pinned E058 winner unchanged"
echo "PID: $!; CPUs: $worm_cpus"
echo "Report: $worm_out/summary.json"
echo "Log: $worm_out.log"
