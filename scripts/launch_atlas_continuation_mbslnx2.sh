#!/bin/sh
set -eu
worm_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$worm_repo"
worm_python=/home/carsonc/miniconda3/envs/r-elegans/bin/python
export JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 "$worm_python" -c 'import jax,scipy; assert jax.config.x64_enabled; assert len(jax.devices("gpu"))>=2; print("Dedicated r-elegans environment ready; two A100s, float64")'
export JAX_PLATFORMS=cpu
worm_parent=$("$worm_python" -c 'from scripts.atlas_fit_continuation import latest_parent; print(latest_parent("/home/carsonc/r-elegans-results"))')
"$worm_python" -c 'import sys; from scripts.atlas_fit_continuation import qualified_parent; d=qualified_parent(sys.argv[1]); print("Pinned parent:",sys.argv[1],"SHA256:",d["summary"]["selected_sha256"])' "$worm_parent"
worm_out="/home/carsonc/r-elegans-results/atlas-continuation-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup "$worm_python" -u -m scripts.run_stable_atlas_fit --output-dir "$worm_out" --own-state --measured-schedule --parent-run "$worm_parent" > "$worm_out.log" 2>&1 < /dev/null &
echo "E116: continue qualified E115 weights; fresh optimizer; original cumulative bounds preserved"
echo "Maximum 240 minutes TOTAL: up to 120 optimization, remaining 120 for validation/audit"
echo "Measured task scheduling; two A100 workers; each model settles its own states; full numerical gates during selection"
echo "PID: $!; safe to disconnect"
echo "Log: $worm_out.log"
echo "Report: $worm_python -m scripts.run_stable_atlas_fit --output-dir $worm_out --report"
