#!/bin/sh
# Download the verified public datasets directly on mbslnx2 (checksums checked, manifests written).
set -u
cd "$(dirname "$0")/.."
export R_ELEGANS_DATA_DIR="${R_ELEGANS_DATA_DIR:-/home/carsonc/r-elegans-data}"
R="$R_ELEGANS_DATA_DIR/raw"
mkdir -p "$R/functional"
echo "Data root: $R_ELEGANS_DATA_DIR"
df -h "$R_ELEGANS_DATA_DIR" | tail -1
python3 scripts/fetch_osf_project.py e2syt "$R/functional/randi2023_osf" --skip pre_review
sh scripts/fetch_secondary_data.sh
echo "Finished. Manifests:"
find "$R" -name download_manifest.json
