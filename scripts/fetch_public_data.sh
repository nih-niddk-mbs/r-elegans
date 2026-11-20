#!/bin/sh
# Fetch every public dataset the project uses that has a download script. Run from the repository root.
# Needs R_ELEGANS_DATA_DIR. Manual sources (publisher supplements) are listed by scripts/fetch_from_manifest.py.
set -u
cd "$(dirname "$0")/.."
: "${R_ELEGANS_DATA_DIR:?Set R_ELEGANS_DATA_DIR to the data directory}"
R="$R_ELEGANS_DATA_DIR/raw"
mkdir -p "$R/functional" "$R/neuropeptides"
df -h "$R_ELEGANS_DATA_DIR" | tail -1
python3 scripts/fetch_osf_project.py e2syt "$R/functional/randi2023_osf" --skip pre_review
sh scripts/fetch_secondary_data.sh
python3 scripts/fetch_zenodo_record.py 19388374 "$R/calcium/atanas_kim_2023" --only processed_h5.tar.bz2 neuropal_label.json.bz2
python3 scripts/fetch_dandi_assets.py 001075 "$R/functional/randi2023_dandi001075_segmentation" --match desc-segmentation
python3 scripts/fetch_figshare_articles.py "$R/neuropeptides/ripoll_sanchez2023_figshare" 24187104 24187113 24187119 24517165 24517180
if [ ! -d "$R/neuropeptides/ripoll_sanchez2023_github/.git" ]; then
  git clone https://github.com/LidiaRipollSanchez/Neuropeptide-Connectome.git "$R/neuropeptides/ripoll_sanchez2023_github"
fi
python3 scripts/fetch_from_manifest.py
echo "Next: python -m scripts.verify_data"
