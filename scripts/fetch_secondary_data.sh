#!/bin/sh
# Verified secondary datasets (Zenodo records, CeNGEN tables) into the data directory.
set -u
cd "$(dirname "$0")/.."
R="${R_ELEGANS_DATA_DIR:-$HOME/OneDrive/r-elegans}/raw"
mkdir -p "$R/expression/cengen"
python3 scripts/fetch_zenodo_record.py 19414639 "$R/calcium/dag2023_serotonin_zenodo19414639"
python3 scripts/fetch_zenodo_record.py 3906530 "$R/calcium/neuropal_yemini2021_zenodo3906530"
python3 scripts/fetch_zenodo_record.py 19388374 "$R/calcium/atanas_kim_2023_summaries_zenodo19388374" --only \
  sampled_tau_vals_median.h5.bz2 relative_encoding_strength_median.h5.bz2 tuning_strength.h5.bz2 \
  encoding_changes_corrected.h5.bz2 neuron_categorization.h5.bz2 fit_ranges.h5.bz2 \
  neuropal_label.json.bz2 analysis_dict.jld2.bz2 processed_h5_scrambled.tar.bz2
for f in 021821_liberal_threshold1.csv 021821_medium_threshold2.csv 021821_conservative_threshold3.csv \
  021821_stringent_threshold4.csv Average_integrated_TMM_counts_lengthNormalized_111521.tsv \
  Barrett_et_al_2022_CeNGEN_bulk_RNAseq_data.tsv CeNGEN_aggr_bio_reps_101821.tsv \
  Barrett_et_al_2022_CeNGEN_bulk_RNAseq_geneLevel_metadata.tsv Ce_bulk_RNAseq_neuronal_counts_WS274.tsv; do
  if curl -fsSL -o "$R/expression/cengen/$f" "https://cengen.org/storage/$f"; then
    echo "done $f $(shasum -a 256 "$R/expression/cengen/$f" | cut -d' ' -f1) $(wc -c < "$R/expression/cengen/$f")"
  else
    echo "FAILED $f"
  fi
done
