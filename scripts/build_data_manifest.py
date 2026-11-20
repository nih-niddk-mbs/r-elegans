"""Write data_manifest/session_sources_manifest.json: sources fetched by the project's download scripts, with hashes.

Same schema as the earlier core manifests (sources -> artifacts with relative_path, bytes, sha256). Sources that have
no per-file URL record the command that fetches them (`fetch`) or say they are manual. Hashes are computed from the
local data directory, so they certify what this project used, not what a publisher currently serves.
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SOURCES = [
    dict(id='atanas_kim2023_processed', title='Atanas & Kim 2023 brain-wide recordings with behavior (Zenodo 19388374): processed_h5 archive, labels and the extracted files used by E049',
         doi='10.1016/j.cell.2023.07.035', license_note='See the Zenodo record.', folder='raw/calcium/atanas_kim_2023',
         fetch='python3 scripts/fetch_zenodo_record.py 19388374 $R_ELEGANS_DATA_DIR/raw/calcium/atanas_kim_2023 --only processed_h5.tar.bz2 neuropal_label.json.bz2; '
               'then extract the archive into processed_h5_E049/ (tar -xjf processed_h5.tar.bz2; the exact extraction command used for E049 was not recorded)',
         verification='md5 from Zenodo for the archives; extracted files are ours'),
    dict(id='randi2023_osf', title='Neural signal propagation atlas of C. elegans: processed exports (OSF e2syt)', doi='10.1038/s41586-023-06683-4',
         license_note='See the OSF project; retain attribution to Randi et al. 2023.', folder='raw/functional/randi2023_osf',
         fetch='python3 scripts/fetch_osf_project.py e2syt $R_ELEGANS_DATA_DIR/raw/functional/randi2023_osf --skip pre_review', verification='md5 from OSF'),
    dict(id='randi2023_dandi001075_segmentation', title='Neural signal propagation atlas (DANDI 001075), segmentation NWB files only', doi='10.1038/s41586-023-06683-4',
         license_note='CC-BY-4.0 (DANDI metadata).', folder='raw/functional/randi2023_dandi001075_segmentation',
         fetch='python3 scripts/fetch_dandi_assets.py 001075 $R_ELEGANS_DATA_DIR/raw/functional/randi2023_dandi001075_segmentation --match desc-segmentation',
         verification='size only at the source; sha256 here is ours'),
    dict(id='dag2023_zenodo19414639', title='Datasets for Dissecting the functional organization of the C. elegans serotonergic system (Zenodo 19414639)', doi='10.1016/j.cell.2023.04.023',
         license_note='See the Zenodo record.', folder='raw/calcium/dag2023_serotonin_zenodo19414639',
         fetch='python3 scripts/fetch_zenodo_record.py 19414639 $R_ELEGANS_DATA_DIR/raw/calcium/dag2023_serotonin_zenodo19414639', verification='md5 from Zenodo'),
    dict(id='neuropal_yemini2021_zenodo3906530', title='NeuroPAL Data (Zenodo 3906530)', doi='10.1016/j.cell.2020.12.012',
         license_note='MIT license (Zenodo record).', folder='raw/calcium/neuropal_yemini2021_zenodo3906530',
         fetch='python3 scripts/fetch_zenodo_record.py 3906530 $R_ELEGANS_DATA_DIR/raw/calcium/neuropal_yemini2021_zenodo3906530', verification='md5 from Zenodo'),
    dict(id='atanas_kim2023_summaries_zenodo19388374', title='Atanas & Kim 2023 summary and scrambled-control files (Zenodo 19388374, subset)', doi='10.1016/j.cell.2023.07.035',
         license_note='See the Zenodo record.', folder='raw/calcium/atanas_kim_2023_summaries_zenodo19388374',
         fetch='sh scripts/fetch_secondary_data.sh (the Atanas step lists the exact files)', verification='md5 from Zenodo'),
    dict(id='cengen_tables', title='CeNGEN thresholded expression and count tables', doi='10.1016/j.cell.2021.06.023',
         license_note='See cengen.org.', folder='raw/expression/cengen', fetch='sh scripts/fetch_secondary_data.sh (curl from https://cengen.org/storage/)',
         verification='sha256 here is ours'),
    dict(id='ripoll_sanchez2023_figshare', title='C. elegans neuropeptide connectome datasets (Figshare collection 6895870, five articles)', doi='10.6084/m9.figshare.c.6895870.v1',
         license_note='CC BY 4.0.', folder='raw/neuropeptides/ripoll_sanchez2023_figshare',
         fetch='python3 scripts/fetch_figshare_articles.py $R_ELEGANS_DATA_DIR/raw/neuropeptides/ripoll_sanchez2023_figshare 24187104 24187113 24187119 24517165 24517180',
         verification='md5 from Figshare'),
    dict(id='ripoll_sanchez2023_supplement', title='Ripoll-Sanchez et al. 2023 Neuron supplementary tables (mmc1-8)', doi='10.1016/j.neuron.2023.09.043',
         license_note='Publisher supplementary files; do not redistribute.', folder='raw/neuropeptides/ripoll_sanchez2023',
         manual='Download from the article page (Supplemental information) under the DOI above; filenames 1-s2.0-S0896627323007560-mmc*.xlsx/pdf.', verification='sha256 here is ours'),
    dict(id='beets2023_supplement', title='Beets et al. 2023 Cell Reports supplementary tables (mmc1-11)', doi='10.1016/j.celrep.2023.113058',
         license_note='Publisher supplementary files; do not redistribute.', folder='raw/neuropeptides/beets2023',
         manual='Download from the article page (Supplemental information) under the DOI above; filenames 1-s2.0-S2211124723010690-mmc*.xlsx/zip/pdf.', verification='sha256 here is ours'),
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path.home() / 'OneDrive/r-elegans')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'data_manifest/session_sources_manifest.json')
    args = parser.parse_args()
    sources = []
    for spec in SOURCES:
        folder = args.data_root / spec['folder']
        artifacts = []
        for path in sorted(p for p in folder.rglob('*') if p.is_file() and p.name not in ('download_manifest.json', 'SHA256SUMS', '.DS_Store') and '.git/' not in str(p)):
            if path.suffix == '.part':
                continue
            artifacts.append(dict(relative_path=str(path.relative_to(args.data_root)), bytes=path.stat().st_size, sha256=sha256(path)))
        entry = {k: v for k, v in spec.items() if k != 'folder'}
        entry.update(artifacts=artifacts, artifact_count=len(artifacts), total_bytes=sum(a['bytes'] for a in artifacts))
        sources.append(entry)
        print(spec['id'], len(artifacts), 'files', round(entry['total_bytes'] / 1e6, 1), 'MB', flush=True)
    github = args.data_root / 'raw/neuropeptides/ripoll_sanchez2023_github'
    commit = subprocess.check_output(['git', '-C', str(github), 'rev-parse', 'HEAD'], text=True).strip() if (github / '.git').exists() else None
    sources.append(dict(id='ripoll_sanchez2023_github', title='LidiaRipollSanchez/Neuropeptide-Connectome (code and data)', license_note='See the repository LICENSE.',
                        fetch=f'git clone https://github.com/LidiaRipollSanchez/Neuropeptide-Connectome.git $R_ELEGANS_DATA_DIR/raw/neuropeptides/ripoll_sanchez2023_github && git -C $R_ELEGANS_DATA_DIR/raw/neuropeptides/ripoll_sanchez2023_github checkout {commit}',
                        commit=commit, artifacts=[], artifact_count=0, total_bytes=0))
    manifest = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(timespec='seconds'), artifact_count=sum(s['artifact_count'] for s in sources),
                    total_bytes=sum(s['total_bytes'] for s in sources), sources=sources)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1))
    print('wrote', args.out, manifest['artifact_count'], 'artifacts', round(manifest['total_bytes'] / 1e9, 2), 'GB')


if __name__ == '__main__':
    main()
