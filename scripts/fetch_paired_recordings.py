"""Fetch pinned E049 sources and prepare external targets for pull-and-run use."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
from urllib.request import Request,urlopen

from r_elegans.data.paths import get_data_root
from scripts.experiment_records import file_sha256


def fetch(url,path,expected):
    if path.exists():
        if file_sha256(path)!=expected:raise ValueError('Existing source checksum mismatch: '+str(path))
        return
    temporary=path.with_suffix(path.suffix+'.part')
    print('Downloading '+url,flush=True)
    with urlopen(Request(url,headers={'User-Agent':'r-elegans/E049'}),timeout=60) as response,temporary.open('wb') as stream:
        shutil.copyfileobj(response,stream,length=1024*1024)
    if file_sha256(temporary)!=expected:raise ValueError('Downloaded source checksum mismatch: '+url)
    temporary.replace(path)


def extract(archive,destination):
    """Stream only regular allowlisted files; reject traversal/links/duplicates."""
    destination.mkdir(parents=True,exist_ok=False)
    seen=set()
    with tarfile.open(archive,'r|bz2') as tar:
        for member in tar:
            if member.isdir():continue
            p=Path(member.name)
            if (not member.isfile() or p.is_absolute() or '..' in p.parts or p.name in seen
                    or not (p.name.endswith('-data.h5') or p.name=='h5_sha256.csv')):
                raise ValueError('Unsafe/unexpected archive entry: '+member.name)
            seen.add(p.name)
            with tar.extractfile(member) as source,(destination/p.name).open('wb') as stream:
                shutil.copyfileobj(source,stream,length=1024*1024)
    if len([n for n in seen if n.endswith('.h5')])!=68:
        raise ValueError('Expected 68 original recording HDF5 files')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    args=parser.parse_args()
    root=get_data_root(args.data_root,must_exist=False)
    raw=root/'raw/calcium/atanas_kim_2023';raw.mkdir(parents=True,exist_ok=True)
    record=json.loads((Path(__file__).resolve().parents[1]/'experiments/records/E049-paired-data-admission.json').read_text())
    for name,checks in record['archive_checksums'].items():
        fetch('https://zenodo.org/records/19388374/files/'+name+'?download=1',raw/name,checks['sha256'])
    metadata=raw/'atanas_kim_2023.csv'
    fetch('https://raw.githubusercontent.com/flavell-lab/WormWideWeb-data/'+record['metadata_revision']+
          '/activity/raw/atanas_kim_2023.csv',metadata,record['metadata_sha256'])
    extracted=raw/'processed_h5_E049'
    if not extracted.exists():extract(raw/'processed_h5.tar.bz2',extracted)
    out=root/'processed/calcium/paired_E049_v1'
    if not out.exists():
        import subprocess,sys
        subprocess.run([sys.executable,'-u','-m','scripts.prepare_paired_recordings',
            '--recordings',str(extracted),'--metadata',str(metadata),
            '--labels',str(raw/'neuropal_label.json.bz2'),'--output-dir',str(out)],check=True)
    manifest=json.loads((out/'manifest.json').read_text())
    if (manifest['metadata_sha256']!=record['metadata_sha256'] or
        file_sha256(out/'splits.json')!=record['split_sha256'] or
        manifest['split_sha256']!=record['split_sha256']):
        raise ValueError('Prepared data are not the frozen E049 cohort')
    for row in manifest['records']:
        if row.get('targets_file') and file_sha256(out/row['targets_file'])!=row['targets_sha256']:
            raise ValueError('Prepared target checksum mismatch')
    print('Verified prepared targets: '+str(out),flush=True)


if __name__=='__main__':main()
