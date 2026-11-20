"""Download the manifest artifacts that list a URL, verify their SHA-256, and print how to get the rest.

Usage: python -m scripts.fetch_from_manifest [--data-root DIR]
Uses curl (system certificate store). Existing files with the right hash are skipped; mismatches are reported and kept.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(os.environ.get('R_ELEGANS_DATA_DIR', '~/OneDrive/r-elegans')).expanduser())
    parser.add_argument('--manifests', type=Path, default=Path(__file__).resolve().parents[1] / 'data_manifest')
    args = parser.parse_args()
    failures = 0
    for manifest_path in sorted(args.manifests.glob('*_manifest.json')):
        for source in json.loads(manifest_path.read_text()).get('sources', []):
            urls = [a for a in source.get('artifacts', []) if a.get('url')]
            for artifact in urls:
                target = args.data_root / artifact['relative_path']
                if target.is_file() and sha256(target) == artifact['sha256']:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(['curl', '-fsSL', '--retry', '5', '-o', str(target), artifact['url']], check=True)
                if sha256(target) != artifact['sha256']:
                    failures += 1
                    print('HASH MISMATCH', artifact['relative_path'])
            if not urls:
                print(f"{source['id']}: " + (source.get('fetch') or source.get('manual') or 'no URL listed; obtain from the DOI ' + str(source.get('doi'))))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
