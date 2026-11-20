"""Check a data directory against the manifests in data_manifest/ (size and SHA-256).

Usage: python -m scripts.verify_data [--data-root DIR] [--quick]
Exit status 1 if any artifact is missing or differs. Sources without listed artifacts are reported, not checked.
"""
import argparse
import hashlib
import json
import os
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
    parser.add_argument('--quick', action='store_true', help='compare sizes only')
    args = parser.parse_args()
    problems = 0
    for manifest_path in sorted(args.manifests.glob('*_manifest.json')):
        manifest = json.loads(manifest_path.read_text())
        for source in manifest.get('sources', []):
            missing = wrong = checked = 0
            for artifact in source.get('artifacts', []):
                path = args.data_root / artifact['relative_path']
                size = artifact.get('bytes')
                if not path.is_file():
                    missing += 1
                elif size is not None and path.stat().st_size != size or (not args.quick and artifact.get('sha256') and sha256(path) != artifact['sha256']):
                    wrong += 1
                else:
                    checked += 1
            problems += missing + wrong
            how = source.get('fetch') or source.get('manual') or 'see manifest URLs (python -m scripts.fetch_from_manifest)'
            status = 'OK' if not (missing or wrong) else 'PROBLEM'
            print(f"{status:7s} {source['id']}: {checked} ok, {missing} missing, {wrong} differ"
                  + ('' if status == 'OK' else f'\n          get it: {how}'))
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
