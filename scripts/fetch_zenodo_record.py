"""Download selected files of a Zenodo record with md5 verification and a manifest."""
import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path


def md5_sha256(path):
    a, b = hashlib.md5(), hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            a.update(block)
            b.update(block)
    return a.hexdigest(), b.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('record')
    parser.add_argument('dest')
    parser.add_argument('--only', nargs='*', default=[], help='exact file names to fetch (default: all)')
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args()
    with urllib.request.urlopen(f'https://zenodo.org/api/records/{args.record}', timeout=60) as response:
        record = json.load(response)
    files = [f for f in record['files'] if not args.only or f['key'] in args.only]
    for f in files:
        print(f['key'], f['size'], flush=True)
    if args.list_only:
        return 0
    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    manifest, ok = [], True
    for f in files:
        target = dest / f['key']
        expected = f['checksum'].split(':', 1)[1]
        if not (target.exists() and md5_sha256(target)[0] == expected):
            urllib.request.urlretrieve(f['links']['self'], target)
        md5, sha = md5_sha256(target)
        verified = md5 == expected
        ok &= verified
        manifest.append(dict(name=f['key'], size=target.stat().st_size, md5=md5, sha256=sha, zenodo_md5=expected, md5_verified=verified))
        print('done', f['key'], 'md5_verified' if verified else 'MD5-MISMATCH', flush=True)
    (dest / 'download_manifest.json').write_text(json.dumps(dict(
        source=record['links']['self_html'], doi=record['metadata'].get('doi'), title=record['metadata']['title'],
        license=record['metadata'].get('license'), files=manifest), indent=1))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
