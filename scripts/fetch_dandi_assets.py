"""Download DANDI assets whose path contains a substring (e.g. 'desc-segmentation'), with size/sha256 checks and a manifest."""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

API = 'https://api.dandiarchive.org/api'


def retry(action, attempts=6):
    for attempt in range(attempts):
        try:
            return action()
        except Exception as error:
            print('retry', attempt + 1, error, file=sys.stderr, flush=True)
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))


def get_json(url):
    def fetch():
        with urllib.request.urlopen(url, timeout=60) as response:
            return json.load(response)
    return retry(fetch)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dandiset')
    parser.add_argument('dest')
    parser.add_argument('--version', default='draft')
    parser.add_argument('--match', default='', help='substring the asset path must contain')
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args()
    url = f'{API}/dandisets/{args.dandiset}/versions/{args.version}/assets/?page_size=300'
    assets = []
    while url:
        page = get_json(url)
        assets += [a for a in page['results'] if args.match in a['path']]
        url = page.get('next')
    total = sum(a['size'] for a in assets)
    print(len(assets), 'assets', round(total / 1e9, 3), 'GB', flush=True)
    if args.list_only:
        return 0
    dest = Path(args.dest).expanduser()
    manifest, ok = [], True
    for a in assets:
        digest = get_json(f"{API}/assets/{a['asset_id']}/info/").get('digest', {})
        expected = digest.get('dandi:sha2-256')
        target = dest / a['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not (target.exists() and target.stat().st_size == a['size']):
            part = target.with_name(target.name + '.part')
            retry(lambda: urllib.request.urlretrieve(f"{API}/assets/{a['asset_id']}/download/", part))
            part.rename(target)
        digest_ok = target.stat().st_size == a['size']
        actual = sha256(target)
        if expected:
            digest_ok &= actual == expected
        ok &= digest_ok
        manifest.append(dict(path=a['path'], asset_id=a['asset_id'], size=target.stat().st_size, sha256=actual,
                             dandi_sha256=expected, size_ok=target.stat().st_size == a['size'], verified=digest_ok))
        print('done' if digest_ok else 'MISMATCH', a['path'], 'sha256-checked' if expected else 'size-only', flush=True)
    (dest / 'download_manifest.json').write_text(json.dumps(dict(
        source=f'https://dandiarchive.org/dandiset/{args.dandiset}/{args.version}', match=args.match, files=manifest), indent=1))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
