"""Download files of an OSF project's osfstorage with checksum verification and a manifest."""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

API = 'https://api.osf.io/v2/nodes/{node}/files/osfstorage/'


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def walk(url):
    while url:
        page = get_json(url)
        for item in page['data']:
            yield item
        url = page['links'].get('next')


def listing(url, prefix=''):
    for item in walk(url):
        attributes = item['attributes']
        if attributes['kind'] == 'folder':
            sub = item['relationships']['files']['links']['related']['href']
            yield from listing(sub + ('&' if '?' in sub else '?') + 'page%5Bsize%5D=100', prefix + attributes['name'] + '/')
        else:
            yield prefix + attributes['name'], attributes, item['links']['download']


def digest(path, name):
    h = hashlib.new(name)
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('node')
    parser.add_argument('dest')
    parser.add_argument('--skip', nargs='*', default=[], help='substrings of file names to skip')
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args()
    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, attributes, link in listing(API.format(node=args.node) + '?page%5Bsize%5D=100'):
        if any(token in name for token in args.skip):
            print('skip', name)
            continue
        print(name, attributes.get('size'), flush=True)
        entries.append((name, attributes, link))
    if args.list_only:
        return 0
    manifest = []
    for name, attributes, link in entries:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = attributes.get('extra', {}).get('hashes', {})
        if not (target.exists() and expected.get('md5') and digest(target, 'md5') == expected['md5']):
            part = target.with_suffix(target.suffix + '.part')
            for attempt in range(3):
                try:
                    urllib.request.urlretrieve(link, part)
                    break
                except Exception as error:
                    print('retry', name, error, file=sys.stderr)
                    time.sleep(10)
            else:
                print('FAILED', name)
                continue
            part.rename(target)
        md5 = digest(target, 'md5')
        verified = bool(expected.get('md5')) and md5 == expected['md5']
        manifest.append(dict(name=name, size=target.stat().st_size, sha256=digest(target, 'sha256'), md5=md5, osf_md5=expected.get('md5'), md5_verified=verified))
        print('done', name, 'md5_verified' if verified else 'MD5-UNVERIFIED', flush=True)
    (dest / 'download_manifest.json').write_text(json.dumps(dict(source=f'https://osf.io/{args.node}/', files=manifest), indent=1))
    return 0 if all(item['md5_verified'] for item in manifest) and len(manifest) == len(entries) else 1


if __name__ == '__main__':
    raise SystemExit(main())
