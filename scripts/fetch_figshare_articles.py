"""Download the files of Figshare articles with md5 verification and a manifest."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def get_json(url):
    return json.loads(subprocess.run(['curl', '-fsSL', '--retry', '5', url], check=True, capture_output=True).stdout)


def fetch(url, target):
    subprocess.run(['curl', '-fsSL', '--retry', '5', '-o', str(target), url], check=True)


def md5_sha256(path):
    a, b = hashlib.md5(), hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            a.update(block)
            b.update(block)
    return a.hexdigest(), b.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dest')
    parser.add_argument('articles', nargs='+', help='Figshare article ids')
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args()
    dest = Path(args.dest).expanduser()
    ok = True
    for article_id in args.articles:
        article = get_json(f'https://api.figshare.com/v2/articles/{article_id}')
        print('==', article_id, article['title'], article.get('doi'), article.get('license', {}).get('name'), flush=True)
        manifest = []
        for f in article['files']:
            print('  ', f['name'], f['size'], flush=True)
            if args.list_only:
                continue
            target = dest / str(article_id) / f['name']
            target.parent.mkdir(parents=True, exist_ok=True)
            if not (target.exists() and md5_sha256(target)[0] == f.get('computed_md5')):
                fetch(f['download_url'], target)
            md5, sha = md5_sha256(target)
            verified = md5 == f.get('computed_md5')
            ok &= verified
            manifest.append(dict(name=f['name'], size=target.stat().st_size, md5=md5, sha256=sha, figshare_md5=f.get('computed_md5'), md5_verified=verified))
            print('   done' if verified else '   MD5-MISMATCH', f['name'], flush=True)
        if not args.list_only:
            (dest / str(article_id) / 'download_manifest.json').write_text(json.dumps(dict(
                source=article['url_public_html'], doi=article.get('doi'), title=article['title'], files=manifest), indent=1))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
