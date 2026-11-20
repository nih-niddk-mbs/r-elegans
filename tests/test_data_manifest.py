import hashlib
import json
import sys

from scripts import verify_data


def run(monkeypatch, root, manifests, *flags):
    monkeypatch.setattr(sys, 'argv', ['verify_data', '--data-root', str(root), '--manifests', str(manifests), *flags])
    return verify_data.main()


def test_verify_data_passes_for_matching_files_and_fails_for_missing_or_changed_ones(tmp_path, monkeypatch, capsys):
    root, manifests = tmp_path / 'data', tmp_path / 'm'
    (root / 'raw').mkdir(parents=True)
    manifests.mkdir()
    payload = b'abc'
    (root / 'raw/a.txt').write_bytes(payload)
    entry = dict(relative_path='raw/a.txt', bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    missing = dict(relative_path='raw/missing.txt', bytes=1, sha256='0' * 64)
    (manifests / 'x_manifest.json').write_text(json.dumps(dict(sources=[dict(id='s', fetch='run the fetcher', artifacts=[entry])])))
    assert run(monkeypatch, root, manifests) == 0
    (manifests / 'x_manifest.json').write_text(json.dumps(dict(sources=[dict(id='s', fetch='run the fetcher', artifacts=[entry, missing])])))
    assert run(monkeypatch, root, manifests) == 1 and 'get it: run the fetcher' in capsys.readouterr().out
    (manifests / 'x_manifest.json').write_text(json.dumps(dict(sources=[dict(id='s', artifacts=[entry])])))
    (root / 'raw/a.txt').write_bytes(b'abd')  # same size, different content
    assert run(monkeypatch, root, manifests, '--quick') == 0 and run(monkeypatch, root, manifests) == 1
