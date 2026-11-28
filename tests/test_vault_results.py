import gzip
import json
import os
from pathlib import Path
import tarfile
import zipfile

import pytest

from scripts import vault_results as vault


@pytest.fixture(autouse=True)
def no_processes(monkeypatch):
    monkeypatch.setattr(vault, 'process_runs', lambda base: set())


def npz(path, *keys, pad=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w') as archive:
        for key in keys:
            archive.writestr(key+'.npy', b'x'*(8+pad))
    return path


def make_tree(tmp_path):
    base = tmp_path/'results'
    repo = tmp_path/'repo'; repo.mkdir()
    npz(base/'run1/actor_best.npz', 'chemical', 'electrical')                 # tier B
    npz(base/'run1/trajectory.npz', 'voltage_mV', 'time_s')                   # simulation
    npz(base/'run1/candidate-001.npz', 'weights', 'offsets')                  # tier C by name
    npz(base/'run2/copy/actor_final.npz', 'chemical', 'electrical')           # duplicate content in another run
    npz(base/'run1/actor_best_copy.npz', 'chemical', 'electrical')            # identical content, same run
    documented = npz(base/'run2/odd-name.npz', 'weights', pad=5)              # tier A via documented hash
    (repo/'notes.md').write_text('selected model sha256 '+vault.digest(documented)+'\n')
    npz(base/'run2/network_huge.npz', 'weights', pad=2**20+10)                # oversize parameter-like
    (base/'run1/progress.jsonl').write_text('{"a": 1}\n'*5)
    (base/'run1/demo.gif').write_bytes(b'GIF89a')
    (base/'run2/audit.json').write_text('{"ok": true}')
    (base/'top.log').write_text('top level')
    (base/'run1/loose.pkl').write_bytes(b'pickle-ish')
    return base, repo


def entries_by_rel(base, repo, cap=2**20):
    entries, summary = vault.classify(base, repo, cap)
    return {e['rel']: e for e in entries}, summary


def test_classification_tiers(tmp_path):
    base, repo = make_tree(tmp_path)
    by, summary = entries_by_rel(base, repo)
    assert by['run1/actor_best.npz']['tier'] == 'B'
    assert by['run1/trajectory.npz']['why'] == 'simulation' and 'sha256' not in by['run1/trajectory.npz']
    assert by['run1/candidate-001.npz']['why'] == 'tier_c'
    assert by['run2/odd-name.npz']['tier'] == 'A'
    assert by['run2/network_huge.npz']['why'] == 'oversize_param_like'
    assert by['run1/demo.gif']['why'] == 'gif'
    assert by['run1/loose.pkl']['tier'] == 'B'
    assert by['run1/progress.jsonl']['kind'] == 'record' and by['top.log']['kind'] == 'record'
    assert summary['documented_full_hashes'] == 1 and summary['documented_full_hashes_found'] == 1


def test_documented_hash_prefix_matches(tmp_path):
    base, repo = make_tree(tmp_path)
    sha = vault.digest(base/'run2/odd-name.npz')
    (repo/'notes.md').write_text('checkpoint '+sha[:12]+'...\n')
    by, summary = entries_by_rel(base, repo)
    assert by['run2/odd-name.npz']['tier'] == 'A'
    assert summary['documented_full_hashes'] == 0


def test_missing_documented_hash_is_reported(tmp_path):
    base, repo = make_tree(tmp_path)
    (repo/'more.md').write_text('lost checkpoint '+'ab'*32+'\n')
    _, summary = entries_by_rel(base, repo)
    assert summary['documented_full_hashes_missing'] == ['ab'*32]


def build(tmp_path, base, repo, name='vault'):
    target = tmp_path/name
    vault.build(base, target, repo, 2**20, allow_same_device=True)
    return target


def test_build_keeps_parameters_and_records_and_dedups(tmp_path):
    base, repo = make_tree(tmp_path)
    target = build(tmp_path, base, repo)
    kept = sorted(p.name for p in (target/'params').iterdir())
    assert len(kept) == 3        # actor (deduped), odd-name, loose.pkl; huge/trajectory/candidate excluded
    index = [json.loads(line) for line in (target/'params_index.jsonl').read_text().splitlines()]
    actor = [row for row in index if len(row['paths']) == 3]
    assert len(actor) == 1 and actor[0]['tier'] == 'B'
    with tarfile.open(target/'records/run1.tar.gz') as tar:
        names = tar.getnames()
    assert names == ['run1/progress.jsonl']
    assert (target/'records/_toplevel.tar.gz').exists()
    assert vault.verify(target)


def test_build_refuses_same_device_nonempty_vault_and_active_runs(tmp_path, monkeypatch):
    base, repo = make_tree(tmp_path)
    with pytest.raises(SystemExit):
        vault.build(base, tmp_path/'v1', repo, 2**20)          # same filesystem without explicit allowance
    target = build(tmp_path, base, repo, 'v2')
    with pytest.raises(SystemExit):
        vault.build(base, target, repo, 2**20, allow_same_device=True)       # not empty
    with pytest.raises(SystemExit):
        vault.build(base, base/'inside', repo, 2**20, allow_same_device=True)   # inside the source
    monkeypatch.setattr(vault, 'process_runs', lambda b: {base/'run1'})
    with pytest.raises(SystemExit):
        vault.build(base, tmp_path/'v3', repo, 2**20, allow_same_device=True)


def test_verify_detects_tampering_and_extras(tmp_path):
    base, repo = make_tree(tmp_path)
    target = build(tmp_path, base, repo)
    victim = next((target/'params').iterdir())
    victim.write_bytes(victim.read_bytes()+b'!')
    with pytest.raises(SystemExit):
        vault.verify(target)
    target2 = build(tmp_path, base, repo, 'vault2')
    (target2/'stray.txt').write_text('extra')
    with pytest.raises(SystemExit):
        vault.verify(target2)


def test_purge_dry_run_confirmation_and_deletion(tmp_path, capsys):
    base, repo = make_tree(tmp_path)
    target = build(tmp_path, base, repo)
    vault.purge(base, target, None, apply=False)
    assert (base/'run1/trajectory.npz').exists()
    assert 'DRY RUN' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        vault.purge(base, target, 'wrong-hash', apply=True)
    assert (base/'run1/trajectory.npz').exists()
    confirm = vault.digest(target/'MANIFEST.jsonl')[:12]
    vault.purge(base, target, confirm, apply=True)
    assert base.is_dir() and list(base.iterdir()) == []
    events = [json.loads(line)['event'] for line in (target/'PURGE_JOURNAL.jsonl').read_text().splitlines()]
    assert events.count('deleted') == 3 and events.count('start') == 3
    assert vault.verify(target)      # journal and verification marker do not break verification


def test_purge_refuses_when_source_changed_after_build(tmp_path):
    base, repo = make_tree(tmp_path)
    target = build(tmp_path, base, repo)
    confirm = vault.digest(target/'MANIFEST.jsonl')[:12]
    (base/'run3').mkdir()
    (base/'run3/new-result.npz').write_bytes(b'new')
    with pytest.raises(SystemExit):
        vault.purge(base, target, confirm, apply=True)
    assert (base/'run3/new-result.npz').exists()
    (base/'run3/new-result.npz').unlink(); (base/'run3').rmdir()
    stamp = (base/'run2/audit.json').stat()
    os.utime(base/'run2/audit.json', ns=(stamp.st_atime_ns, stamp.st_mtime_ns+10**9))
    with pytest.raises(SystemExit):
        vault.purge(base, target, confirm, apply=True)       # modified file


def test_purge_refuses_active_runs_and_unlinks_symlinks_without_following(tmp_path, monkeypatch):
    base, repo = make_tree(tmp_path)
    outside = tmp_path/'elsewhere'; outside.mkdir(); (outside/'keep.txt').write_text('precious')
    (base/'linked').symlink_to(outside)
    target = build(tmp_path, base, repo)
    confirm = vault.digest(target/'MANIFEST.jsonl')[:12]
    monkeypatch.setattr(vault, 'process_runs', lambda b: {base/'run1'})
    with pytest.raises(SystemExit):
        vault.purge(base, target, confirm, apply=True)
    monkeypatch.setattr(vault, 'process_runs', lambda b: set())
    vault.purge(base, target, confirm, apply=True)
    assert (outside/'keep.txt').read_text() == 'precious' and not os.path.lexists(base/'linked')
    inventory = [json.loads(line) for line in gzip.open(target/'inventory.jsonl.gz', 'rt')]
    assert any(row['kind'] == 'symlink' and row['rel'] == 'linked' for row in inventory)
