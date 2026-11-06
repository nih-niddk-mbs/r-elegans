"""Reproduction records preserve dirty source, dependencies and detectable corruption."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts.experiment_records import capture_run_record, file_sha256
from scripts.archive_motor_run import archive_run, verify_archive


def repository(tmp_path):
    repo=tmp_path/'repo'
    repo.mkdir()
    subprocess.run(['git','init','-q',str(repo)],check=True)
    (repo/'train.py').write_text('original source\n')
    (repo/'.gitignore').write_text('ignored-data\n')
    subprocess.run(['git','add','.'],cwd=repo,check=True)
    subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid',
                    'commit','-qm','source'],cwd=repo,check=True)
    return repo


def test_invocations_preserve_dirty_source_and_never_capture_arbitrary_secrets(tmp_path,monkeypatch):
    repo=repository(tmp_path)
    (repo/'train.py').write_text('modified source\n')
    (repo/'ignored-data').write_text('not source')
    (repo/'private-config.json').write_text('private configuration')
    (repo/'experimental.py').write_text('experimental source\n')
    monkeypatch.setenv('TEST_RUN_SECRET','do-not-record-this-value')
    monkeypatch.setenv('OMP_NUM_THREADS','4')
    first=capture_run_record(tmp_path/'output',repo,dict(seed=3))
    second=capture_run_record(tmp_path/'output',repo,dict(seed=4))
    assert first != second
    assert (first/'source/train.py').read_text()=='modified source\n'
    assert not (first/'source/ignored-data').exists()
    assert not (first/'source/private-config.json').exists()
    assert (first/'source/experimental.py').is_file()
    env=(first/'environment.json').read_text()
    assert 'do-not-record-this-value' not in env
    assert json.loads(env)['numerical_environment']['OMP_NUM_THREADS']=='4'
    assert json.loads((first/'command.json').read_text())['parsed_arguments']['seed']==3


def scientific_runs(tmp_path):
    roots=[]
    for name in ('wave-001','reverse-002'):
        root=tmp_path/name
        stage=root/'seed-0'/('wave' if name.startswith('wave') else 'reverse')
        stage.mkdir(parents=True)
        (root/'summary.json').write_text(json.dumps(dict(workers=[dict(seed=0)])))
        (stage/'actor_best.npz').write_bytes(name.encode())
        roots.append((root,stage))
    original=roots[0][1]/'actor_best.npz'
    (roots[0][1]/'manifest.json').write_text('{}')
    (roots[1][1]/'manifest.json').write_text(json.dumps(dict(initial_actor=str(original),
        initial_actor_sha256=file_sha256(original))))
    return roots


def test_archive_copies_dependency_cohort_and_detects_corruption(tmp_path):
    repo=repository(tmp_path)
    runs=scientific_runs(tmp_path)
    archive=tmp_path/'archive'
    record=archive_run(runs[1][0],archive,repo)
    assert (archive/'COMPLETE').is_file()
    assert len(record['run_mapping'])==2
    assert record['dependencies'][0]['sha256']==file_sha256(runs[0][1]/'actor_best.npz')
    assert record['provenance_gaps']  # Do not pretend historical environments were captured.
    assert (archive/'repository.bundle').is_file()
    verify_archive(archive)
    saved=archive/record['run_mapping'][1]['archived']/'seed-0/wave/actor_best.npz'
    saved.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum mismatch'):
        verify_archive(archive)


def test_missing_original_checkpoint_prevents_false_complete_archive(tmp_path):
    repo=repository(tmp_path)
    runs=scientific_runs(tmp_path)
    (runs[0][1]/'actor_best.npz').unlink()
    archive=tmp_path/'archive'
    with pytest.raises(ValueError,match='Missing checkpoint'):
        archive_run(runs[1][0],archive,repo)
    assert not (archive/'COMPLETE').exists()
