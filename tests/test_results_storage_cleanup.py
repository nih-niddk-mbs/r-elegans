import json
import os
from pathlib import Path
import time
import numpy as np
import pytest
from scripts import clean_results_storage as clean


def done(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root/'budget_receipt.json'
    path.write_text(json.dumps(dict(all_steps_finished=True, audit_complete=True)))
    os.utime(path, (time.time()-7200,)*2)


def raw(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **(arrays or dict(dense_time_s=np.arange(3), dense_joints=np.zeros((3, 11)))))
    os.utime(path, (time.time()-7200,)*2)


def test_apply_deletes_only_recognized_raw_and_preserves_parameters_and_records(tmp_path, monkeypatch):
    root = tmp_path/'completed'; done(root)
    raw(root/'trajectory.npz')
    raw(root/'network_final.npz', voltage_mV=np.zeros(302))
    raw(root/'optimizer/trajectory.npz')
    raw(root/'run_records/source/trajectory.npz')
    raw(root/'unknown.npz')
    raw(root/'response.npz', response=np.zeros(3), chemical=np.ones(3))
    (root/'log.txt').write_text('detailed operations')
    before = {str(p): clean.digest(p) for p in root.rglob('*') if p.is_file() and p.name != 'trajectory.npz'}
    monkeypatch.setattr(clean, 'process_runs', lambda base: set())
    events = []
    count, size = clean.clean(tmp_path, False, 1, lambda event, **kw: events.append((event, kw)))
    assert count == 1 and (root/'trajectory.npz').exists()
    count, size = clean.clean(tmp_path, True, 1, lambda event, **kw: events.append((event, kw)))
    assert count == 1 and size > 0 and not (root/'trajectory.npz').exists()
    assert all(clean.digest(Path(p)) == h for p, h in before.items())
    deleted = next(row for event, row in events if event == 'deleted')
    assert len(deleted['sha256']) == 64 and deleted['bytes'] > 0
    assert (root/'optimizer/trajectory.npz').exists() and (root/'run_records/source/trajectory.npz').exists()


def test_active_and_recursive_dependencies_are_never_deleted(tmp_path, monkeypatch):
    live = tmp_path/'live'; source = tmp_path/'source'; ancestor = tmp_path/'ancestor'
    for root in (live, source, ancestor):
        done(root); raw(root/'trajectory.npz')
    (live/'manifest.json').write_text(json.dumps(dict(source_run=str(source))))
    (source/'manifest.json').write_text(json.dumps(dict(input_sha256={str(ancestor/'trajectory.npz'): 'hash'})))
    monkeypatch.setattr(clean, 'process_runs', lambda base: {live})
    assert clean.clean(tmp_path, True, 1, lambda *a, **k: None) == (0, 0)
    assert all((r/'trajectory.npz').exists() for r in (live, source, ancestor))


def test_unknown_incomplete_run_protects_its_dependencies(tmp_path, monkeypatch):
    live = tmp_path/'interrupted'; live.mkdir()
    source = tmp_path/'source'; done(source); raw(source/'final.npz', voltage_mV=np.zeros(302))
    (live/'manifest.json').write_text(json.dumps(dict(source_run=str(source))))
    monkeypatch.setattr(clean, 'process_runs', lambda base: set())
    assert clean.clean(tmp_path, True, 1, lambda *a, **k: None) == (0, 0)
    assert (source/'final.npz').exists()


def test_recent_symlink_and_parameter_keys_are_protected(tmp_path, monkeypatch):
    root = tmp_path/'completed'; done(root)
    raw(root/'final.npz', log_density=np.zeros(3), voltage_mV=np.zeros(3))
    target = tmp_path/'outside.npz'; raw(target)
    (root/'trajectory.npz').symlink_to(target)
    raw(root/'active.npz'); os.utime(root/'active.npz', None)
    monkeypatch.setattr(clean, 'process_runs', lambda base: set())
    assert clean.clean(tmp_path, True, 1, lambda *a, **k: None) == (0, 0)
    assert target.exists()


def test_unreadable_incomplete_metadata_refuses_cleanup(tmp_path, monkeypatch):
    live = tmp_path/'live'; live.mkdir(); (live/'manifest.json').write_text('not json')
    root = tmp_path/'completed'; done(root); raw(root/'trajectory.npz')
    monkeypatch.setattr(clean, 'process_runs', lambda base: set())
    with pytest.raises(RuntimeError):
        clean.clean(tmp_path, True, 1, lambda *a, **k: None)
    assert (root/'trajectory.npz').exists()


def test_new_job_appearing_while_hashing_stops_deletion(tmp_path, monkeypatch):
    root = tmp_path/'completed'; done(root); raw(root/'trajectory.npz')
    calls = []
    def processes(base):
        calls.append(1)
        return {tmp_path/'new-job'} if len(calls) >= 5 else set()
    monkeypatch.setattr(clean, 'process_runs', processes)
    events = []
    assert clean.clean(tmp_path, True, 1, lambda event, **kw: events.append(event)) == (0, 0)
    assert (root/'trajectory.npz').exists() and 'stopped_new_job' in events
