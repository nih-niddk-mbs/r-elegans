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


def test_parameter_reference_does_not_protect_unrelated_nested_parent_trajectories(tmp_path, monkeypatch):
    live = tmp_path/'incomplete'; live.mkdir()
    parent = tmp_path/'large-completed'; done(parent)
    raw(parent/'network_best.npz', chemical=np.ones(3))
    raw(parent/'parent_run/warm_start_parent/active-trajectory.npz')
    raw(parent/'active-trajectory.npz')
    (live/'manifest.json').write_text(json.dumps(dict(input_sha256={str(parent/'network_best.npz'): 'hash'})))
    monkeypatch.setattr(clean, 'process_runs', lambda base: set())
    count, _ = clean.clean(tmp_path, True, 1, lambda *a, **k: None)
    assert count == 2
    assert (parent/'network_best.npz').exists()


def test_exact_raw_input_and_its_referenced_json_dependencies_remain(tmp_path, monkeypatch):
    live = tmp_path/'incomplete'; live.mkdir()
    parent = tmp_path/'parent'; done(parent)
    raw(parent/'active-trajectory.npz'); raw(parent/'sham-trajectory.npz')
    raw(parent/'other/active-trajectory.npz')
    (parent/'inputs.json').write_text(json.dumps(dict(raw_input=str(parent/'active-trajectory.npz'))))
    (live/'manifest.json').write_text(json.dumps(dict(input_manifest=str(parent/'inputs.json'), input_sha256={str(parent/'sham-trajectory.npz'): 'hash'})))
    monkeypatch.setattr(clean, 'process_runs', lambda base: set())
    count, _ = clean.clean(tmp_path, True, 1, lambda *a, **k: None)
    assert count == 1
    assert (parent/'active-trajectory.npz').exists() and (parent/'sham-trajectory.npz').exists()


def test_directory_dependency_protects_subtree_not_sibling_copy(tmp_path, monkeypatch):
    live = tmp_path/'incomplete'; live.mkdir()
    parent = tmp_path/'parent'; done(parent)
    raw(parent/'needed/active-trajectory.npz'); raw(parent/'oldcopy/active-trajectory.npz')
    (live/'manifest.json').write_text(json.dumps(dict(source_run=str(parent/'needed'))))
    monkeypatch.setattr(clean, 'process_runs', lambda base: set())
    count, _ = clean.clean(tmp_path, True, 1, lambda *a, **k: None)
    assert count == 1 and (parent/'needed/active-trajectory.npz').exists()


def test_relative_input_paths_are_protected_without_protecting_sibling_arrays(tmp_path, monkeypatch):
    live = tmp_path/'incomplete'; live.mkdir()
    parent = tmp_path/'parent'; done(parent)
    raw(parent/'active-trajectory.npz'); raw(parent/'sham-trajectory.npz')
    (live/'manifest.json').write_text(json.dumps(dict(input='../parent/active-trajectory.npz')))
    monkeypatch.setattr(clean, 'process_runs', lambda base: set())
    count, _ = clean.clean(tmp_path, True, 1, lambda *a, **k: None)
    assert count == 1 and (parent/'active-trajectory.npz').exists()


def test_historical_parent_lineage_validation_copies_are_removed_but_inputs_and_weights_kept(tmp_path, monkeypatch):
    root = tmp_path/'completed'; done(root)
    lineage = root/'parent_run/resume_parent/inputs/parent_lineage'
    raw(lineage/'validation-long.npz', voltage_mV=np.zeros((10, 302)), body_joint_angles=np.zeros((10, 11)))
    raw(lineage/'network_best.npz', chemical=np.ones(3))
    raw(lineage/'source_inputs/parent_actor.npz', weights=np.ones(3))
    raw(root/'inputs/response.npz', response=np.ones(3))
    monkeypatch.setattr(clean, 'process_runs', lambda base: set())
    count, _ = clean.clean(tmp_path, True, 1, lambda *a, **k: None)
    assert count == 1 and not (lineage/'validation-long.npz').exists()
    assert (lineage/'network_best.npz').exists() and (lineage/'source_inputs/parent_actor.npz').exists()
    assert (root/'inputs/response.npz').exists()


def test_output_destination_cannot_turn_parameter_reference_into_whole_run_protection(tmp_path, monkeypatch):
    live = tmp_path/'incomplete'; live.mkdir()
    parent = tmp_path/'completed'; done(parent)
    raw(parent/'network_best.npz', chemical=np.ones(3)); raw(parent/'trajectory.npz')
    (parent/'configuration.json').write_text(json.dumps(dict(arguments=dict(output_dir=str(parent)))))
    (live/'manifest.json').write_text(json.dumps(dict(input_sha256={str(parent/'configuration.json'): 'hash', str(parent/'network_best.npz'): 'hash'})))
    monkeypatch.setattr(clean, 'process_runs', lambda base: set())
    count, _ = clean.clean(tmp_path, True, 1, lambda *a, **k: None)
    assert count == 1 and (parent/'network_best.npz').exists()


def test_finished_saved_data_audit_does_not_seed_live_dependency_protection(tmp_path, monkeypatch):
    parent = tmp_path/'completed'; done(parent); raw(parent/'trajectory.npz')
    audit = tmp_path/'saved-audit'; audit.mkdir()
    path = audit/'audit.json'
    path.write_text(json.dumps(dict(audit_complete=True, integrity_passed=True, run=str(parent))))
    os.utime(path, (time.time()-7200,)*2)
    monkeypatch.setattr(clean, 'process_runs', lambda base: set())
    count, _ = clean.clean(tmp_path, True, 1, lambda *a, **k: None)
    assert count == 1
    # An explicit incomplete computational receipt still takes priority.
    (audit/'budget_receipt.json').write_text(json.dumps(dict(all_steps_finished=False, audit_complete=True)))
    assert not clean.completed(audit)


def test_active_lineage_dependency_stays_protected_despite_new_trace_recognition(tmp_path, monkeypatch):
    live = tmp_path/'active'; live.mkdir()
    parent = tmp_path/'completed'; done(parent)
    trace = parent/'inputs/parent_lineage/validation-long.npz'
    raw(trace, voltage_mV=np.zeros((10, 302)))
    (live/'manifest.json').write_text(json.dumps(dict(input_sha256={str(trace): 'hash'})))
    monkeypatch.setattr(clean, 'process_runs', lambda base: {live})
    assert clean.clean(tmp_path, True, 1, lambda *a, **k: None) == (0, 0)
    assert trace.exists()


def test_legacy_final_assay_without_status_requires_matching_parameter_hash(tmp_path, monkeypatch):
    root = tmp_path/'cue-precision-legacy'; root.mkdir()
    raw(root/'parent_run/network_best.npz', chemical=np.ones(3))
    raw(root/'parent_run/inputs/parent_lineage/validation-long.npz', voltage_mV=np.zeros((10, 302)))
    path = root/'summary.json'
    path.write_text(json.dumps(dict(checkpoint_sha256=clean.digest(root/'parent_run/network_best.npz'),
         all_grids_response_pass=False, cases=[dict(case='coarse')], comparisons=[])))
    os.utime(path, (time.time()-7200,)*2)
    monkeypatch.setattr(clean, 'process_runs', lambda base: set())
    assert clean.completed(root)
    count, _ = clean.clean(tmp_path, True, 1, lambda *a, **k: None)
    assert count == 1 and (root/'parent_run/network_best.npz').exists()
    (root/'parent_run/network_best.npz').write_bytes(b'changed')
    assert not clean.completed(root)


def test_compact_report_is_readonly_and_never_launches_cleanup(tmp_path, monkeypatch, capsys):
    import sys
    monkeypatch.setattr(clean, 'clean', lambda *a, **k: pytest.fail('Report started cleanup'))
    monkeypatch.setattr(sys, 'argv', ['clean', '--base', str(tmp_path), '--report'])
    clean.main()
    assert list(tmp_path.iterdir()) == []
    directory = tmp_path/'.cleanup-records'; directory.mkdir()
    (directory/'storage-test.jsonl').write_text(json.dumps(dict(event='finished', apply=True, files=10, bytes=1024))+'\n')
    clean.main()
    assert '"files": 10' in capsys.readouterr().out


def test_audit_of_incomplete_computation_never_counts_as_completed_run(tmp_path):
    root = tmp_path/'partial'; root.mkdir()
    (root/'audit.json').write_text(json.dumps(dict(audit_complete=True, integrity_passed=True, execution_complete=False)))
    assert not clean.completed(root)
