from pathlib import Path
import os,shutil
import pytest
import scripts.archive_results_mbslnx2 as archive


def fixture(tmp_path):
    source=tmp_path/'home'/'results'/'old-run';source.mkdir(parents=True)
    (source/'network.npz').write_bytes(b'important weights')
    (source/'trajectory.npz').write_bytes(b'important raw trajectory')
    destination=tmp_path/'data'/'results'/'old-run';destination.parent.mkdir(parents=True)
    return source,destination


def swap_for_test(a,b):
    # Test-only emulation; production requires Linux atomic renameat2 exchange.
    tmp=a.parent/'.test-swap';a.rename(tmp);b.rename(a);tmp.rename(b)


def test_verified_copy_then_home_duplicate_removed_with_old_paths_working(tmp_path,monkeypatch):
    if not shutil.which('rsync'):pytest.skip('rsync unavailable')
    source,destination=fixture(tmp_path);events=[]
    monkeypatch.setattr(archive,'referenced_runs',lambda root:set())
    monkeypatch.setattr(archive,'atomic_exchange',swap_for_test)
    monkeypatch.setattr(archive,'open_references',lambda p:False)
    monkeypatch.setattr(os,'sync',lambda:None,raising=False)
    size=archive.migrate(source,destination,lambda kind,**k:events.append(kind))
    assert size>0 and source.is_symlink()
    assert (source/'network.npz').read_bytes()==b'important weights'
    assert (destination/'trajectory.npz').read_bytes()==b'important raw trajectory'
    assert not list(source.parent.glob('.archive-link-*'))
    assert events.index('verified')<events.index('path_redirected')<events.index('home_duplicate_deleted')


def test_checksum_failure_never_deletes_source(tmp_path,monkeypatch):
    if not shutil.which('rsync'):pytest.skip('rsync unavailable')
    source,destination=fixture(tmp_path)
    monkeypatch.setattr(archive,'referenced_runs',lambda p:set())
    monkeypatch.setattr(archive,'checksum_differences',lambda a,b:'mismatch')
    with pytest.raises(RuntimeError,match='Checksum verification'):archive.migrate(source,destination,lambda *a,**k:None)
    assert source.is_dir() and not source.is_symlink()
    assert (source/'trajectory.npz').read_bytes()==b'important raw trajectory'


def test_active_writer_or_old_descriptor_keeps_original_copy(tmp_path,monkeypatch):
    if not shutil.which('rsync'):pytest.skip('rsync unavailable')
    source,destination=fixture(tmp_path)
    monkeypatch.setattr(archive,'referenced_runs',lambda p:{source.name})
    with pytest.raises(RuntimeError,match='became active'):archive.migrate(source,destination,lambda *a,**k:None)
    assert not source.is_symlink()
    monkeypatch.setattr(archive,'referenced_runs',lambda p:set())
    monkeypatch.setattr(archive,'atomic_exchange',swap_for_test)
    monkeypatch.setattr(archive,'open_references',lambda p:True)
    monkeypatch.setattr(os,'sync',lambda:None,raising=False)
    with pytest.raises(RuntimeError,match='descriptors'):archive.migrate(source,destination,lambda *a,**k:None)
    assert source.is_symlink() and list(source.parent.glob('.archive-link-*'))
    assert (destination/'network.npz').read_bytes()==b'important weights'


def test_own_process_references_protect_command_and_open_log_runs(tmp_path):
    base=tmp_path/'results';base.mkdir();proc=tmp_path/'proc';proc.mkdir()
    entry=proc/'123';entry.mkdir();(entry/'fd').mkdir()
    (entry/'cmdline').write_bytes((str(base/'active-run')+'\0').encode())
    (entry/'cwd').symlink_to(tmp_path,target_is_directory=True)
    (entry/'fd'/'1').symlink_to(base/'another-run.log')
    assert archive.referenced_runs(base,proc)=={'active-run','another-run'}


def test_symlink_tree_cannot_be_deleted_by_migration(tmp_path):
    source,destination=fixture(tmp_path);(source/'external').symlink_to(tmp_path)
    with pytest.raises(ValueError,match='symlinks'):archive.tree_state(source)
    assert (source/'network.npz').exists()
