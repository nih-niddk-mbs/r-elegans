import json,os,time
from pathlib import Path
from scripts.prune_intermediate_trajectories import prune


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def age(base):
    old=time.time()-7200
    for p in base.rglob('*'):
        if not p.is_symlink():os.utime(p,(old,old))


def fitted(base):
    worker=base/'run'/'seed-1';selected=worker/'selected'
    for i in range(4):
        folder=selected/f'iteration-{i:06d}';folder.mkdir(parents=True)
        for name in ('trajectory.npz','network.npz','optimizer_state.npz'):(folder/name).write_bytes(name.encode())
    (worker/'network_best.npz').write_bytes(b'keep best')
    write(worker/'summary.json',dict(status='completed',selected_iteration=2,initial_iteration=0))
    write(worker/'best.json',dict(iteration=2));age(base)
    return worker


def test_only_intermediate_trace_deleted_all_checkpoints_and_endpoints_kept(tmp_path):
    worker=fitted(tmp_path)
    count,size=prune(tmp_path,apply=True)
    assert count==1 and size>0
    assert not (worker/'selected/iteration-000001/trajectory.npz').exists()
    for i in range(4):
        folder=worker/'selected'/f'iteration-{i:06d}'
        assert (folder/'network.npz').exists() and (folder/'optimizer_state.npz').exists()
        if i!=1:assert (folder/'trajectory.npz').exists()


def test_explicit_parent_reference_and_input_capsule_are_protected(tmp_path):
    worker=fitted(tmp_path);trace=worker/'selected/iteration-000001/trajectory.npz'
    write(tmp_path/'other-run/inputs/manifest.json',dict(sources={'parent_training_trace.npz':str(trace)}))
    assert prune(tmp_path,apply=True)[0]==0 and trace.exists()


def test_incomplete_recent_and_unreadable_metadata_are_not_pruned(tmp_path):
    worker=fitted(tmp_path);write(worker/'summary.json',dict(status='running',selected_iteration=2))
    age(tmp_path);assert prune(tmp_path,apply=True)[0]==0
    write(worker/'summary.json',dict(status='completed',selected_iteration=2))
    assert prune(tmp_path,apply=True)[0]==0
    age(tmp_path);(tmp_path/'run/broken.json').write_text('{')
    assert prune(tmp_path,apply=True)[0]==0


def test_committed_cma_only_unselected_arrays_removed(tmp_path):
    root=tmp_path/'cma-run'
    write(root/'summary.json',dict(all_stages_finished=True))
    write(root/'optimizer/g-00000-told/metadata.json',dict(generation=0))
    for i in range(2):
        folder=root/'evaluations'/f'g-00000-c-{i:02d}'
        write(folder/'summary.json',dict(status='completed',output=str(folder)))
        for n in ('no_cue.npz','stop_only.npz','start_only.npz','stop_start.npz','network.npz','coordinates.npz'):(folder/n).write_bytes(n.encode())
    keep=root/'evaluations/g-00000-c-00'
    write(root/'selected/best/reference.json',dict(evaluation=str(keep)))
    age(tmp_path);assert prune(tmp_path,apply=True)[0]==4
    assert (keep/'no_cue.npz').exists()
    other=root/'evaluations/g-00000-c-01';assert not (other/'no_cue.npz').exists()
    assert (other/'network.npz').exists() and (other/'coordinates.npz').exists()


def test_legacy_flat_trace_dry_run_and_selected_iteration(tmp_path):
    worker=tmp_path/'legacy';folder=worker/'selected';folder.mkdir(parents=True)
    for i in range(4):
        (folder/f'trace-{i:05d}.npz').write_bytes(b'raw')
        (folder/f'network-{i:05d}.npz').write_bytes(b'weights')
        (folder/f'state-{i:05d}.npz').write_bytes(b'state')
    write(worker/'summary.json',dict(status='completed',selected_iteration=2));age(tmp_path)
    assert prune(tmp_path,apply=False)[0]==1
    assert len(list(folder.glob('trace-*')))==4
    assert prune(tmp_path,apply=True)[0]==1
    assert (folder/'state-00001.npz').exists()
