import numpy as np
import pytest
from r_elegans.data.paired_observations import (
    chord_bends,source_bends,model_bends,paired_objective,SOURCE_POINTS,SOURCE_STEP_M,
)


def test_geometry_is_invariant_and_detects_missing_chords():
    points=np.array([[0.,0.],[1.,0.],[1.,1.],[2.,1.]])
    expected=[np.pi/2,np.pi/2]
    np.testing.assert_allclose(chord_bends(points),expected)
    reflected=points*np.array([1.,-1.])+[20.,3.]
    np.testing.assert_allclose(chord_bends(reflected),expected)
    points[1]=np.nan
    assert np.isnan(chord_bends(points)).all()


def test_source_and_model_observe_same_constant_curvature():
    step=SOURCE_STEP_M
    # Fine centerline compared against coarse 12-rod approximation: no inversion.
    radius=.001
    tangents=(np.arange(62)+.5)*step/radius
    measured=source_bends(tangents[None,:])
    np.testing.assert_allclose(measured,8*step/radius,atol=1e-10)
    predicted=model_bends(np.full((1,11),(.001/12)/radius),.001/12)
    np.testing.assert_allclose(predicted,measured,atol=.011)
    np.testing.assert_allclose(model_bends(np.zeros((2,11)),.001/12),0.,atol=1e-14)
    with pytest.raises(ValueError,match='beyond'):model_bends(np.zeros((1,11)),1e-5)


def test_joint_objective_gain_invariant_and_penalizes_flat_calcium():
    t=np.linspace(0,2*np.pi,16)
    indicator=np.column_stack((.4+.2*np.sin(t),.5+.1*np.cos(t)))
    observed=10*(.1+indicator)
    bends=np.tile(.2+.1*np.sin(t)[:,None],(1,5))
    score=paired_objective(indicator,bends,observed,bends)
    assert score['total']<1e-20
    flat=paired_objective(np.zeros_like(indicator),bends,observed,bends)
    assert flat['total']>score['total']+1
    observed[2,0]=np.nan
    score=paired_objective(indicator,bends,observed,bends,fluorescence_valid=np.isfinite(observed))
    assert score['total']<1e-20


def test_training_window_refuses_heldout_tampering_and_gaps(tmp_path):
    import json,hashlib
    from r_elegans.brain.paired_fitting import training_windows
    target=tmp_path/'a.npz'
    def save(segments):
        np.savez(target,time_s=np.arange(8.),segment_id=segments,neuron_indices=[0],
                 fluorescence=np.ones((8,1)),fluorescence_valid=np.ones((8,1),dtype=bool),
                 body_angle_absolute=np.zeros((8,62)),head_angle=np.zeros(8))
    save(np.zeros(8))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    split=tmp_path/'splits.json'
    split.write_text(json.dumps({'assignments':{'train':'train','heldout':'test'}}))
    def manifest():
        (tmp_path/'manifest.json').write_text(json.dumps(dict(split_sha256=sha(split),records=[
            dict(uid=uid,fold=fold,status='prepared_not_fit_admitted',targets_file='a.npz',targets_sha256=sha(target))
            for uid,fold in [('train','train'),('heldout','test')]])))
    manifest()
    windows,_=training_windows(tmp_path,['train'])
    assert windows[0]['bends'].shape==(8,5)
    with pytest.raises(ValueError,match='training-fold'):training_windows(tmp_path,['heldout'])
    save(np.r_[np.zeros(4),np.ones(4)])
    with pytest.raises(ValueError,match='checksum'):training_windows(tmp_path,['train'])
    manifest()
    with pytest.raises(ValueError,match='contiguous'):training_windows(tmp_path,['train'])


def test_archive_rejects_traversal_and_links(tmp_path):
    import tarfile,io
    from scripts.fetch_paired_recordings import extract
    for index,name in enumerate(('../outside-data.h5','/outside-data.h5')):
        archive=tmp_path/f'{index}.tar.bz2'
        with tarfile.open(archive,'w:bz2') as tar:
            entry=tarfile.TarInfo(name);entry.size=1
            tar.addfile(entry,io.BytesIO(b'x'))
        with pytest.raises(ValueError,match='Unsafe'):
            extract(archive,tmp_path/f'extracted-{index}')
    archive=tmp_path/'link.tar.bz2'
    with tarfile.open(archive,'w:bz2') as tar:
        entry=tarfile.TarInfo('link-data.h5');entry.type=tarfile.SYMTYPE;entry.linkname='/tmp/no'
        tar.addfile(entry)
    with pytest.raises(ValueError,match='Unsafe'):
        extract(archive,tmp_path/'extracted-link')
