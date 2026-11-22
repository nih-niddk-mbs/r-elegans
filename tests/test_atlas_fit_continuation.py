"""Synthetic qualified parent pinning; no scientific server run."""
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.atlas_fit_continuation import qualified_parent, latest_parent, prepare_continuation
from scripts.stable_atlas_policy import Coordinates
from scripts.selected_atlas_audit_checks import saved_arrays
from scripts.stable_atlas_worker import atomic_json as write
from scripts.experiment_records import file_sha256


def fixture_parent(tmp_path):
    from test_atlas_target_refinement import BUNDLE
    root=tmp_path/'measured-atlas-fit-20261122-fixture';root.mkdir()
    origin=root/'origin.npz';origin.write_bytes(BUNDLE.read_bytes());arrays=saved_arrays(origin);codec=Coordinates(arrays)
    folder=root/'checkpoints'/'validation-selected';folder.mkdir(parents=True)
    x=np.full(codec.size,.002);model=folder/'network.npz';np.savez_compressed(model,**codec.apply(x))
    np.savez_compressed(folder/'coordinates.npz',offsets=x)
    write(folder/'identity.json',dict(sha256=file_sha256(model),coordinates_sha256=file_sha256(folder/'coordinates.npz')))
    anchors=root/'anchors.npz';np.savez_compressed(anchors,model_sha256=file_sha256(model),start_voltage_mV=np.zeros((4,302)))
    write(root/'autonomy.json',dict(selected=dict(anchors=str(anchors),anchors_sha256=file_sha256(anchors))))
    stats=root/'statistics.json';write(stats,dict(targets=['ADAL']))
    np.savez_compressed(root/'final-matrices.npz',**{'level_'+str(i):np.zeros((4,1,302)) for i in range(3)})
    steps=(.125,.0625,.03125,.0078125,.00390625,.001953125)
    for i,step in enumerate(steps):
        folder=root/'tasks'/('final-'+str(i));folder.mkdir(parents=True)
        artifact=folder/'response.npz';np.savez_compressed(artifact,zeros=np.zeros(1))
        write(folder/'result.json',dict(artifact=str(artifact),sha256=file_sha256(artifact),rhythms=['synthetic'],
            task=dict(step_ms=step,model_sha256=file_sha256(model),anchors_sha256=file_sha256(anchors))))
    write(root/'manifest.json',dict(experiment='E115',parent=str(origin),parent_sha256=file_sha256(origin),
        evidence_sha256={},statistics=str(stats),rhythm_references={str(step):[] for step in steps}))
    write(root/'summary.json',dict(experiment='E115',validation_complete=True,autonomy_passed=True,
        selection_reproduced=True,fit_qualified_before_audit=True,selected_sha256=file_sha256(model),
        parameters=dict(passed=True,changed_coordinates=codec.size),final_metrics=dict(passed=True)))
    write(root/'audit.json',dict(audit_complete=True,execution_complete=True,integrity_passed=True,
        problems=[],unavailable_evidence=[],evidence_sha256={str(model):file_sha256(model)}))
    write(root/'budget_receipt.json',dict(all_steps_finished=True,validation_complete=True,audit_complete=True,fit_qualified=True,timed_out=False))
    write(root/'selection.json',dict(selected=str(model),sha256=file_sha256(model)))
    return root,x


def test_pin_copy_cumulative_origin_and_no_parent_mutation(tmp_path):
    source,x=fixture_parent(tmp_path);before={str(p):file_sha256(p) for p in source.rglob('*') if p.is_file()}
    assert latest_parent(tmp_path)==source
    out=tmp_path/'new';out.mkdir();m=prepare_continuation(source,out)
    assert m['experiment']=='E116' and m['seed']==1160
    assert m['parent_sha256']==qualified_parent(source)['summary']['selected_sha256']
    np.testing.assert_array_equal(saved_arrays(m['parent_absolute_offsets'])['offsets'],x)
    np.testing.assert_array_equal(saved_arrays(m['absolute_origin'])['params_raw_chemical'],saved_arrays(source/'origin.npz')['params_raw_chemical'])
    assert before=={str(p):file_sha256(p) for p in source.rglob('*') if p.is_file()}


@pytest.mark.parametrize('failure',['hash','audit','budget','unchanged','latest_incomplete','evidence'])
def test_refuse_invalid_parent_without_fallback(tmp_path,failure):
    source,_=fixture_parent(tmp_path)
    if failure=='hash':
        p=source/'selection.json';d=json.loads(p.read_text());d['sha256']='wrong';write(p,d)
    elif failure=='audit':
        p=source/'audit.json';d=json.loads(p.read_text());d['problems']=['bad'];write(p,d)
    elif failure=='budget':
        p=source/'budget_receipt.json';d=json.loads(p.read_text());d['fit_qualified']=False;write(p,d)
    elif failure=='unchanged':
        p=source/'summary.json';d=json.loads(p.read_text());d['parameters']['changed_coordinates']=0;write(p,d)
    elif failure=='latest_incomplete':(tmp_path/'measured-atlas-fit-zz-new-incomplete').mkdir()
    elif failure=='evidence':
        p=source/'audit.json';d=json.loads(p.read_text());d['evidence_sha256'][str(source/'statistics.json')]='wrong';write(p,d)
        out=tmp_path/'new';out.mkdir()
        with pytest.raises(ValueError):prepare_continuation(source,out)
        return
    with pytest.raises((ValueError,FileNotFoundError)):latest_parent(tmp_path)


def test_refuse_cumulative_bounds_even_with_qualifying_flags(tmp_path):
    source,x=fixture_parent(tmp_path);folder=source/'checkpoints/validation-selected'
    np.savez_compressed(folder/'coordinates.npz',offsets=np.full_like(x,.101))
    p=folder/'identity.json';identity=json.loads(p.read_text());identity['coordinates_sha256']=file_sha256(folder/'coordinates.npz');write(p,identity)
    out=tmp_path/'new';out.mkdir()
    with pytest.raises(ValueError,match='cumulative bounds'):prepare_continuation(source,out)
