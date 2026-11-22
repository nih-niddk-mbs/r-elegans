"""Synthetic E109/E110/E111 records; no scientific forward run."""
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pytest

from scripts.experiment_records import file_sha256
from test_atlas_target_refinement import fixture_source, BUNDLE


def archive(root,label,critical,version='same',content='same-forward'):
    record=root/label/'run_records'/'fixture';record.mkdir(parents=True)
    files={}
    for name in critical:
        path=record/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
        files[name]=file_sha256(path)
    (record/'source_record.json').write_text(json.dumps(dict(source_sha256=files)))
    (record/'command.json').write_text(json.dumps(dict(parsed_arguments=dict(label=label,root=str(root)))))
    (record/'environment.json').write_text(json.dumps(dict(packages={k:version for k in ('jax','jaxlib','numpy')},
        python_version='same-python',numerical_environment=dict(JAX_ENABLE_X64='true'))))


def worker_outputs(root,manifest,steps):
    for label,model in manifest['models'].items():
        folder=root/label;folder.mkdir()
        with np.load(model['baseline']) as z:baseline=dict(z)
        old_targets=baseline['targets'].tolist();targets=manifest['targets']
        means=baseline['indicator_means'][:,[old_targets.index(t) for t in targets]]
        n=len(targets);nctrl=2*math.ceil(n/8)+(-n)%8;levels=[]
        for i,step in enumerate(steps):
            path=folder/('level-'+str(i)+'.npz')
            np.savez_compressed(path,indicator_means=means,control_means=np.zeros((nctrl,4,302)),
                control_voltage_delta=np.zeros((nctrl,4,12000,2)),reference_sham_rmd_mV=baseline['reference_sham_rmd_mV'],
                control_labels=[f'batch-0-condition-{k}' for k in range(n,10)],targets=targets,
                anchors_sha256=model['anchors_sha256'],neural_step_ms=step,sample_dt_s=.001)
            levels.append(dict(level=i,artifact=str(path),sha256=file_sha256(path)))
        (folder/'summary.json').write_text(json.dumps(dict(checkpoint_sha256=model['sha256'],anchors_sha256=model['anchors_sha256'],
                                                           manifest_sha256=file_sha256(root/'manifest.json'),levels=levels)))


def fixture_sources(tmp_path,monkeypatch):
    import scripts.run_atlas_target_refinement as targeted
    import scripts.audit_atlas_composite as composite
    coarse=tmp_path/'coarse';fixture_source(coarse,monkeypatch,include_extra_pair=True)
    digest=file_sha256(BUNDLE)
    monkeypatch.setattr(composite,'SELECTED_SHA',digest);monkeypatch.setattr(composite,'PARENT_SHA',digest)
    monkeypatch.setattr(composite,'ANCHOR_SHA',targeted.ANCHOR_SHA)
    statpath=coarse/'statistics.json';statistics=json.loads(statpath.read_text());statistics['seed']=0
    for stats in statistics['stats'].values():
        count=len(stats['pairs']);stats.update(n=[3]*count,mean_z=list(range(count)),stouffer=[4.]*count)
    statpath.write_text(json.dumps(statistics))
    mp=coarse/'manifest.json';m=json.loads(mp.read_text());m['experiment']='E109';m['observed_downstream_pairs']=5
    m['evidence_sha256'][str(statpath)]=file_sha256(statpath);mp.write_text(json.dumps(m))
    ap=coarse/'audit.json';a=json.loads(ap.read_text());a['evidence_sha256'][str(statpath)]=file_sha256(statpath)
    for label in ('selected','parent'):
        with np.load(coarse/label/'level-3.npz') as z:data=dict(z)
        for i,step in enumerate((.25,.125,.0625)):
            path=coarse/label/('level-'+str(i)+'.npz');data['neural_step_ms']=np.asarray(step)
            np.savez_compressed(path,**data);a['models'][label]['levels'][i]['artifact_sha256']=file_sha256(path)
        archive(coarse,label,composite.CRITICAL)
    ap.write_text(json.dumps(a))
    mid=tmp_path/'mid';mid.mkdir();mm=targeted.prepare(coarse,mid);worker_outputs(mid,mm,targeted.REPLAY_STEPS)
    assert targeted.audit(mid,True)['integrity_passed']
    (mid/'budget_receipt.json').write_text(json.dumps(dict(all_steps_finished=True)))
    fine=tmp_path/'fine';fine.mkdir();fm=targeted.prepare_continuation(mid,fine);worker_outputs(fine,fm,targeted.PROTOCOLS['E111'])
    assert targeted.audit(fine,True)['integrity_passed']
    (fine/'budget_receipt.json').write_text(json.dumps(dict(all_steps_finished=True)))
    for label in ('selected','parent'):archive(fine,label,composite.CRITICAL)
    output=tmp_path/'output';output.mkdir()
    return coarse,fine,output


def test_composite_complete_coverage_step_maps_and_no_source_mutation(tmp_path,monkeypatch):
    from scripts.audit_atlas_composite import analyze
    coarse,fine,output=fixture_sources(tmp_path,monkeypatch)
    original={p:file_sha256(p) for root in (coarse,fine) for p in root.rglob('*') if p.is_file()}
    result=analyze(coarse,fine,output)
    assert result['audit_complete'] and result['integrity_passed']
    assert result['targets']==6 and result['observed_downstream_pairs']==5
    assert not result['model_promoted'] and not result['simulation_performed'] and not result['uniform_finest_grid_claim']
    for label,model in result['models'].items():
        assert model['bridges_passed'] and model['full_atlas_per_phase_checks_passed']
        assert model['full_atlas_phase_mean_checks_passed']
        for level,row in enumerate(model['levels']):
            with np.load(row['artifact']) as z:
                assert z['response'].shape==(4,6,302)
                if level:
                    with np.load(model['levels'][level-1]['artifact']) as old:
                        np.testing.assert_array_equal(z['neural_step_ms_by_target']*2,old['neural_step_ms_by_target'])
    assert original=={p:file_sha256(p) for root in (coarse,fine) for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('failure',['code','package','missing_archive','changed_artifact','missing_grid','amplitude','statistics'])
def test_composite_refuses_missing_or_inconsistent_evidence(tmp_path,monkeypatch,failure):
    from scripts.audit_atlas_composite import analyze
    coarse,fine,output=fixture_sources(tmp_path,monkeypatch)
    record=fine/'selected'/'run_records'/'fixture'
    if failure=='code':
        path=record/'source'/'r_elegans/brain/common_state_replay.py';path.write_text('different forward')
        sp=record/'source_record.json';s=json.loads(sp.read_text());s['source_sha256']['r_elegans/brain/common_state_replay.py']=file_sha256(path);sp.write_text(json.dumps(s))
    elif failure=='package':
        p=record/'environment.json';d=json.loads(p.read_text());d['packages']['jax']='different';p.write_text(json.dumps(d))
    elif failure=='missing_archive':shutil.rmtree(record)
    elif failure in ('changed_artifact','missing_grid'):
        path=fine/'selected'/'level-2.npz'
        if failure=='missing_grid':path.unlink()
        else:
            with path.open('ab') as stream:stream.write(b'changed')
    elif failure=='amplitude':
        path=fine/'manifest.json';m=json.loads(path.read_text());m['stimulus_nS']*=2;path.write_text(json.dumps(m))
    else:
        m=json.loads((fine/'manifest.json').read_text());Path(m['statistics']).write_text('{}')
    with pytest.raises((ValueError,FileNotFoundError,KeyError)):
        analyze(coarse,fine,output)


def test_assembly_uses_target_names_and_preserves_unaffected_rows():
    from scripts.audit_atlas_composite import assemble_level
    coarse=np.broadcast_to(np.arange(3,dtype=float)[None,:,None],(4,3,2)).copy()
    fine=np.broadcast_to(np.asarray([30.,10.])[None,:,None],(4,2,2)).copy()
    before=coarse.copy()
    response,steps=assemble_level(coarse,fine,['A','B','C'],['C','A'],.125,.0078125)
    np.testing.assert_array_equal(response[:,0],10.)
    np.testing.assert_array_equal(response[:,1],coarse[:,1])
    np.testing.assert_array_equal(response[:,2],30.)
    np.testing.assert_array_equal(steps,[.0078125,.125,.0078125])
    np.testing.assert_array_equal(coarse,before)
    with pytest.raises(ValueError,match='Unique'):
        assemble_level(coarse,fine,['A','B','C'],['A','A'],.125,.0078125)


def test_bridge_checks_downstream_even_when_self_response_dominates():
    from scripts.audit_atlas_composite import compare_bridge
    a=np.ones((4,2,2));a[:,0,1]=1e-6;a[:,1,0]=2e-6
    b=a.copy();b[:,0,1]+=1e-7
    checks=compare_bridge(a,b,['A','B'],[('A','B'),('B','A')],['A','B'])
    assert all(c['passed'] for c in checks['all_modeled_receivers'])
    assert not any(c['passed'] for c in checks['observed_downstream'])
