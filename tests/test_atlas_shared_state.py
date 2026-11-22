import json
from pathlib import Path

import numpy as np
import pytest

from scripts.atlas_shared_checks import observed_pairs, compare_downstream
from scripts.experiment_records import file_sha256

BUNDLE = Path('r_elegans/assets/mac_rhythm_v1_polarized/network.npz')


def test_measured_support_excludes_self_and_requires_coverage():
    stats = dict(stats=dict(train=dict(pairs=[['A','A'],['A','B'],['B','A']]),test=dict(pairs=[['A','B']])))
    pairs, rows, columns = observed_pairs(stats, ['A','B'], ['A','B'])
    assert pairs == [('A','B'),('B','A')]
    np.testing.assert_array_equal(rows,[0,1]);np.testing.assert_array_equal(columns,[1,0])
    with pytest.raises(ValueError,match='coverage gap'):
        observed_pairs(stats,['A'],['A','B'])


def test_downstream_error_is_not_hidden_by_self_or_large_peer():
    pairs=[('A','B'),('B','A')]
    a=np.array([2e-6,1e-3]);b=np.array([1e-6,1e-3])
    result,ratios,flips=compare_downstream(a,b,pairs)
    assert result['vector']['passed']  # large peer masks local error in global normalization
    assert not result['mixed_tolerance_passed']
    assert ratios[0]>1 and result['worst_pairs'][0]['receiver']=='B'
    result,_,flips=compare_downstream(np.array([2e-6,0.]),np.array([-2e-6,0.]),pairs)
    assert result['active_sign_flips']==1 and flips[0]


def test_tiny_opposite_responses_do_not_replace_vector_failure():
    result,_,_=compare_downstream(np.array([1e-10,2e-10]),np.array([-1e-10,-2e-10]),[('A','B'),('B','A')])
    assert result['mixed_tolerance_passed']
    assert not result['vector']['passed']


def fixture_job(root,monkeypatch):
    import scripts.run_atlas_shared_state as job
    root.mkdir();digest=file_sha256(BUNDLE)
    monkeypatch.setattr(job,'SELECTED_SHA',digest);monkeypatch.setattr(job,'PARENT_SHA',digest)
    with np.load(BUNDLE) as z:ids=z['neuron_ids'].tolist()
    targets=ids[:2];pairs=[[targets[0],targets[1]],[targets[1],targets[0]]]
    stats=dict(targets=targets,seed=0,stats={s:dict(pairs=pairs,n=[3,3],mean_z=[1.,-1.],stouffer=[4.,-4.]) for s in ('train','val','test')})
    statspath=root/'statistics.json';statspath.write_text(json.dumps(stats))
    anchor=root/'anchors.npz';np.savez(anchor,value=np.ones(1));anchor_sha=file_sha256(anchor)
    monkeypatch.setattr(job,'ANCHOR_SHA',dict(selected=anchor_sha,parent=anchor_sha))
    models={label:dict(path=str(BUNDLE),sha256=digest,anchors=str(anchor),anchors_sha256=anchor_sha) for label in ('selected','parent')}
    manifest=dict(models=models,evidence_sha256={str(BUNDLE):digest,str(anchor):anchor_sha,str(statspath):file_sha256(statspath)},targets=targets,statistics=str(statspath),observed_downstream_pairs=2)
    (root/'manifest.json').write_text(json.dumps(manifest));manifest_sha=file_sha256(root/'manifest.json')
    means=np.zeros((4,2,302));means[:,0,1]=2e-6;means[:,1,0]=-1e-6
    t=(np.arange(12000)+1)*.001
    base=np.broadcast_to(np.stack([10*np.sin(2*np.pi*t/.25),10*np.cos(2*np.pi*t/.25)],axis=1),(4,12000,2))
    labels=[f'batch-0-condition-{k}' for k in range(2,10)]
    for label in models:
        folder=root/label;folder.mkdir();levels=[]
        for i,step in enumerate(job.STEPS):
            path=folder/('level-'+str(i)+'.npz')
            np.savez_compressed(path,indicator_means=means,control_means=np.zeros((8,4,302)),
                control_voltage_delta=np.zeros((8,4,12000,2)),control_labels=labels,reference_sham_rmd_mV=base,
                targets=targets,anchors_sha256=anchor_sha,neural_step_ms=step,sample_dt_s=.001)
            levels.append(dict(level=i,artifact=str(path),sha256=file_sha256(path)))
        (folder/'summary.json').write_text(json.dumps(dict(checkpoint_sha256=digest,anchors_sha256=anchor_sha,manifest_sha256=manifest_sha,levels=levels)))
    return root


def test_full_saved_audit_checks_pairs_without_promoting(tmp_path,monkeypatch):
    from scripts.run_atlas_shared_state import audit
    root=fixture_job(tmp_path/'job',monkeypatch)
    result=audit(root,True)
    assert result['integrity_passed'],result['problems']
    assert all(m['per_phase_checks_passed'] and m['phase_mean_checks_passed'] for m in result['models'].values())
    assert not result['model_promoted']
    for label in ('selected','parent'):
        with np.load(root/label/'pair-errors.npz') as z:
            assert z['tolerance_ratios'].shape==(3,5,2)


@pytest.mark.parametrize('failure',['sham','padding','local_error','tamper','timeout','missing'])
def test_negative_cases_never_pass(tmp_path,monkeypatch,failure):
    from scripts.run_atlas_shared_state import audit
    root=fixture_job(tmp_path/'job',monkeypatch)
    path=root/'selected'/'level-3.npz';summary_path=root/'selected'/'summary.json'
    summary=json.loads(summary_path.read_text())
    if failure=='tamper':
        with path.open('ab') as stream:stream.write(b'changed')
    elif failure=='missing':summary['levels'].pop()
    elif failure!='timeout':
        with np.load(path) as z:data=dict(z)
        if failure=='sham':data['control_voltage_delta'][0,0,0,0]=1e-3
        elif failure=='padding':data['control_means'][0,0,0]=1e-3
        else:data['indicator_means'][:,0,1]+=1e-5
        np.savez_compressed(path,**data);summary['levels'][3]['sha256']=file_sha256(path)
    summary_path.write_text(json.dumps(summary))
    result=audit(root,failure!='timeout')
    assert not result['models']['selected'].get('per_phase_checks_passed')
    assert not result['models']['selected'].get('phase_mean_checks_passed')


def test_prepare_recovers_hashed_statistics_lineage(tmp_path,monkeypatch):
    import shutil
    import scripts.run_atlas_shared_state as job
    digest=file_sha256(BUNDLE)
    monkeypatch.setattr(job,'SELECTED_SHA',digest);monkeypatch.setattr(job,'PARENT_SHA',digest)
    e106=tmp_path/'e106';e106.mkdir();e107=tmp_path/'e107';e107.mkdir();e108=tmp_path/'e108';e108.mkdir()
    with np.load(BUNDLE) as z:targets=z['neuron_ids'].tolist()[:2]
    statistics=e106/'statistics.json'
    statistics.write_text(json.dumps(dict(targets=targets,stats=dict(train=dict(pairs=[[targets[0],targets[1]]])))))
    data=e106/'data_evidence.json';data.write_text('[]')
    atlas_manifest=e106/'input-manifest.json'
    atlas_manifest.write_text(json.dumps(dict(evidence=[dict(path=str(p),sha256=file_sha256(p)) for p in (statistics,data)])))
    atlas_job=e106/'E104-network_control-job.json'
    atlas_job.write_text(json.dumps(dict(manifest=str(atlas_manifest),manifest_sha256=file_sha256(atlas_manifest),statistics=str(statistics))))
    (e107/'manifest.json').write_text(json.dumps(dict(source_audit=str(e106))))
    models={};old_models={};anchor_shas={}
    for label in ('selected','parent'):
        folder=e108/label;folder.mkdir();path=folder/'network.npz';shutil.copyfile(BUNDLE,path)
        anchor=folder/'anchors.npz';np.savez(anchor,value=np.ones(1))
        anchor_shas[label]=file_sha256(anchor)
        models[label]=dict(path=str(path),sha256=digest)
        old_models[label]=dict(per_phase_response_checks_passed=True,phase_mean_response_checks_passed=True,
                               anchors_sha256=anchor_shas[label],checkpoint_sha256=digest)
    monkeypatch.setattr(job,'ANCHOR_SHA',anchor_shas)
    old_evidence={str(p):file_sha256(p) for p in (e107/'manifest.json',atlas_job)}
    (e108/'manifest.json').write_text(json.dumps(dict(source_run=str(e107),evidence_sha256=old_evidence,models=models,stimulus_nS=.5)))
    (e108/'audit.json').write_text(json.dumps(dict(audit_complete=True,integrity_passed=True,execution_complete=True,evidence_sha256=old_evidence,models=old_models)))
    (e108/'budget_receipt.json').write_text(json.dumps(dict(all_steps_finished=True)))
    out=tmp_path/'e109';out.mkdir()
    result=job.prepare(e108,out)
    assert result['targets']==targets and result['observed_downstream_pairs']==1
    assert result['targets_without_observed_pairs']==[targets[1]]
    assert file_sha256(result['statistics'])==file_sha256(statistics)
    statistics.write_text('{}')
    bad=tmp_path/'bad';bad.mkdir()
    with pytest.raises(ValueError,match='hash mismatch'):
        job.prepare(e108,bad)


def test_timeout_runs_post_audit_without_qualifying(tmp_path,monkeypatch):
    import shutil
    import scripts.run_atlas_shared_state as job
    import scripts.total_job_budget as budget
    template=fixture_job(tmp_path/'template',monkeypatch)
    def prepare_mock(source,root):
        for p in template.iterdir():
            q=root/p.name
            if p.is_dir():
                shutil.copytree(p,q)
                summary_path=q/'summary.json';summary=json.loads(summary_path.read_text())
                for row in summary['levels']:row['artifact']=str(q/Path(row['artifact']).name)
                summary_path.write_text(json.dumps(summary))
            else:shutil.copy2(p,q)
        # Keep manifest paths pinned to original immutable synthetic inputs.
        return json.loads((root/'manifest.json').read_text())
    monkeypatch.setattr(job,'prepare',prepare_mock);monkeypatch.setattr(job,'capture_run_record',lambda *a,**kw:None)
    def timed_out(command,seconds):
        assert 0<seconds<240
        return 124,True
    monkeypatch.setattr(budget,'run_group',timed_out)
    output=tmp_path/'out'
    monkeypatch.setattr('sys.argv',['job','--source-run',str(tmp_path/'source'),'--output-dir',str(output),'--minutes','5'])
    assert job.main()==124
    result=json.loads((output/'budget_receipt.json').read_text())
    assert result['timed_out'] and result['audit_complete']
    assert not result['all_steps_finished'] and not result['model_promoted']
