import json
import math
from pathlib import Path
import shutil

import numpy as np
import pytest

from scripts.experiment_records import file_sha256
from scripts.atlas_refinement_checks import choose_targets
from scripts.atlas_shared_checks import observed_pairs

BUNDLE=Path('r_elegans/assets/mac_rhythm_v1_polarized/network.npz')


def fixture_source(root,monkeypatch,include_extra_pair=False):
    import scripts.run_atlas_target_refinement as job
    root.mkdir();digest=file_sha256(BUNDLE)
    monkeypatch.setattr(job,'SELECTED_SHA',digest);monkeypatch.setattr(job,'PARENT_SHA',digest)
    with np.load(BUNDLE) as z:ids=z['neuron_ids'].tolist()
    targets=sorted(['ADAL','AIYL','AIYR','RMDL','RMDR','RIVL'])
    pairs=sorted([('ADAL','AIYR'),('AIYR','ADAL'),('RMDL','RMDR'),('RIVL','RMDVL')])
    if include_extra_pair:pairs=sorted(pairs+[('RMDR','RMDL')])
    statistics=dict(targets=targets,stats=dict(train=dict(pairs=pairs)))
    statpath=root/'statistics.json';statpath.write_text(json.dumps(statistics))
    models={};audited={};anchor_shas={}
    t=(np.arange(12000)+1)*.001
    base=np.broadcast_to(np.stack([10*np.sin(2*np.pi*t/.25),10*np.cos(2*np.pi*t/.25)],axis=1),(4,12000,2))
    means=np.zeros((4,len(targets),302))
    for k,(target,receiver) in enumerate(pairs):means[:,targets.index(target),ids.index(receiver)]=(k+1)*1e-6
    for label,failed_target in (('selected','RMDL'),('parent','RIVL')):
        folder=root/label;folder.mkdir()
        model=folder/'network.npz';shutil.copyfile(BUNDLE,model)
        anchor=folder/'anchors.npz';np.savez(anchor,stub=np.zeros(1));anchor_shas[label]=file_sha256(anchor)
        models[label]=dict(path=str(model),sha256=digest,anchors=str(anchor),anchors_sha256=anchor_shas[label])
        ratios=np.zeros((3,5,len(pairs)));flips=np.zeros(ratios.shape,bool)
        pair=next(i for i,p in enumerate(pairs) if p[0]==failed_target)
        ratios[1,0,pair]=2.;ratios[2,4,pair]=3.
        mask=folder/'pair-errors.npz';np.savez_compressed(mask,pairs=np.asarray(pairs),tolerance_ratios=ratios,active_sign_flips=flips)
        comparisons=[]
        for i in range(3):
            checks=[dict(mixed_failures=int((ratios[i,p]>1).sum()),active_sign_flips=int(flips[i,p].sum()),maximum_tolerance_ratio=float(ratios[i,p].max())) for p in range(5)]
            comparisons.append(dict(levels=[i,i+1],per_phase=checks[:4],phase_mean=checks[4]))
        path=folder/'level-3.npz';nctrl=2*math.ceil(len(targets)/8)+(-len(targets))%8
        np.savez_compressed(path,indicator_means=means,control_means=np.zeros((nctrl,4,302)),
            control_voltage_delta=np.zeros((nctrl,4,12000,2)),reference_sham_rmd_mV=base,
            control_labels=[f'batch-0-condition-{k}' for k in range(len(targets),10)],targets=targets,
            anchors_sha256=anchor_shas[label],neural_step_ms=.03125,sample_dt_s=.001)
        levels=[dict(neural_step_ms=s,artifact_sha256=file_sha256(path)) for s in (.25,.125,.0625,.03125)]
        audited[label]=dict(complete=True,comparisons=comparisons,levels=levels,pair_errors_sha256=file_sha256(mask))
    monkeypatch.setattr(job,'ANCHOR_SHA',anchor_shas)
    evidence={str(statpath):file_sha256(statpath)}
    (root/'manifest.json').write_text(json.dumps(dict(targets=targets,statistics=str(statpath),evidence_sha256=evidence,models=models,stimulus_nS=.5)))
    (root/'audit.json').write_text(json.dumps(dict(audit_complete=True,execution_complete=True,integrity_passed=True,models=audited,evidence_sha256=evidence)))
    (root/'budget_receipt.json').write_text(json.dumps(dict(all_steps_finished=True)))
    return statistics,ids


def fixture_job(tmp_path,monkeypatch):
    import scripts.run_atlas_target_refinement as job
    source=tmp_path/'source';fixture_source(source,monkeypatch)
    root=tmp_path/'job';root.mkdir();manifest=job.prepare(source,root)
    for label,model in manifest['models'].items():
        folder=root/label;folder.mkdir()
        with np.load(model['baseline']) as z:baseline=dict(z)
        means=baseline['indicator_means'][:,[manifest['original_targets'].index(t) for t in manifest['targets']]]
        n=len(manifest['targets']);nctrl=2*math.ceil(n/8)+(-n)%8
        levels=[]
        for i,step in enumerate(job.REPLAY_STEPS):
            path=folder/('level-'+str(i)+'.npz')
            np.savez_compressed(path,indicator_means=means,control_means=np.zeros((nctrl,4,302)),
                control_voltage_delta=np.zeros((nctrl,4,12000,2)),reference_sham_rmd_mV=baseline['reference_sham_rmd_mV'],
                control_labels=[f'batch-0-condition-{k}' for k in range(n,10)],targets=manifest['targets'],
                anchors_sha256=model['anchors_sha256'],neural_step_ms=step,sample_dt_s=.001)
            levels.append(dict(level=i,artifact=str(path),sha256=file_sha256(path)))
        (folder/'summary.json').write_text(json.dumps(dict(checkpoint_sha256=model['sha256'],anchors_sha256=model['anchors_sha256'],manifest_sha256=file_sha256(root/'manifest.json'),levels=levels)))
    return root


def test_selection_uses_all_last_two_masks_both_models_and_controls(tmp_path,monkeypatch):
    statistics,ids=fixture_source(tmp_path/'source',monkeypatch)
    source=tmp_path/'source';audit=json.loads((source/'audit.json').read_text())
    masks={label:dict(np.load(source/label/'pair-errors.npz')) for label in ('selected','parent')}
    selected,failures=choose_targets(statistics,ids,masks,audit)
    assert selected==sorted(['ADAL','AIYL','AIYR','RMDL','RIVL'])
    assert 'RMDR' not in selected
    assert failures['selected'][0]['receiver']=='RMDR'
    masks['selected']['pairs']=masks['selected']['pairs'][::-1]
    with pytest.raises(ValueError,match='identities differ'):choose_targets(statistics,ids,masks,audit)


def test_targeted_audit_passes_without_full_atlas_promotion(tmp_path,monkeypatch):
    from scripts.run_atlas_target_refinement import audit
    root=fixture_job(tmp_path,monkeypatch)
    result=audit(root,True)
    assert result['integrity_passed'],result['problems']
    assert all(m['targeted_per_phase_checks_passed'] and m['targeted_phase_mean_checks_passed'] for m in result['models'].values())
    assert all(len(m['comparisons'])==2 and len(m['baseline_reproduction'])==5 for m in result['models'].values())
    assert not result['model_promoted'] and not result['full_atlas_qualified']


@pytest.mark.parametrize('failure',['same_step','fine_error','missing','tamper','timeout','selection'])
def test_failed_controls_or_incomplete_never_qualify(tmp_path,monkeypatch,failure):
    from scripts.run_atlas_target_refinement import audit
    root=fixture_job(tmp_path,monkeypatch)
    summary_path=root/'selected'/'summary.json';summary=json.loads(summary_path.read_text())
    if failure=='selection':
        path=root/'manifest.json';m=json.loads(path.read_text());m['targets'].remove('RIVL');path.write_text(json.dumps(m))
        with pytest.raises(ValueError,match='selection changed'):audit(root,True)
        return
    if failure=='missing':summary['levels'].pop()
    elif failure=='tamper':
        with (root/'selected'/'level-2.npz').open('ab') as stream:stream.write(b'changed')
    elif failure in ('same_step','fine_error'):
        for i in (range(3) if failure=='same_step' else [2]):
            path=root/'selected'/('level-'+str(i)+'.npz')
            with np.load(path) as z:data=dict(z)
            data['indicator_means'][:,0,0]+=1e-3
            # Choose a measured downstream cell, not the direct self response.
            data['indicator_means'][:,0,1]+=1e-3
            with np.load(BUNDLE) as z:ids=z['neuron_ids'].tolist()
            data['indicator_means'][:,0,ids.index('AIYR')]+=1e-3
            np.savez_compressed(path,**data);summary['levels'][i]['sha256']=file_sha256(path)
    summary_path.write_text(json.dumps(summary))
    result=audit(root,failure!='timeout')
    assert not result['models']['selected'].get('targeted_per_phase_checks_passed')
    assert not result['models']['selected'].get('targeted_phase_mean_checks_passed')


def test_supervisor_timeout_always_audits_and_cannot_pass(tmp_path,monkeypatch):
    import scripts.run_atlas_target_refinement as job
    import scripts.total_job_budget as budget
    template=fixture_job(tmp_path,monkeypatch)
    output=tmp_path/'timeout-output'
    def prepare_mock(source,root):
        for p in template.iterdir():
            target=root/p.name
            if p.is_dir():
                shutil.copytree(p,target)
                summary_path=target/'summary.json'
                if summary_path.exists():
                    summary=json.loads(summary_path.read_text())
                    for row in summary['levels']:row['artifact']=str(target/Path(row['artifact']).name)
                    summary_path.write_text(json.dumps(summary))
            else:shutil.copy2(p,target)
        return json.loads((root/'manifest.json').read_text())
    monkeypatch.setattr(job,'prepare',prepare_mock);monkeypatch.setattr(job,'capture_run_record',lambda *a,**kw:None)
    def timeout(command,seconds):
        assert 0<seconds<240
        return 124,True
    monkeypatch.setattr(budget,'run_group',timeout)
    monkeypatch.setattr('sys.argv',['job','--source-run',str(tmp_path/'dummy-source'),'--output-dir',str(output),'--minutes','5'])
    assert job.main()==124
    receipt=json.loads((output/'budget_receipt.json').read_text())
    assert receipt['audit_complete'] and receipt['timed_out']
    assert not receipt['all_steps_finished'] and not receipt['model_promoted']
    result=json.loads((output/'audit.json').read_text())
    assert not any(m.get('targeted_per_phase_checks_passed') for m in result['models'].values())


def continuation_job(tmp_path,monkeypatch):
    import scripts.run_atlas_target_refinement as job
    source=fixture_job(tmp_path,monkeypatch)
    source_audit=job.audit(source,True)
    assert source_audit['integrity_passed']
    (source/'budget_receipt.json').write_text(json.dumps(dict(all_steps_finished=True)))
    root=tmp_path/'continuation';root.mkdir()
    manifest=job.prepare_continuation(source,root)
    for label,model in manifest['models'].items():
        folder=root/label;folder.mkdir()
        with np.load(model['baseline']) as z:baseline=dict(z)
        levels=[]
        for i,step in enumerate(job.PROTOCOLS['E111']):
            path=folder/('level-'+str(i)+'.npz')
            data=dict(baseline);data['neural_step_ms']=np.asarray(step)
            np.savez_compressed(path,**data)
            levels.append(dict(level=i,artifact=str(path),sha256=file_sha256(path)))
        (folder/'summary.json').write_text(json.dumps(dict(checkpoint_sha256=model['sha256'],anchors_sha256=model['anchors_sha256'],manifest_sha256=file_sha256(root/'manifest.json'),levels=levels)))
    return source,root


def test_continuation_preserves_target_union_baseline_and_all_prior_files(tmp_path,monkeypatch):
    import scripts.run_atlas_target_refinement as job
    source,root=continuation_job(tmp_path,monkeypatch)
    before={p:file_sha256(p) for p in source.rglob('*') if p.is_file()}
    result=job.audit(root,True)
    assert result['integrity_passed'],result['problems']
    assert result['experiment']=='E111'
    original=json.loads((source/'manifest.json').read_text())
    assert result['targets']==original['targets']
    assert all(m['targeted_per_phase_checks_passed'] and m['targeted_phase_mean_checks_passed'] for m in result['models'].values())
    assert not result['full_atlas_qualified'] and not result['model_promoted']
    assert before=={p:file_sha256(p) for p in source.rglob('*') if p.is_file()}
    m=result['models']['selected']
    assert [c['steps_ms'] for c in m['comparisons']]==[[.0078125,.00390625],[.00390625,.001953125]]


def test_continuation_rejects_grid_or_subset_changes(tmp_path,monkeypatch):
    import scripts.run_atlas_target_refinement as job
    _,root=continuation_job(tmp_path,monkeypatch)
    path=root/'manifest.json';manifest=json.loads(path.read_text())
    manifest['neural_step_ms'][0]=.03125;path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='protocol/grid changed'):job.audit(root,True)
    manifest['neural_step_ms']=list(job.PROTOCOLS['E111']);manifest['targets'].remove('RIVL')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='selection changed'):job.audit(root,True)


def test_continuation_timeout_cannot_qualify_saved_results(tmp_path,monkeypatch):
    import scripts.run_atlas_target_refinement as job
    _,root=continuation_job(tmp_path,monkeypatch)
    result=job.audit(root,False)
    assert result['integrity_passed']
    assert not any(m['targeted_phase_mean_checks_passed'] for m in result['models'].values())


def test_continuation_rejects_changed_source_baseline_before_replay(tmp_path,monkeypatch):
    import scripts.run_atlas_target_refinement as job
    source=fixture_job(tmp_path,monkeypatch);job.audit(source,True)
    (source/'budget_receipt.json').write_text(json.dumps(dict(all_steps_finished=True)))
    with (source/'selected'/'level-2.npz').open('ab') as stream:stream.write(b'changed')
    root=tmp_path/'bad-continuation';root.mkdir()
    with pytest.raises(ValueError,match='hash mismatch'):job.prepare_continuation(source,root)
