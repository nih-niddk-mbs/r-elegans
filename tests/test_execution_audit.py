"""E121 source/data inventory audit tests; synthetic data, no fitting."""
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.stable_atlas_worker import atomic_json as write
from scripts.experiment_records import file_sha256


def test_readiness_distinguishes_inactive_implementation_and_plan():
    from scripts.run_execution_audit import local_readiness
    result=local_readiness()
    assert result['audit_complete'] and not result['ready_for_requested_training']
    assert result['flags']['optional_extrasynaptic_module_present']
    assert not result['flags']['active_extrasynaptic_state'] and not result['flags']['active_replay_calls_peptide_step']
    assert len(result['flags']['active_anchor_fields'])==4
    assert {'GAP-CV','GAP-EXTRA','GAP-OPT','RISK-ID','RISK-GENO'} <= {f['id'] for f in result['findings']}
    assert all(len(sha)==64 for sha in result['source_sha256'].values())


def test_draft_folds_reproducible_grouped_and_explicitly_not_training():
    from scripts.atlas_dataset_inventory import draft_folds
    a=draft_folds([str(i) for i in range(87)]);b=draft_folds([str(i) for i in reversed(range(87))])
    assert a==b and not a['training_ready']
    held=[]
    for f in a['folds']:
        assert not set(f['train_subjects'])&set(f['heldout_subjects'])
        assert len(f['heldout_subjects']) in (17,18);held+=f['heldout_subjects']
    assert len(set(held))==87 and len(held)==87
    assert not draft_folds(['one'])['folds']


def test_raw_inventory_counts_files_ids_genotypes_exclusions_and_preserves_inputs(tmp_path):
    import h5py
    from scripts.atlas_dataset_inventory import inventory
    raw=tmp_path/'data/raw/functional/randi2023_dandi001075_segmentation';paths=[]
    # Seven known animals, one additional file for subject0, one unknown-ID file.
    for i,subject in enumerate([str(i) for i in range(7)]+['0',None]):
        path=raw/f'sub-{i}'/'trace.nwb';path.parent.mkdir(parents=True)
        with h5py.File(path,'w') as f:
            g=f.create_group('general/subject')
            if subject is not None:g['subject_id']=subject
            g['genotype']='unc-31' if i==6 else 'WT'
        paths.append(path)
    before={str(p):file_sha256(p) for p in paths}
    def loader(path,ids):
        with h5py.File(path) as f:
            subject=f['general/subject/subject_id'][()].decode() if 'general/subject/subject_id' in f else path.name
        return dict(subject=subject,stim_start_s=np.arange(2),names=['A','B'],status=['named','named'])
    def trials(animal,seed):
        if animal['subject']=='6':return []
        return [dict(subject=animal['subject'],target='A',z={'B':1.}),dict(subject=animal['subject'],target='A',z={'B':2.})]
    out=tmp_path/'audit';r=inventory(tmp_path/'data',out,neuron_ids=['A','B'],loader=loader,trial_builder=trials)
    assert r['integrity_passed'] and r['raw_nwb_files']==9 and r['raw_verified_unique_subjects']==7
    assert r['legacy_usable_subjects']==7 and r['verified_usable_subjects']==6
    assert r['files_with_no_accepted_trials']==1 and len(r['files_with_unverified_identity'])==1
    assert len(r['repeated_subject_files']['0'])==2 and r['genotype_file_counts']=={'WT':8,'unc-31':1}
    assert r['draft_fold_count']==5 and r['cross_validation_fits_executed']==0 and not r['training_ready']
    draft=json.loads((out/'draft-fivefold.json').read_text())
    assert all('trace.nwb' not in f['heldout_subjects']+f['train_subjects'] for f in draft['folds'])
    assert before=={str(p):file_sha256(p) for p in paths}
    from r_elegans.atlas_evaluation import pair_statistics,split_subjects
    all_trials=[t for path in paths for t in trials(loader(path,['A','B']),0)]
    subjects=sorted({t['subject'] for t in all_trials});dev,test=split_subjects(subjects);fit,val=split_subjects(dev,every=4)
    cached={}
    for split,names in [('train',fit),('val',val),('test',test)]:
        st=pair_statistics([t for t in all_trials if t['subject'] in names])
        cached[split]={k:([list(p) for p in v] if k=='pairs' else v.tolist()) for k,v in st.items()}
    cache=tmp_path/'statistics.json';write(cache,dict(stats=cached,targets=['A'],seed=0,validation_every_nth=4))
    repeated=inventory(tmp_path/'data',tmp_path/'compared',neuron_ids=['A','B'],loader=loader,trial_builder=trials,expected_statistics=cache)
    assert repeated['cached_statistics_comparison']['passed'] and repeated['integrity_passed']
    d=json.loads(cache.read_text());d['stats']['train']['mean_z'][0]+=1;write(cache,d)
    changed=inventory(tmp_path/'data',tmp_path/'bad-cache',neuron_ids=['A','B'],loader=loader,trial_builder=trials,expected_statistics=cache)
    assert not changed['integrity_passed'] and 'train: mean_z' in changed['cached_statistics_comparison']['mismatches']


def test_independent_saved_audit_output_does_not_overwrite_original(tmp_path):
    from scripts.run_saved_atlas_candidate import audit
    source=tmp_path/'source';source.mkdir();write(source/'audit.json',dict(original=True))
    digest=file_sha256(source/'audit.json');out=tmp_path/'new-audit'
    result=audit(source,output_dir=out)
    assert result['audit_complete'] and not result['integrity_passed']
    assert file_sha256(source/'audit.json')==digest and (out/'audit.json').exists()
    with pytest.raises(ValueError):audit(source,output_dir=source/'nested')


def test_checkpoint_scope_reports_templates_without_claiming_measured_neurons(tmp_path):
    from scripts.run_execution_audit import checkpoint_scope
    from test_atlas_target_refinement import BUNDLE
    source=tmp_path/'source';source.mkdir();model=tmp_path/'model.npz';model.write_bytes(BUNDLE.read_bytes())
    write(source/'manifest.json',dict(candidate=str(model)))
    digest=file_sha256(model);result=checkpoint_scope(source)
    assert result['neurons']==302 and result['channel_templates']==17
    assert result['checkpoint_sha256']==digest and file_sha256(model)==digest
    assert not result['checkpoint_contains_peptide_fields'] and result['all_effective_conductances_finite_nonnegative']
    assert result['chemical_edges']>0 and 'independently validated' in result['scope']


def test_completed_components_resolve_risk_labels_but_not_missing_implementations():
    from scripts.run_execution_audit import local_readiness,reconcile_readiness
    original=local_readiness()
    components=dict(dataset=dict(integrity_passed=True,raw_nwb_files=110,genotype_file_counts={'WT':110},
        files_with_unverified_identity=[],repeated_subject_files={},subjects_with_conflicting_genotypes={}),
        checkpoint=dict(reported_qualification_confirmed=True,parameter_scope=dict(base_matches_bundled_polarized_reference=True,unique_base_conductance_rows=2,unique_effective_conductance_rows=297)))
    result=reconcile_readiness(original,components);statuses={f['id']:f['status'] for f in result['findings']}
    assert statuses['RISK-ID']=='VERIFIED_METADATA_NO_ID_CONFLICTS'
    assert statuses['RISK-GENO']=='VERIFIED_METADATA_ALL_WT'
    assert statuses['SCOPE-NUMERIC']=='SAVED_DATA_RECHECK_PASSED'
    assert statuses['GAP-CELLPRIORS']=='CONFIRMED_BASE_TEMPLATE_PROVENANCE'
    assert statuses['GAP-CV']=='NOT_IMPLEMENTED_IN_ACTIVE_FITTER' and not result['ready_for_requested_training']
    assert next(f for f in original['findings'] if f['id']=='RISK-ID')['status']=='SERVER_DATA_CHECK_REQUIRED'
    incomplete=reconcile_readiness(original,dict(dataset=dict(integrity_passed=True)))
    assert next(f for f in incomplete['findings'] if f['id']=='RISK-ID')['status']=='SERVER_DATA_CHECK_REQUIRED'
