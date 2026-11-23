"""E121 read-only raw atlas inventory and identity-only draft folds, not fitting."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from scripts.experiment_records import file_sha256
from scripts.stable_atlas_worker import atomic_json as write


def draft_folds(subjects, *, seed=1210, count=5):
    ordered=sorted(set(subjects))
    if len(ordered)<count:return dict(status='unavailable_fewer_subjects_than_folds',folds=[],training_ready=False)
    rng=np.random.default_rng(seed);rng.shuffle(ordered)
    groups=[sorted(ordered[i::count]) for i in range(count)]
    folds=[]
    for i,held in enumerate(groups):
        train=sorted(set(ordered)-set(held))
        if set(train)&set(held):raise ValueError('Animal overlap')
        folds.append(dict(fold=i,train_subjects=train,heldout_subjects=held))
    if sorted(s for g in groups for s in g)!=sorted(ordered):raise ValueError('Incomplete heldout coverage')
    return dict(status='identity_only_draft_requires_genotype_and_protocol_review',seed=seed,folds=folds,training_ready=False,
        note='Assignments generated, no cross-validation model trained. Prior dataset exposure remains disclosed.')


def inventory(data_root, output, *, neuron_ids=None, loader=None, trial_builder=None, expected_statistics=None):
    import h5py
    from r_elegans.assets import load_asset_document
    from r_elegans.data.randi_atlas import load_randi_animal
    from r_elegans.atlas_evaluation import animal_trials,pair_statistics,split_subjects
    data_root=Path(data_root).resolve();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    paths=sorted((data_root/'raw/functional/randi2023_dandi001075_segmentation').glob('sub-*/*.nwb'))
    if not paths:raise ValueError('No raw atlas segmentation NWB files: '+str(data_root))
    expected=json.loads(Path(expected_statistics).read_text()) if expected_statistics is not None else None
    seed=expected.get('seed',0) if expected else 0
    every=expected.get('validation_every_nth',4) if expected else 4
    ids=neuron_ids or list(load_asset_document('runtime_model_v1.json')['neuron_ids'])
    loader=loader or load_randi_animal;trial_builder=trial_builder or animal_trials
    rows=[];trials=[];verified_trials=[];hashes={};errors=[];subject_files=defaultdict(list);genotypes=Counter()
    def text(value):return value.decode() if isinstance(value,bytes) else str(value)
    for index,path in enumerate(paths):
        row=dict(path=str(path),bytes=path.stat().st_size)
        try:
            sha=file_sha256(path);hashes[str(path)]=sha;row['sha256']=sha
            with h5py.File(path,'r') as f:
                raw_id=text(f['general/subject/subject_id'][()]).strip() if 'general/subject/subject_id' in f else ''
                genotype=text(f['general/subject/genotype'][()]).strip() if 'general/subject/genotype' in f else ''
            known=bool(raw_id and raw_id.lower() not in ('unknown','none','nan','n/a','na'))
            animal=loader(path,ids);kept=trial_builder(animal,seed=seed)
            trials.extend(kept)
            if known:
                if animal['subject']!=raw_id:raise ValueError('Raw and loader animal IDs differ')
                verified_trials.extend(kept);subject_files[raw_id].append(str(path))
            genotype=genotype or '<missing>';genotypes[genotype]+=1
            row.update(raw_subject_id=raw_id or None,legacy_subject=animal['subject'],identity_verified=known,
                genotype=genotype,raw_stimulations=len(animal['stim_start_s']),successful_trials=len(kept),
                named_unique_neurons=sorted({n for n in animal['names'] if n}),
                named_roi_count=sum(n is not None for n in animal['names']),
                label_status_counts=dict(Counter(animal['status'])),
                accepted_targets=sorted({t['target'] for t in kept}),
                accepted_receivers=sorted({n for t in kept for n in t['z']}),
                exclusion=None if kept else 'no_trials_after_existing_normalization_label_target_response_filters')
            if file_sha256(path)!=sha:raise ValueError('Raw file changed during reading')
        except Exception as exc:
            row['error']=repr(exc);errors.append(dict(path=str(path),error=repr(exc)))
        rows.append(row)
        write(out/'progress.json',dict(files_done=index+1,total_files=len(paths),errors=len(errors)))
        if (index+1)%5==0 or index+1==len(paths):write(out/'files-progress.json',rows)
        print(json.dumps(dict(event='raw_inventory',files_done=index+1,total=len(paths),successful_trials=row.get('successful_trials'),identity_verified=row.get('identity_verified'))),flush=True)
    legacy_subjects=sorted({t['subject'] for t in trials});dev,test=split_subjects(legacy_subjects);fit,val=split_subjects(dev,every=every)
    eligible=sorted({t['subject'] for t in verified_trials});proposed=draft_folds(eligible)
    def summary_stats(selected,pool=trials):
        ts=[t for t in pool if t['subject'] in set(selected)];st=pair_statistics(ts)
        return dict(subjects=len(selected),successful_trials=len(ts),observed_pairs=len(st['pairs']),targets=sorted({t['target'] for t in ts}))
    legacy={name:dict(subject_ids=names,**summary_stats(names)) for name,names in [('train',fit),('validation',val),('test',test)]}
    comparison=None
    if expected is not None:
        mismatches=[]
        for split,subjects in [('train',fit),('val',val),('test',test)]:
            actual=pair_statistics([t for t in trials if t['subject'] in set(subjects)]);cached=expected['stats'][split]
            if [list(p) for p in actual['pairs']]!=[list(p) for p in cached['pairs']]:mismatches.append(split+': pair identities')
            else:
                for key in ('n','mean_z','stouffer'):
                    old=np.asarray(cached[key]);new=np.asarray(actual[key])
                    if old.shape!=new.shape or not np.allclose(new,old,rtol=1e-12,atol=1e-12,equal_nan=False):mismatches.append(split+': '+key)
        if sorted({t['target'] for t in trials})!=expected['targets']:mismatches.append('target identities')
        for key,subjects in [('fit',fit),('validation',val),('test',test)]:
            if expected.get('subjects') and expected['subjects'][key]!=subjects:mismatches.append('subject identities: '+key)
        comparison=dict(passed=not mismatches,mismatches=mismatches,statistics_path=str(Path(expected_statistics).resolve()),
            statistics_sha256=file_sha256(expected_statistics),seed=seed,validation_every_nth=every,relative_tolerance=1e-12,absolute_tolerance=1e-12)
        write(out/'cached-statistics-comparison.json',comparison)
    for fold in proposed['folds']:
        train=summary_stats(fold['train_subjects'],verified_trials);held=summary_stats(fold['heldout_subjects'],verified_trials)
        ta=pair_statistics([t for t in verified_trials if t['subject'] in set(fold['train_subjects'])]);he=pair_statistics([t for t in verified_trials if t['subject'] in set(fold['heldout_subjects'])])
        fold.update(train_coverage=train,heldout_coverage=held,heldout_only_pairs=len(set(he['pairs'])-set(ta['pairs'])),
            heldout_targets_missing_train=sorted(set(held['targets'])-set(train['targets'])))
    write(out/'files.json',rows);write(out/'draft-fivefold.json',proposed);write(out/'legacy-splits.json',legacy)
    by_subject=defaultdict(set)
    for row in rows:
        if row.get('identity_verified'):by_subject[row['raw_subject_id']].add(row['genotype'])
    result=dict(inventory_complete=True,integrity_passed=not errors and (comparison is None or comparison['passed']),cached_statistics_comparison=comparison,errors=errors,raw_nwb_files=len(paths),
        raw_verified_unique_subjects=len(subject_files),legacy_usable_subjects=len(legacy_subjects),verified_usable_subjects=len(eligible),
        successful_trials=len(trials),files_with_no_accepted_trials=sum(r.get('successful_trials')==0 for r in rows),
        files_with_unverified_identity=[r['path'] for r in rows if not r.get('identity_verified') and 'error' not in r],
        repeated_subject_files={k:v for k,v in subject_files.items() if len(v)>1},
        genotype_file_counts=dict(genotypes),subjects_with_conflicting_genotypes={k:sorted(v) for k,v in by_subject.items() if len(v)>1},
        genotype_selection_applied=False,legacy_split_counts={k:v['subjects'] for k,v in legacy.items()},
        draft_fold_count=len(proposed['folds']),cross_validation_fits_executed=0,training_ready=False,
        input_sha256=hashes,output_sha256={str(out/name):file_sha256(out/name) for name in ('files.json','draft-fivefold.json','legacy-splits.json')},
        scope='Raw inventory and draft identities only; no model training, no genotype-approved CV dataset.',
        limits=['NWB file counts are not animal counts.','Filename fallback is recorded only for historical split reconstruction; excluded from proposed identity folds.',
               'No stage-specific rejected-trial causes inferred; existing pipeline reapplied unchanged.','Mixed or missing genotype metadata needs review before fitting.','Identity verified means a consistent metadata ID is present; physical animal/session linkage still depends on metadata.'])
    write(out/'summary.json',result);return result
