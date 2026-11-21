"""E106 coordinate read-only E104/E105 best-validation audits on two GPUs."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from scripts.experiment_records import capture_run_record,file_sha256
from scripts.selected_atlas_audit_checks import write,declared_performance

E104_NAMES=('all_lr005','all_lr002','all_from_e102','network_control')
E105_NAMES=('pep_joint_lr005','pep_joint_lr002','pep_only_from_ctrl','pep_joint_from_ctrl')


def prepare_case(source,inputs,experiment):
    source=source.resolve();s=json.loads((source/'summary.json').read_text())
    if s.get('experiment')!=experiment or s.get('status')!='completed':raise ValueError('Completed '+experiment+' source required')
    records=[]
    for p in source.glob('run_records/*/command.json'):
        record=json.loads(p.read_text());args=record.get('parsed_arguments',{})
        if isinstance(args,dict) and str(Path(args.get('output_dir','')).expanduser().resolve())==str(source) and args.get('experiment_id')==experiment:records.append((p,args))
    if len(records)!=1:raise ValueError('Exactly one original matching command record required')
    command,args=records[0];inputs.mkdir(parents=True,exist_ok=False);evidence=[]
    def copy(name,path,expected=None):
        sha=file_sha256(path)
        if expected is not None and sha!=expected:raise ValueError('Source hash mismatch: '+str(path))
        target=inputs/name;shutil.copy2(path,target)
        if file_sha256(target)!=sha or file_sha256(path)!=sha:raise ValueError('Source changed while copying')
        evidence.extend([dict(path=str(path),sha256=sha),dict(path=str(target),sha256=sha)])
    copy('summary.json',source/'summary.json');copy('command.json',command)
    copy('network_best_val.npz',source/'network_best_val.npz',s['best_val_sha256'])
    copy('parent_network.npz',source/'parent_network.npz',s['parent_sha256'])
    anchor=source/'anchor_network.npz'
    if not anchor.exists():
        if s['anchor_sha256']!=s['parent_sha256']:raise ValueError('Missing distinct anchor copy')
        anchor=source/'parent_network.npz'
    copy('anchor_network.npz',anchor,s['anchor_sha256']);copy('responses.npz',source/'responses.npz')
    for name in ('trainable_masks.npz','fit_policy.json','history.jsonl'):
        if (source/name).exists():copy(name,source/name)
    if experiment=='E105':copy('network_best_val_peptides.npz',source/'network_best_val_peptides.npz',s['best_val_peptides_sha256'])
    record_root=command.parent;provenance={};historical=[];source_checks=[]
    sr=record_root/'source_record.json'
    if sr.exists():
        copy('original_source_record.json',sr);provenance=json.loads(sr.read_text())
        for name,sha in provenance['source_sha256'].items():
            path=record_root/'source'/name
            if file_sha256(path)!=sha:raise ValueError('Historical source archive hash mismatch: '+name)
        # All historical source files are verified; only replay-relevant modules
        # must still match the current forward implementation.
        critical=['r_elegans/brain/conductance_network.py','r_elegans/brain/single_compartment.py','r_elegans/brain/fit_constraints.py','r_elegans/atlas_evaluation.py','r_elegans/data/randi_atlas.py','scripts/run_atlas_baseline.py','scripts/train_atlas_gradient.py',
            'scripts/train_all_parameters.py' if experiment=='E104' else 'scripts/train_atlas_peptides.py']
        if experiment=='E105':critical+=['r_elegans/brain/extrasynaptic.py']
        repo=Path(__file__).resolve().parents[1]
        for name in critical:source_checks.append(dict(file=name,matched=provenance['source_sha256'].get(name)==file_sha256(repo/name)))
    else:historical.append('Original source archive missing; replay-source identity unavailable')
    env=record_root/'environment.json'
    if env.exists():copy('original_environment.json',env)
    else:historical.append('Original environment record missing')
    optimizer_files=[]
    for p in source.glob('optimizer*.msgpack'):
        optimizer_files.append(dict(path=str(p),sha256=file_sha256(p)));evidence.append(optimizer_files[-1])
    historical+=['Historical responses.npz has no recorded checksum in source summary; current checksum and independent replay are recorded',
        'Historical raw NWB hashes not recorded by these trainers; current input hashes/counts/metrics are verified',
        'Saved optimizer bytes do not include a documented exact RNG/settling-state resume; audit does not deserialize optimizer files',
        'No body/gait/sensory start-stop/swimming evidence from atlas fitting']
    history_audit=dict(available=False)
    if (source/'history.jsonl').exists():
        history=[json.loads(l) for l in (source/'history.jsonl').read_text().splitlines() if l.strip()]
        finite=all(np.isfinite(x.get('loss',np.nan)) and x.get('finite',True) for x in history)
        history_audit=dict(available=True,records=len(history),steps_match=len(history)==s['steps'],finite=bool(finite))
    bundle=Path(__file__).resolve().parents[1]/'r_elegans/assets/mac_rhythm_v1_polarized/network.npz'
    contamination=False if s['parent_sha256']==file_sha256(bundle) else None
    parent_summary=Path(s['parent_network']).parent/'summary.json'
    if parent_summary.exists():
        ps=json.loads(parent_summary.read_text())
        contamination=bool(ps.get('experiment') in ('E101','E102'))
    return dict(arguments=args,evidence=evidence,source_experiment=experiment,provenance=dict(
        original_git_revision=provenance.get('git_revision'),original_git_status=provenance.get('git_status'),
        replay_source_checks=source_checks,original_source_archive_verified=bool(provenance),optimizer_files=optimizer_files,history=history_audit),
        historical_limits=historical,validation_leakage=contamination)


def prepare_statistics(data_root,seed,every,out):
    from scripts.run_atlas_baseline import build_statistics
    from r_elegans.assets import load_asset_document
    from r_elegans.atlas_evaluation import split_subjects,pair_statistics
    root=data_root/'raw/functional/randi2023_dandi001075_segmentation';paths=sorted(root.glob('sub-*/*.nwb'))
    if not paths:raise ValueError('Atlas NWB traces missing: '+str(root))
    print('Verifying atlas files:',len(paths),flush=True)
    evidence=[dict(path=str(p.resolve()),sha256=file_sha256(p)) for p in paths]
    ids=list(load_asset_document('runtime_model_v1.json')['neuron_ids'])
    animals,trials,_,test_stats,train_subjects,test_subjects,_,n_test=build_statistics(root,ids,seed)
    fit,val=split_subjects(train_subjects,every=every)
    fit_trials=[t for t in trials if t['subject'] in set(fit)];val_trials=[t for t in trials if t['subject'] in set(val)]
    stats=dict(train=pair_statistics(fit_trials),val=pair_statistics(val_trials),test=test_stats)
    metadata=dict(animals=animals,successful_trials=len(trials),fit_trials=len(fit_trials),validation_trials=len(val_trials),test_trials=n_test,
        fit_animals=len(fit),validation_animals=len(val),test_animals=len(test_subjects),
        fit_pairs=len(stats['train']['pairs']),validation_pairs=len(stats['val']['pairs']),test_pairs=len(test_stats['pairs']))
    for p in evidence:
        if file_sha256(p['path'])!=p['sha256']:raise ValueError('Data changed while rebuilding statistics')
    result=dict(stats=stats,metadata=metadata,targets=sorted({t['target'] for t in trials}),
        subjects=dict(fit=fit,validation=val,test=test_subjects),seed=seed,validation_every_nth=every)
    write(out/'statistics.json',result);write(out/'data_evidence.json',evidence)
    return out/'statistics.json',result


def gpu_environment(gpu):
    env=os.environ.copy();env.update(JAX_PLATFORMS='cuda',CUDA_VISIBLE_DEVICES=str(gpu),JAX_ENABLE_X64='true',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.8',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='4')
    return env


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--e104-sweep',type=Path,required=True);p.add_argument('--e105-sweep',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    out=a.output_dir.resolve();capture_run_record(out,Path(__file__).resolve().parents[1],a)
    from importlib.resources import files
    asset=Path(str(files('r_elegans.assets').joinpath('extrasynaptic_v1.npz')))
    prepared=[];failures=[];statistics={};statistics_errors={};references={};finished=[]
    def persist(complete=False):
        write(out/'summary.json',dict(experiment='E106',audit_complete=complete,preparation_failures=failures,finished=finished,
            all_selected_integrity_and_atlas_numerics_passed=bool(complete and all(r['result']['integrity_and_atlas_numerics_passed'] for r in finished)),
            model_promoted=False,full_start_stop_qualified=False,swimming_qualified=False,
            scope='Audit only; source models untouched, no reselection. Numerical checks cover first13 targets; full coarse responses and atlas metrics replayed.'))
    for exp,root,names in [('E104',a.e104_sweep,E104_NAMES),('E105',a.e105_sweep,E105_NAMES)]:
        for name in names:
            label=exp+'-'+name;source=(root/name).resolve();inputs=out/'inputs'/label
            try:
                m=prepare_case(source,inputs,exp)
                args=m['arguments'];key=(args['seed'],args['validation_every_nth'])
                if key in statistics_errors:raise ValueError('Shared data preparation failed: '+statistics_errors[key])
                if key not in statistics:
                    cache=out/f'statistics-{key[0]}-{key[1]}';cache.mkdir()
                    try:statistics[key]=prepare_statistics(a.data_root.resolve(),*key,cache)
                    except Exception as e:
                        statistics_errors[key]=repr(e);raise
                path,stats=statistics[key];m['statistics']=dict(metadata=stats['metadata'],targets=stats['targets'])
                m['evidence'].append(dict(path=str(path),sha256=file_sha256(path)))
                data_record=path.parent/'data_evidence.json';m['evidence'].append(dict(path=str(data_record),sha256=file_sha256(data_record)))
                m['peptide_asset_sha256']=file_sha256(asset)
                if exp=='E105':m['evidence'].append(dict(path=str(asset),sha256=file_sha256(asset)))
                write(inputs/'manifest.json',m)
                parent_sha=json.loads((inputs/'summary.json').read_text())['parent_sha256']
                reference=out/'references'/parent_sha;references.setdefault(parent_sha,(inputs/'parent_network.npz',reference))
                task=dict(label=label,source=str(source),inputs=str(inputs),manifest=str(inputs/'manifest.json'),manifest_sha256=file_sha256(inputs/'manifest.json'),statistics=str(path),reference=str(reference/'summary.json'))
                job=out/(label+'-job.json');write(job,task);prepared.append((label,job))
            except Exception as e:failures.append(dict(label=label,reason=repr(e)));print('PREPARATION_FAILED',label,repr(e),flush=True)
    persist()
    for parent_sha,(reference,folder) in references.items():
        log=(out/('reference-'+parent_sha[:12]+'.log')).open('w')
        process=subprocess.Popen([sys.executable,'-u','-m','scripts.audit_selected_atlas_worker','--reference-network',str(reference),'--output-dir',str(folder)],env=gpu_environment(0),stdout=log,stderr=subprocess.STDOUT)
        print('Checking parent rhythm:',parent_sha,flush=True)
        while process.poll() is None:time.sleep(1)
        log.close()
        if process.returncode!=0:
            failures.append(dict(label='reference-'+parent_sha,reason='Parent reference worker failed; see reference log'));persist()
    pending=list(prepared);running={};last=time.monotonic()
    while pending or running:
        for gpu in (0,1):
            if gpu not in running and pending:
                label,job=pending.pop(0);log=(out/(label+'.log')).open('w');folder=out/'cases'/label
                proc=subprocess.Popen([sys.executable,'-u','-m','scripts.audit_selected_atlas_worker','--case',str(job),'--output-dir',str(folder)],env=gpu_environment(gpu),stdout=log,stderr=subprocess.STDOUT)
                running[gpu]=(proc,log,label,folder);print('Auditing',label,'on GPU',gpu,flush=True)
        for gpu,(proc,log,label,folder) in list(running.items()):
            if proc.poll() is not None:
                log.close();path=folder/'summary.json';result=json.loads(path.read_text()) if path.exists() else dict(audit_complete=False,problems=['Missing worker summary'])
                finished.append(dict(label=label,exit_code=proc.returncode,report=str(path),result=result));del running[gpu];persist()
                print('Finished',label,'complete=',result['audit_complete'],flush=True)
        if time.monotonic()-last>30:print('Running:',[v[2] for v in running.values()],'pending=',len(pending),flush=True);last=time.monotonic()
        time.sleep(1)
    complete=not failures and len(finished)==8 and all(r['result']['audit_complete'] for r in finished)
    control=next((r['result'].get('selected_metrics',{}).get('test',{}).get('spearman_mean_z') for r in finished if r['label']=='E104-network_control'),None)
    for r in finished:
        if r['result'].get('audit_complete'):r['declared_performance']=declared_performance(r['result'],control)
    persist(complete);return 0 if complete else 2


if __name__=='__main__':raise SystemExit(main())
