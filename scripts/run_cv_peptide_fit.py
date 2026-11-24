"""E124 eight-hour TOTAL fivefold initialization fit; paired A100 workers.

Development CV, structured unknown groups, no body/RL/biological qualification.
Final checks use a new prospective60s full-window protocol; E123 labels unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from scripts.experiment_records import capture_run_record,file_sha256
from scripts.run_cv_peptide_pilot import write,read_trials,physical_cpus,REFERENCE_SHA
from scripts.stable_atlas_policy import loss,eligible_targets,Coordinates
from scripts.cv_fit_core import (Evaluator,load_arrays,group_directions,raw_peptide_gains,penalty,
    objective,choose,response_check,prediction_checks,full_window_checks,admit_probe)

REPO=Path(__file__).resolve().parents[1]
BASE=Path('/home/carsonc/r-elegans-results')
MODULE='scripts.run_cv_peptide_fit'


def active_jobs():
    from scripts.cv_validation_job import active_jobs as validation_jobs,ancestor_pids
    excluded=ancestor_pids(os.getpid());found=validation_jobs(exclude_pid=os.getpid())
    for line in subprocess.check_output(['ps','-ww','-u',str(os.getuid()),'-o','pid=,etime=,args='],text=True).splitlines():
        row=line.split(None,2)
        if len(row)!=3:continue
        pid,elapsed,command=row
        if int(pid) in excluded:continue
        try:a=shlex.split(command)
        except ValueError:continue
        if not any(a[i:i+2]==['-m',MODULE] for i in range(len(a)-1)) or any(x in a for x in ('--status','--report','--check')):continue
        folder=a[a.index('--output-dir')+1] if '--output-dir' in a else '<unknown>'
        found.append(dict(pid=int(pid),elapsed=elapsed,output_dir=folder))
    return found


def training_choice(candidate_score,parent_score,candidate_safe):
    """Full-training fine-clock selection; deliberately has no held-out argument."""
    return bool(candidate_safe and np.isfinite(candidate_score) and candidate_score<parent_score-1e-10)


def make_folds(rows):
    from scripts.atlas_dataset_inventory import draft_folds
    from r_elegans.atlas_evaluation import pair_statistics
    subjects=[r['subject'] for r in rows]
    if len(subjects)!=len(set(subjects)):raise ValueError('Repeated animal identity')
    trials=[t for r in rows for t in r['trials']];eligible=sorted({t['subject'] for t in trials})
    if (len(rows),len(eligible),len(trials))!=(110,87,650):raise ValueError('Dataset differs from approved E121/E122 inventory; inspect before fitting')
    assignments=draft_folds(eligible)['folds'];result=[]
    for i,fold in enumerate(assignments):
        if set(fold['train_subjects'])&set(fold['heldout_subjects']):raise ValueError('Animal leakage')
        data=dict(subjects=fold,stats={name:pair_statistics([t for t in trials if t['subject'] in set(names)]) for name,names in
            [('train',fold['train_subjects']),('heldout',fold['heldout_subjects'])]})
        rng=np.random.default_rng(1240+i);pool=eligible_targets(data,'train')
        if len(pool)<16:raise ValueError('Insufficient training-panel coverage')
        data.update(fold=i,seed=1240+i,panel=sorted(rng.choice(pool,16,replace=False).tolist()),target_rng_state=rng.bit_generator.state)
        data['animal_statistics']={s:pair_statistics([t for t in trials if t['subject']==s],min_trials=1) for s in fold['heldout_subjects']}
        result.append(data)
    if sorted(s for f in result for s in f['subjects']['heldout_subjects'])!=eligible:raise ValueError('Held-out coverage not exactly once')
    return result,sorted({t['target'] for t in trials})


def prepare(root,data_root,start_unix):
    paths=sorted((data_root/'raw/functional/randi2023_dandi001075_segmentation').glob('sub-*/*.nwb'))
    if not paths:raise ValueError('No raw NWB files')
    with ProcessPoolExecutor(max_workers=6) as pool:rows=list(pool.map(read_trials,map(str,paths)))
    folds,targets=make_folds(rows);write(root/'folds.json',folds)
    write(root/'raw_inventory.json',[{k:v for k,v in r.items() if k!='trials'} for r in rows])
    parent=REPO/'r_elegans/assets/mac_rhythm_v1_polarized/network.npz'
    if file_sha256(parent)!=REFERENCE_SHA:raise ValueError('Pre-atlas parent identity mismatch')
    shutil.copy2(parent,root/'parent.npz');asset=REPO/'r_elegans/assets/extrasynaptic_v1.npz';shutil.copy2(asset,root/'peptide_asset.npz')
    cpus=physical_cpus()
    if len(cpus)<8:raise ValueError('Eight physical cores required')
    policy=dict(version='E124-initialization-v1',seconds=60,late_window=[30,60],subwindows=[[30,40],[40,50],[50,60]],
        period_relative_tolerance=.01,amplitude_relative_tolerance=.02,waveform_rmse_mV=.5,
        response_absolute_tolerance=1e-7,response_relative_tolerance=.1,loss_grid_tolerance=.02,
        bounds=.1,probe_step=.01,max_probes=20,search_minutes_cap=90,
        scope='Prospective initialization checks, not E123 reclassification,300s validation, exact optimizer resume or behavior qualification')
    write(root/'policy.json',policy)
    critical=['scripts/run_cv_peptide_fit.py','scripts/cv_fit_core.py','scripts/stable_atlas_policy.py',
        'r_elegans/brain/peptide_state_replay.py','r_elegans/brain/peptide_checkpoint.py','r_elegans/brain/conductance_network.py',
        'r_elegans/brain/single_compartment.py','r_elegans/brain/extrasynaptic.py','scripts/analyze_convergence_cycles.py']
    m=dict(experiment='E124',parent_sha256=REFERENCE_SHA,targets=targets,folds=5,batch=16,
        worker_cpus=dict(network=cpus[:4],peptide=cpus[4:8]),worker_deadline_unix=start_unix+450*60,
        total_minutes=480,source_sha256={n:file_sha256(REPO/n) for n in critical},
        input_sha256={str(parent):REFERENCE_SHA,str(root/'parent.npz'):REFERENCE_SHA,str(asset):file_sha256(asset),
            str(root/'peptide_asset.npz'):file_sha256(asset),str(root/'folds.json'):file_sha256(root/'folds.json'),
            str(root/'policy.json'):file_sha256(root/'policy.json'),str(root/'raw_inventory.json'):file_sha256(root/'raw_inventory.json'),
            **{r['path']:r['sha256'] for r in rows}},
        labels='Development CV on historically inspected data, not a pristine test; fixed inherited physiology priors',
        unknown_groups='Four wired/channel directions plus one peptide balance direction; no claim of independent9538-parameter recovery')
    write(root/'manifest.json',m)


def subset(data,all_targets,panel):return data['response'][:,[all_targets.index(n) for n in panel]]


def worker(root,arm):
    m=json.loads((root/'manifest.json').read_text());policy=json.loads((root/'policy.json').read_text())
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,m['worker_cpus'][arm])
    import jax
    if not jax.config.x64_enabled or jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('One visible float64 A100 per worker required')
    for path,sha in m['source_sha256'].items():
        if file_sha256(REPO/path)!=sha:raise ValueError('Runtime source changed')
    folder=root/arm;folder.mkdir();capture_run_record(folder,REPO,dict(arm=arm),phase='E124-persistent-fit-worker')
    engine=Evaluator(root,arm);directions=group_directions(engine.codec);zero=np.zeros(engine.codec.size)
    all_targets=m['targets'];folds=json.loads((root/'folds.json').read_text());baselines=[];baseline_rows=[]
    for level in (0,1):
        write(folder/'stage.json',dict(stage='shared_clean_parent_baseline',level=level))
        data,row=engine.predict(zero,0.,all_targets,level,folder/'baseline'/f'level-{level}')
        if not row['checks']['safe']:raise ValueError('Clean parent safety/control checks failed')
        baselines.append(data);baseline_rows.append(row)
    completed=[]
    for index,fold in enumerate(folds):
        out=folder/f'fold-{index}';out.mkdir();now=time.time()
        allowance=(m['worker_deadline_unix']-now)/(5-index);fold_deadline=now+allowance
        reserve=engine.validation_reserve(len(all_targets))
        search_deadline=min(now+policy['search_minutes_cap']*60,fold_deadline-reserve)
        panel=fold['panel'];x=zero.copy();balance=0.;selected='parent';history=[]
        current=dict(name='parent',safe=True,score=objective(subset(baselines[0],all_targets,panel),panel,engine.ids,fold,x,balance))
        groups=['inward','chemical','outward','gap']+(['peptide_balance'] if arm=='peptide' else [])
        groups=groups[index%len(groups):]+groups[:index%len(groups)]
        group_states={};stop='probe_limit'
        for probe in range(policy['max_probes']):
            pair=2*(engine.anchor_seconds[0]+engine.batch_seconds[0])
            if not admit_probe(time.time(),search_deadline,fold_deadline,pair,reserve):stop='validation_reserve';break
            if shutil.disk_usage(root).free<3*1024**3:stop='disk_reserve';break
            group=groups[probe%len(groups)];proposals=[];before=current['name']
            write(folder/'stage.json',dict(stage='fitting',fold=index,group=group,probe=probe))
            for sign in (-1,1):
                xx=x.copy();bb=balance;name=f'probe-{probe:02d}-{sign:+d}'
                if group=='peptide_balance':bb=float(np.clip(bb+sign*.01,-.1,.1))
                else:xx=np.clip(xx+sign*.01*directions[group],-.1,.1)
                path=out/name
                try:
                    prediction,row=engine.predict(xx,bb,panel,0,path)
                    score=objective(prediction['response'],panel,engine.ids,fold,xx,bb)
                    p=dict(name=name,safe=row['checks']['safe'],score=score)
                except Exception as exc:
                    traceback.print_exc();p=dict(name=name,safe=False,score=None,error=repr(exc));path.mkdir(exist_ok=True);write(path/'error.json',p)
                proposals.append(p);group_states[name]=(xx,bb)
            chosen=choose(current,proposals)
            if chosen['name']!=current['name']:x,balance=group_states[chosen['name']]
            current=chosen;selected=current['name']
            history.append(dict(group=group,before=before,proposals=proposals,selected=selected))
            write(out/'history.json',history)
            np.savez_compressed(out/'optimizer.npz',offsets=x,peptide_balance=np.array(balance),next_probe=np.array(probe+1))
        write(out/'history.json',history)
        np.savez_compressed(out/'optimizer.npz',offsets=x,peptide_balance=np.array(balance),next_probe=np.array(len(history)))
        candidate=[];candidate_rows=[]
        if np.any(x) or balance:
            for level in (0,1):
                write(folder/'stage.json',dict(stage='full_training_selection',fold=index,level=level))
                data,row=engine.predict(x,balance,all_targets,level,out/'candidate'/f'level-{level}');candidate.append(data);candidate_rows.append(row)
        else:candidate,candidate_rows=baselines,baseline_rows
        parent_score=objective(baselines[1]['response'],all_targets,engine.ids,fold,zero,0.)
        candidate_score=objective(candidate[1]['response'],all_targets,engine.ids,fold,x,balance)
        selected_candidate=training_choice(candidate_score,parent_score,all(r['checks']['safe'] for r in candidate_rows))
        candidate_x=x.copy();candidate_balance=balance
        if not selected_candidate:x=zero.copy();balance=0.
        chosen_data=candidate if selected_candidate else baselines;chosen_rows=candidate_rows if selected_candidate else baseline_rows
        write(folder/'stage.json',dict(stage='fine_panel_validation',fold=index))
        quarter,qrow=engine.predict(x,balance,panel,2,out/'validation'/'quarter-panel')
        response_checks=[response_check(chosen_data[0]['response'],chosen_data[1]['response']),
            response_check(subset(chosen_data[1],all_targets,panel),quarter['response'])]
        grid_loss=abs(loss(chosen_data[0]['response'],all_targets,engine.ids,fold,'train')-loss(chosen_data[1]['response'],all_targets,engine.ids,fold,'train'))
        traces=[];rest_rows=[]
        for level in (1,2):
            write(folder/'stage.json',dict(stage='sustained_validation',fold=index,level=level))
            trace,row=engine.rest(x,balance,level,out/'validation'/f'rest-{level}');traces.append(trace);rest_rows.append(row)
        windows=full_window_checks(*traces)
        # Actual chosen endpoint gets its own complete explicit peptide checkpoint.
        engine.save_complete_model(x,balance,qrow['path'],out/'selected-model.npz')
        np.savez_compressed(out/'selected-coordinates.npz',offsets=x,peptide_balance=np.array(balance))
        checks=dict(parameters=True,short_controls=all(r['checks']['safe'] for r in chosen_rows+[qrow]),
            response_convergence=all(r['passed'] for r in response_checks),training_loss_grid_stable=grid_loss<=.02,
            sustained=all(r['finite'] and r['all_cell_voltage_bounded'] and all(c['passed'] for c in r['rhythms']) for r in rest_rows),
            full_windows=all(r['passed'] for r in windows))
        # Held-out labels are first read for scoring after all selection is final.
        heldout=loss(chosen_data[1]['response'],all_targets,engine.ids,fold,'heldout')
        animal_scores=[]
        for subject,statistics in fold['animal_statistics'].items():
            try:value=loss(chosen_data[1]['response'],all_targets,engine.ids,dict(stats=dict(animal=statistics)),'animal');error=None
            except ValueError as exc:value=None;error=str(exc)
            animal_scores.append(dict(subject=subject,loss=value,unavailable_reason=error))
        result=dict(fold=index,completed=True,optimization_probes=len(history),stop_reason=stop,selected_candidate=selected_candidate,
            selected_panel_point=selected,candidate_full_train_score=candidate_score,parent_full_train_score=parent_score,
            selected_full_train_score=candidate_score if selected_candidate else parent_score,
            heldout_loss=heldout,parent_heldout_loss=loss(baselines[1]['response'],all_targets,engine.ids,fold,'heldout'),
            animal_scores=animal_scores,response_checks=response_checks,training_grid_loss_difference=grid_loss,
            rest=rest_rows,full_window_checks=windows,gates=checks,development_checks_passed=all(checks.values()),
            changed_coordinates=int(np.count_nonzero(x)),peptide_balance=balance,candidate_peptide_balance=candidate_balance,
            candidate_changed_coordinates=int(np.count_nonzero(candidate_x)),
            prediction_paths=[r['path'] for r in chosen_rows],quarter_path=qrow['path'],
            model_sha256=file_sha256(out/'selected-model.npz'),policy_version=policy['version'],
            scope='Development initialization only; two full-target clocks plus third clock on16training targets;60s prospective stability, not300s or biological/behavior qualification')
        write(out/'summary.json',result);completed.append(result)
        write(folder/'summary.json',dict(arm=arm,completed_folds=len(completed),folds=completed))
    write(folder/'stage.json',dict(stage='complete',folds=5))


def coordinate(root):
    m=json.loads((root/'manifest.json').read_text());processes=[];logs=[]
    try:
        for gpu,arm in enumerate(('network','peptide')):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',JAX_ENABLE_X64='true',XLA_PYTHON_CLIENT_PREALLOCATE='false')
            log=(root/(arm+'.log')).open('w');logs.append(log)
            processes.append(subprocess.Popen([sys.executable,'-u','-m',MODULE,'--output-dir',str(root),'--worker',arm],env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            stages={}
            for arm in ('network','peptide'):
                path=root/arm/'stage.json';stages[arm]=json.loads(path.read_text()) if path.exists() else 'starting'
            print(json.dumps(dict(stages=stages,minutes_left=max(0,(m['worker_deadline_unix']-time.time())/60))),flush=True)
            try:
                r=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                with (root/'gpu_telemetry.jsonl').open('a') as f:f.write(json.dumps(dict(time_unix=time.time(),returncode=r.returncode,rows=r.stdout.splitlines()))+'\n')
            except (OSError,subprocess.TimeoutExpired):pass
            time.sleep(15)
        write(root/'worker_exit_codes.json',{arm:p.returncode for arm,p in zip(('network','peptide'),processes)})
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for f in logs:f.close()


def audit(root):
    """Saved-data audit runs even when a worker failed or exhausted its budget."""
    problems=[];unavailable=[];arms={};m_path=root/'manifest.json'
    if not m_path.exists():
        result=dict(audit_complete=True,execution_complete=False,integrity_passed=False,problems=['Preparation incomplete'],unavailable=['All fitting/validation evidence'])
        write(root/'audit.json',result);return result
    m=json.loads(m_path.read_text());folds=json.loads((root/'folds.json').read_text());origin=load_arrays(root/'parent.npz');codec=Coordinates(origin);ids=origin['neuron_ids'].tolist();asset=load_arrays(root/'peptide_asset.npz')
    for path,sha in m['input_sha256'].items():
        if not Path(path).is_file() or file_sha256(path)!=sha:problems.append('Input hash '+path)
    for path,sha in m['source_sha256'].items():
        if file_sha256(REPO/path)!=sha:problems.append('Runtime source '+path)
    prediction_cache={}
    for path in root.glob('**/metrics.json'):
        row=json.loads(path.read_text())
        if 'sha256' not in row:continue
        out=path.parent
        for name,sha in row['sha256'].items():
            if file_sha256(name)!=sha:problems.append('Prediction hash '+name)
        coords=load_arrays(out/'coordinates.npz');x=coords['offsets'];balance=float(coords['peptide_balance'])
        if not codec.audit(load_arrays(out/'network.npz'),x)['passed']:problems.append('Parameter arithmetic '+str(out))
        if abs(balance)>.1+1e-12:problems.append('Peptide bounds '+str(out))
        data=load_arrays(out/'prediction.npz');checks=prediction_checks(data,row['level'])
        if checks!=row['checks']:problems.append('Control/rhythm arithmetic '+str(out))
        prediction_cache[str(out)]=(data,row,coords)
    for arm in ('network','peptide'):
        parent=[]
        for level in (0,1):
            path=str(root/arm/'baseline'/f'level-{level}')
            if path in prediction_cache:parent.append(prediction_cache[path][0])
        results=[]
        if len(parent)!=2:
            unavailable.append(arm+' shared baseline incomplete');arms[arm]=[];continue
        for i,fold in enumerate(folds):
            out=root/arm/f'fold-{i}';path=out/'summary.json'
            if not path.exists():unavailable.append(f'{arm} fold{i} incomplete');continue
            s=json.loads(path.read_text());history=json.loads((out/'history.json').read_text());panel=fold['panel']
            current=dict(name='parent',safe=True,score=objective(subset(parent[0],m['targets'],panel),panel,ids,fold,np.zeros(codec.size),0.))
            for h in history:
                candidates=[]
                for r in h['proposals']:
                    p=str(out/r['name'])
                    if p not in prediction_cache:
                        if not r.get('error'):problems.append('Missing proposal '+p)
                        continue
                    data,row,c=prediction_cache[p]
                    if r.get('error'):
                        if r.get('safe') or r.get('score') is not None:problems.append('Failed proposal eligible '+p)
                        continue
                    score=objective(data['response'],panel,ids,fold,c['offsets'],float(c['peptide_balance']))
                    if abs(score-r['score'])>1e-12 or r['safe']!=row['checks']['safe']:problems.append('Proposal score '+p)
                    candidates.append(dict(name=r['name'],score=score,safe=row['checks']['safe']))
                current=choose(current,candidates)
                if current['name']!=h['selected']:problems.append('Train-only probe selection '+str(out))
            candidate_paths=[str(out/'candidate'/f'level-{j}') for j in (0,1)]
            candidate=[prediction_cache[p] for p in candidate_paths] if all(p in prediction_cache for p in candidate_paths) else None
            c=load_arrays(out/'optimizer.npz');candidate_score=objective((candidate[1][0] if candidate else parent[1])['response'],m['targets'],ids,fold,c['offsets'],float(c['peptide_balance']))
            parent_score=objective(parent[1]['response'],m['targets'],ids,fold,np.zeros(codec.size),0.)
            selected=training_choice(candidate_score,parent_score,all(r[1]['checks']['safe'] for r in candidate) if candidate else True)
            if selected!=s['selected_candidate'] or abs(candidate_score-s['candidate_full_train_score'])>1e-12:problems.append('Full-training selection '+str(out))
            chosen=[prediction_cache[p][0] for p in s['prediction_paths']];quarter=prediction_cache[s['quarter_path']][0]
            checks=[response_check(chosen[0]['response'],chosen[1]['response']),response_check(subset(chosen[1],m['targets'],panel),quarter['response'])]
            if checks!=s['response_checks']:problems.append('Response-grid arithmetic '+str(out))
            held=loss(chosen[1]['response'],m['targets'],ids,fold,'heldout')
            if abs(held-s['heldout_loss'])>1e-12:problems.append('Held-out arithmetic '+str(out))
            coords=load_arrays(out/'selected-coordinates.npz');selected_x=coords['offsets'];balance=float(coords['peptide_balance'])
            expected_x=c['offsets'] if selected else np.zeros(codec.size)
            expected_balance=float(c['peptide_balance']) if selected else 0.
            if not np.array_equal(selected_x,expected_x) or balance!=expected_balance:problems.append('Selected coordinates '+str(out))
            for prediction_path in s['prediction_paths']+[s['quarter_path']]:
                recorded=prediction_cache[prediction_path][2]
                if not np.array_equal(recorded['offsets'],selected_x) or float(recorded['peptide_balance'])!=balance:problems.append('Selected prediction coordinates '+prediction_path)
            from r_elegans.brain.peptide_checkpoint import validate
            model=load_arrays(out/'selected-model.npz');validate(model)
            schema1={k:model[k] for k in json.loads(str(model['network_keys_json']))};schema1['schema_version']=model['network_schema_version']
            if not codec.audit(schema1,selected_x)['passed'] or bool(model['extrasynaptic_enabled'])!=(arm=='peptide'):problems.append('Full model/frozen fields '+str(out))
            raw_exc,raw_inh=raw_peptide_gains(balance)
            if not np.all(model['peptide_raw_exc']==raw_exc) or not np.all(model['peptide_raw_inh']==raw_inh):problems.append('Full peptide model '+str(out))
            if (float(model['peptide_log_tau_s'])!=float(np.log(5.)) or
                not np.array_equal(model['peptide_sender'],asset['sender']) or
                not np.array_equal(model['peptide_receiver'],asset['receiver']) or
                str(model['peptide_asset_sha256'])!=file_sha256(root/'peptide_asset.npz')):problems.append('Fixed peptide fields '+str(out))
            if file_sha256(out/'selected-model.npz')!=s['model_sha256']:problems.append('Selected model hash '+str(out))
            for name in ('voltage_mV','gates','calcium_uM','indicator','peptide'):
                if not np.array_equal(model['start_'+name],quarter['start_'+name]):problems.append('Selected full state '+str(out)+' '+name)
            for row in s['animal_scores']:
                try:value=loss(chosen[1]['response'],m['targets'],ids,dict(stats=dict(animal=fold['animal_statistics'][row['subject']])),'animal')
                except ValueError:value=None
                if value!=row['loss']:problems.append('Per-animal score '+str(out)+' '+row['subject'])
            from r_elegans.brain.rhythm_qualification import rhythm_metrics
            traces=[];rests=[]
            for level in (1,2):
                path=out/'validation'/f'rest-{level}';data=load_arrays(path/'trace.npz');end=load_arrays(path/'final.npz');traces.append(data['rmd'])
                row=dict(level=level,dt=.001/2**level,seconds=60.,rhythms=[rhythm_metrics(data['rmd'][:,j],.001/2**level) for j in range(2)],finite=all(np.isfinite(a).all() for a in list(data.values())+list(end.values())),all_cell_voltage_bounded=bool(data['voltage_min'].min()>=-120 and data['voltage_max'].max()<=80));rests.append(row)
            windows=full_window_checks(*traces)
            if rests!=s['rest'] or windows!=s['full_window_checks']:problems.append('Full-window stability arithmetic '+str(out))
            grid_loss=abs(loss(chosen[0]['response'],m['targets'],ids,fold,'train')-loss(chosen[1]['response'],m['targets'],ids,fold,'train'))
            gates=dict(parameters=True,short_controls=all(prediction_cache[p][1]['checks']['safe'] for p in s['prediction_paths']+[s['quarter_path']]),response_convergence=all(r['passed'] for r in checks),training_loss_grid_stable=grid_loss<=.02,sustained=all(r['finite'] and r['all_cell_voltage_bounded'] and all(c['passed'] for c in r['rhythms']) for r in rests),full_windows=all(r['passed'] for r in windows))
            if gates!=s['gates'] or all(gates.values())!=s['development_checks_passed']:problems.append('Gate arithmetic '+str(out))
            results.append({k:s[k] for k in ('fold','optimization_probes','selected_candidate','changed_coordinates','peptide_balance','heldout_loss','parent_heldout_loss','development_checks_passed','model_sha256')})
        arms[arm]=results
    for record in root.glob('**/run_records/*/source_record.json'):
        source=json.loads(record.read_text())
        for name,sha in source['source_sha256'].items():
            if file_sha256(record.parent/'source'/name)!=sha:problems.append('Archived source '+name)
    # Include failed/interrupted proposal weights and every completed artifact.
    for network in root.glob('**/network.npz'):
        if 'run_records' in network.parts:continue
        coordinates=network.with_name('coordinates.npz')
        if coordinates.exists():
            try:
                if not codec.audit(load_arrays(network),load_arrays(coordinates)['offsets'])['passed']:problems.append('Retained proposal parameters '+str(network))
            except (ValueError,OSError) as exc:unavailable.append('Incomplete retained proposal '+str(network)+': '+str(exc))
    artifact_hashes={str(p):file_sha256(p) for p in root.rglob('*.npz')}
    result=dict(audit_complete=True,execution_complete=all(len(v)==5 for v in arms.values()),integrity_passed=not problems,problems=problems,arms=arms,
        development_checks_passed=bool(all(len(v)==5 for v in arms.values()) and all(r['development_checks_passed'] for v in arms.values() for r in v)),
        artifact_sha256=artifact_hashes,
        unavailable=unavailable+['No300s or full-atlas third-clock qualification','No optimizer/reboot recovery test','Structured groups, not9538 independently identified parameters','No full channel-current attribution or complete neural/sensory/body propagation assay; retained control RMD voltages, downstream calcium response and peptide endpoint conductance only','No body/swimming/sensory start-stop or biological qualification'])
    write(root/'audit.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path);p.add_argument('--data-root',type=Path,default=Path('/home/carsonc/r-elegans-data'))
    p.add_argument('--worker',choices=('network','peptide'));p.add_argument('--coordinate',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--status',action='store_true');p.add_argument('--check',action='store_true');p.add_argument('--report',action='store_true')
    a=p.parse_args()
    if a.status or a.check:
        jobs=active_jobs();print('READ-ONLY CHECK; no job started. Active jobs:',json.dumps(jobs))
        if a.status:
            for path in sorted(BASE.glob('cv-peptide-fit-*/budget_receipt.json'),reverse=True)[:3]:print(str(path.parent),json.loads(path.read_text()))
        return 3 if a.check and jobs else 0
    if not a.output_dir:p.error('--output-dir required')
    root=a.output_dir.resolve()
    if a.report:
        print('READ-ONLY REPORT:',root)
        for name in ('audit.json','budget_receipt.json'):
            path=root/name;r=json.loads(path.read_text()) if path.exists() else {};print(name,{k:v for k,v in r.items() if k!='artifact_sha256'})
        return
    if a.worker:return worker(root,a.worker)
    if a.coordinate:return coordinate(root)
    if a.execute:
        job=json.loads((root/'job.json').read_text())
        capture_run_record(root,REPO,job,phase='E124-total-eight-hours')
        prepare(root,Path(job['data_root']),job['start_unix'])
        return coordinate(root)
    if active_jobs():p.error('Active validation/fitting exists; nothing started')
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'.cv-peptide-validation.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('Another GPU job holds the lock; nothing started')
        if active_jobs():p.error('Active GPU job detected after locking; nothing started')
        if root==REPO or REPO in root.parents or root in REPO.parents:p.error('External output required')
        root.mkdir(parents=True,exist_ok=False);started=time.monotonic();start_unix=time.time()
        write(root/'job.json',dict(data_root=str(a.data_root.resolve()),start_unix=start_unix,parent_argv=sys.argv,total_minutes=480))
        write(root/'budget_receipt.json',dict(status='running',total_minutes=480,all_steps_finished=False,audit_complete=False,training_qualified=False))
        code=1;timed_out=False
        try:
            if shutil.disk_usage(root).free<5*1024**3:raise ValueError('At least5GiB free required')
            from scripts.total_job_budget import run_group
            code,timed_out=run_group([sys.executable,'-u','-m',MODULE,'--output-dir',str(root),'--execute'],max(.01,455*60-(time.monotonic()-started)))
        except Exception as exc:
            traceback.print_exc();write(root/'failure.json',dict(error=repr(exc)))
        try:result=audit(root)
        except Exception as exc:
            traceback.print_exc();result=dict(audit_complete=False,execution_complete=False,integrity_passed=False,problems=[repr(exc)]);write(root/'audit.json',result)
        receipt=dict(status='finished',exit_code=code,timed_out=timed_out,total_minutes=480,elapsed_seconds=time.monotonic()-started,
            all_steps_finished=bool(code==0 and result['execution_complete'] and result['audit_complete']),audit_complete=result['audit_complete'],
            development_checks_passed=bool(result.get('development_checks_passed') and result['integrity_passed']),behavior_qualified=False)
        write(root/'budget_receipt.json',receipt);print(json.dumps(receipt),flush=True)
        return 0 if receipt['all_steps_finished'] and result['integrity_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
