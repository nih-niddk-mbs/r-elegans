"""E117: bounded two-hour, eight-target/two-direction own-state fitting pilot."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from scripts.run_stable_atlas_fit import Pool, task, replay_tasks, analyze_results, REPO
from scripts.atlas_fit_continuation import prepare_continuation, latest_parent
from scripts.stable_atlas_policy import Coordinates, eligible_targets, prior, adam_step, loss, preserved
from scripts.stable_atlas_worker import atomic_json as write
from scripts.selected_atlas_audit_checks import saved_arrays
from scripts.experiment_records import capture_run_record, file_sha256


def training_panel(statistics, hard, rng):
    pool=eligible_targets(statistics,'train');fine=sorted(set(pool)&set(hard));coarse=sorted(set(pool)-set(hard))
    if not fine or len(coarse)<7:raise ValueError('Eight-target stratified train support unavailable')
    return sorted(rng.choice(fine,1,replace=False).tolist()+rng.choice(coarse,7,replace=False).tolist())


def expensive_first(specs):
    # Same fixed BATCH8/physics; only dispatch order changes. Work scales with
    # number of target batches and inverse neural step.
    return sorted(specs,key=lambda s: -((len(s.get('targets',[]))+7)//8)/s.get('step_ms',1))


def analyze_panel(root,manifest,statistics,ids,panel,phase):
    rows=[json.loads(p.read_text()) for p in (root/'tasks').glob(phase+'-*/result.json')]
    return analyze_results(rows,manifest,statistics,ids,panel,[1,2],'train')


def decision(metrics):
    return bool(all(metrics[k] for k in ('parent_replay_passed','gradient_screen_passed','training_improved',
        'validation_improved','full_numerical_passed','parameters_passed','parameters_changed','autonomy_passed',
        'state_reproduced','panel_reproduced')))


def coordinator(root):
    deadline=json.loads((root/'deadline.json').read_text())['coordinator_deadline']
    source=Path(json.loads((root/'deadline.json').read_text())['parent_run'])
    m=prepare_continuation(source,root);m.update(experiment='E117',seed=1170,own_state=True,
        pilot=dict(targets=8,directions=2,learning_rate=.0005,one_proposal=True,
            final_numerics='Full atlas once; independently repeat fresh-state finest panel, not full loss'))
    m['continuation']['seed']=1170
    m['policy'].update(lr=.0005,minibatch=8,directions=2,total_minutes=120,training_cutoff_minutes=110,
        selection='Single own-state full-atlas candidate evaluation plus independent fresh-state/panel replay')
    write(root/'manifest.json',m)
    arrays=saved_arrays(m['parent']);codec=Coordinates(arrays);ids=arrays['neuron_ids'].tolist()
    stats=json.loads(Path(m['statistics']).read_text());rng=np.random.default_rng(1170)
    panel=training_panel(stats,m['hard'],rng);directions=rng.choice([-1.,1.],(2,codec.size))
    absolute=saved_arrays(m['parent_absolute_offsets'])['offsets'];lower=np.maximum(-.1,-.1-absolute);upper=np.minimum(.1,.1-absolute)
    trials=root/'trials';trials.mkdir();values={};models={'current':m['parent']}
    def save(name,x):
        if np.max(abs(absolute+x))>.1+1e-12:raise ValueError('Cumulative bound exceeded')
        folder=trials/name;folder.mkdir();path=folder/'network.npz';np.savez_compressed(path,**codec.apply(x))
        np.savez_compressed(folder/'coordinates.npz',offsets=x);values[name]=x;models[name]=str(path);return path
    for i in range(2):
        save(f'd{i}-plus',np.clip(.005*directions[i],lower,upper))
        save(f'd{i}-minus',np.clip(-.005*directions[i],lower,upper))
    np.savez_compressed(root/'directions.npz',directions=directions)
    write(root/'rng.json',rng.bit_generator.state);write(root/'panel.json',dict(targets=panel,seed=1170,labels='train only'))
    pool=Pool(root)
    try:
        # All models get their own full states; current parent is reproduced too.
        specs=[]
        for name,model in models.items():
            spec=task(root,'anchors-'+name,model);spec['kind']='anchors';specs.append(spec)
        rows=pool.tasks(specs,deadline);anchors={r['task']['name'].removeprefix('anchors-'):r['anchors'] for r in rows}
        def replays(name,levels=(1,2),names=None,prefix=None,state=None):
            specs=replay_tasks(root,m,prefix or ('panel-'+name),models[name],state or anchors[name],names or panel,list(levels))
            for spec in specs:spec['require_own_state']=True
            return expensive_first(specs)
        specs=[s for name in models for s in replays(name)]
        pool.tasks(expensive_first(specs),deadline)
        metrics={};matrices={}
        for name in models:metrics[name],matrices[name]=analyze_panel(root,m,stats,ids,panel,'panel-'+name)
        parent_errors=[]
        for level in (1,2):
            old=saved_arrays(m['baselines'][level])['response'][:,[m['targets'].index(t) for t in panel]]
            parent_errors.append(float(abs(matrices['current'][level]-old).max()))
        gradients=[]
        for i in range(2):
            a,b=f'd{i}-plus',f'd{i}-minus'
            gradients.append(((metrics[a]['data_loss']+prior(values[a]))-(metrics[b]['data_loss']+prior(values[b])))/.01*directions[i])
        averaged=np.mean(gradients,axis=0)
        gradient_pass=all(r['passed'] for r in metrics.values()) and max(parent_errors)<=1e-10
        used=averaged if gradient_pass else np.zeros_like(averaged)
        z=np.zeros(codec.size);proposal,mom,var=adam_step(z,z,z,used,1,learning_rate=.0005);proposal=np.clip(proposal,lower,upper)
        save('proposal',proposal)
        np.savez_compressed(root/'optimizer.npz',per_direction_gradients=gradients,averaged_gradient=averaged,
            used_gradient=used,first_moment=mom,second_moment=var,proposal=proposal,iteration=1)
        spec=task(root,'anchors-proposal',models['proposal']);spec['kind']='anchors'
        row=pool.tasks([spec],deadline)[0];anchors['proposal']=row['anchors']
        pool.tasks(replays('proposal'),deadline)
        proposed,pmat=analyze_panel(root,m,stats,ids,panel,'panel-proposal')
        full=pool.tasks(replays('proposal',(0,1,2),m['targets'],'candidate-final'),deadline)
        final,fullmat=analyze_results(full,m,stats,ids,m['targets'],[0,1,2],'val',final=True,reference=False)
        np.savez_compressed(root/'candidate-full-matrices.npz',**{'level_'+str(i):a for i,a in fullmat.items()})
        # Independently regenerate the candidate's state and long neural dynamics.
        row=pool.tasks([task(root,'autonomy-proposal',models['proposal'])],deadline)[0]
        write(root/'candidate-autonomy.json',row)
        oldauto=json.loads((source/'autonomy.json').read_text())['selected']
        cached,fresh=saved_arrays(anchors['proposal']),saved_arrays(row['anchors'])
        state_errors={k:float(abs(cached['start_'+k]-fresh['start_'+k]).max()) for k in ('voltage_mV','gates','calcium_uM','indicator')}
        repeat=pool.tasks(replays('proposal',(2,),prefix='reproduced-panel',state=row['anchors']),deadline)
        _,repmat=analyze_results(repeat,m,stats,ids,panel,[2],'train',reference=False)
        panel_error=float(abs(repmat[2]-fullmat[2][:,[m['targets'].index(t) for t in panel]]).max())
        baseline=saved_arrays(m['baselines'][2])['response'];parent_val=loss(baseline,m['targets'],ids,stats,'val')
        gates=dict(parent_replay_passed=max(parent_errors)<=1e-10,gradient_screen_passed=gradient_pass,
            training_improved=bool(proposed['passed'] and proposed['data_loss']+prior(proposal)<metrics['current']['data_loss']),
            validation_improved=bool(final['data_loss']+prior(proposal)<parent_val),
            full_numerical_passed=final['passed'],parameters_passed=codec.audit(saved_arrays(models['proposal']),proposal)['passed'],
            parameters_changed=bool(np.count_nonzero(proposal)),
            autonomy_passed=all(preserved(oldauto['rhythms'][l][j],row['rhythms'][l][j]) for l in range(2) for j in range(2)),
            state_reproduced=max(state_errors.values())<=1e-12,panel_reproduced=panel_error<=1e-10)
        chosen=decision(gates)
        # Selection is frozen before test-score calculation.
        write(root/'selection.json',dict(selected=models['proposal'] if chosen else m['parent'],candidate_selected=chosen,
            selected_sha256=file_sha256(models['proposal'] if chosen else m['parent']),gates=gates,
            parent_validation_loss=parent_val,candidate_validation_loss=final['data_loss']))
        scores={split:loss(fullmat[2],m['targets'],ids,stats,split) for split in ('train','val','test')}
        result=dict(experiment='E117',execution_complete=True,validation_complete=True,iterations=1,
            accepted_updates=int(gates['training_improved']),fit_qualified_before_audit=chosen,
            selected_sha256=file_sha256(models['proposal'] if chosen else m['parent']),candidate_scores=scores,
            candidate_parameters=codec.audit(saved_arrays(models['proposal']),proposal),gates=gates,
            final_metrics=final,panel_metrics=metrics,proposal_panel=proposed,parent_panel_errors=parent_errors,
            state_errors=state_errors,reproduced_panel_error=panel_error,parent_validation_loss=parent_val,
            independent_full_loss_reproduction=False,model_promoted=False,biological_fidelity_qualified=False,
            sensory_start_stop_qualified=False,swimming_qualified=False,
            limits=['Full atlas numerical validation executed once, not independently repeated.',
                'Fresh-state and finest eight-target panel replay independently repeated.',
                'Parent autonomy/atlas evidence reused with pinned hashes and unchanged forward runtime identity.',
                'Test set historically inspected; descriptive, not untouched holdout.'])
        write(root/'summary.json',result)
    finally:pool.close()


def audit(root):
    """Recompute every panel/selection/optimizer decision; hash complete artifacts."""
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    problems=[];missing=[];evidence={};report={}
    def pin(path,expected=None):
        path=Path(path);sha=file_sha256(path);evidence[str(path)]=sha
        if expected is not None and sha!=expected:raise ValueError('Artifact hash mismatch: '+str(path))
        return path
    try:
        m=json.loads(pin(root/'manifest.json').read_text())
        for path,sha in m['evidence_sha256'].items():pin(path,sha)
        stats=json.loads(pin(m['statistics']).read_text());codec=Coordinates(saved_arrays(pin(m['parent'],m['parent_sha256'])))
        ids=saved_arrays(m['parent'])['neuron_ids'].tolist();panel=json.loads(pin(root/'panel.json').read_text())['targets']
        checkrng=np.random.default_rng(1170)
        if panel!=training_panel(stats,m['hard'],checkrng):raise ValueError('Training panel selection differs')
        directions=saved_arrays(pin(root/'directions.npz'))['directions']
        if not np.array_equal(directions,checkrng.choice([-1.,1.],(2,codec.size))):raise ValueError('Direction RNG differs')
        if json.loads(pin(root/'rng.json').read_text())!=checkrng.bit_generator.state:raise ValueError('RNG state differs')
        absolute=saved_arrays(pin(m['parent_absolute_offsets'],m['parent_absolute_offsets_sha256']))['offsets']
        values={};parameter_checks={}
        for folder in (root/'trials').iterdir():
            x=saved_arrays(pin(folder/'coordinates.npz'))['offsets'];values[folder.name]=x
            checks=codec.audit(saved_arrays(pin(folder/'network.npz')),x)
            if not checks['passed'] or np.max(abs(absolute+x))>.1+1e-12:raise ValueError('Parameter/frozen/cumulative bound failure')
            parameter_checks[folder.name]=checks
        tasks={}
        for path in (root/'tasks').glob('*/task.json'):
            spec=json.loads(pin(path).read_text());pin(spec['model'],spec['model_sha256'])
            if not Path(spec['result']).exists():missing.append(spec['name']);continue
            row=json.loads(pin(spec['result']).read_text());tasks[spec['name']]=row
            if row['status']!='completed' or row['task']!=spec:raise ValueError('Failed/task identity mismatch')
            if spec['kind'] in ('autonomy','anchors'):
                a=saved_arrays(pin(row['anchors'],row['anchors_sha256']))
                if str(a['model_sha256'])!=spec['model_sha256']:raise ValueError('Anchor provenance differs')
                if spec['kind']=='autonomy':
                    metrics=[]
                    for step in (.03125,.015625):
                        a=saved_arrays(pin(Path(spec['output'])/f'rhythm-{step}.npz'))
                        metrics.append([rhythm_metrics(a['rmd_mV'][:,j],.002) for j in range(2)])
                    if metrics!=row['rhythms']:raise ValueError('Autonomy metrics differ')
                continue
            pin(spec['anchors'],spec['anchors_sha256']);a=saved_arrays(pin(row['artifact'],row['sha256']))
            if str(saved_arrays(spec['anchors'])['model_sha256'])!=spec['model_sha256']:raise ValueError('Replay own-state mismatch')
            n=len(spec['targets']);labels=[f'batch-{begin//8}-condition-{k}' for begin in range(0,n,8) for k in range(min(8,n-begin),10)]
            if a['targets'].tolist()!=spec['targets'] or a['neuron_ids'].tolist()!=ids or a['control_labels'].tolist()!=labels:
                raise ValueError('Replay target/control coverage differs')
            if float(a['neural_step_ms'])!=spec['step_ms'] or str(a['anchors_sha256'])!=spec['anchors_sha256']:raise ValueError('Replay metadata differs')
            for key in ('response','control_means','reference_sham_rmd_mV'):
                if a[key].dtype!=np.float64 or not np.isfinite(a[key]).all():raise ValueError('Nonfinite/nonfloat64 signal')
            cm=float(abs(a['control_means']-a['control_means'][0]).max());cv=float(a['maximum_control_voltage_delta_mV'])
            rm=[[rhythm_metrics(a['reference_sham_rmd_mV'][p,:,j],.001) for j in range(2)] for p in range(4)]
            if cm!=row['duplicate_mean_error'] or cv!=row['duplicate_trace_error'] or row['controls_passed']!=(cm<=1e-12 and cv<=1e-12) or rm!=row['rhythms']:
                raise ValueError('Controls/rhythm score differs')
        if not (root/'summary.json').exists():missing.append('Final summary unavailable')
        else:
            s=json.loads(pin(root/'summary.json').read_text());metrics={};matrices={}
            for name in ('current','d0-plus','d0-minus','d1-plus','d1-minus','proposal'):
                metrics[name],matrices[name]=analyze_panel(root,m,stats,ids,panel,'panel-'+name)
            for name in s['panel_metrics']:
                if metrics[name]!=s['panel_metrics'][name]:raise ValueError('Panel score differs')
            if metrics['proposal']!=s['proposal_panel']:raise ValueError('Proposal panel differs')
            final_rows=[r for name,r in tasks.items() if name.startswith('candidate-final-')]
            final,full=analyze_results(final_rows,m,stats,ids,m['targets'],[0,1,2],'val',final=True,reference=False)
            if final!=s['final_metrics']:raise ValueError('Full numerical/validation gates differ')
            optim=saved_arrays(pin(root/'optimizer.npz'));grads=[]
            for i in range(2):
                plus,minus=f'd{i}-plus',f'd{i}-minus'
                expected_plus=np.clip(.005*directions[i],np.maximum(-.1,-.1-absolute),np.minimum(.1,.1-absolute))
                expected_minus=np.clip(-.005*directions[i],np.maximum(-.1,-.1-absolute),np.minimum(.1,.1-absolute))
                if not np.array_equal(values[plus],expected_plus) or not np.array_equal(values[minus],expected_minus):raise ValueError('Perturbation differs')
                grads.append(((metrics[plus]['data_loss']+prior(values[plus]))-(metrics[minus]['data_loss']+prior(values[minus])))/.01*directions[i])
            parent_errors=[float(abs(matrices['current'][l]-saved_arrays(m['baselines'][l])['response'][:,[m['targets'].index(t) for t in panel]]).max()) for l in (1,2)]
            screen=all(metrics[name]['passed'] for name in metrics if name!='proposal') and max(parent_errors)<=1e-10
            averaged=np.mean(grads,axis=0);used=averaged if screen else np.zeros_like(averaged);z=np.zeros(codec.size)
            x,mom,var=adam_step(z,z,z,used,1,learning_rate=.0005);x=np.clip(x,np.maximum(-.1,-.1-absolute),np.minimum(.1,.1-absolute))
            for key,a in dict(per_direction_gradients=np.asarray(grads),averaged_gradient=averaged,used_gradient=used,first_moment=mom,second_moment=var,proposal=x).items():
                if not np.array_equal(optim[key],a):raise ValueError('Optimizer arithmetic differs: '+key)
            if not np.array_equal(x,values['proposal']):raise ValueError('Proposed coordinates differ')
            auto=tasks['autonomy-proposal'];oldauto=json.loads((Path(m['source'])/'autonomy.json').read_text())['selected']
            a=saved_arrays(tasks['anchors-proposal']['anchors']);b=saved_arrays(auto['anchors'])
            state_errors={k:float(abs(a['start_'+k]-b['start_'+k]).max()) for k in ('voltage_mV','gates','calcium_uM','indicator')}
            repeated=[r for name,r in tasks.items() if name.startswith('reproduced-panel-')]
            _,rep=analyze_results(repeated,m,stats,ids,panel,[2],'train',reference=False)
            panel_error=float(abs(rep[2]-full[2][:,[m['targets'].index(t) for t in panel]]).max())
            parent_val=loss(saved_arrays(m['baselines'][2])['response'],m['targets'],ids,stats,'val')
            gates=dict(parent_replay_passed=max(parent_errors)<=1e-10,gradient_screen_passed=screen,
                training_improved=bool(metrics['proposal']['passed'] and metrics['proposal']['data_loss']+prior(x)<metrics['current']['data_loss']),
                validation_improved=bool(final['data_loss']+prior(x)<parent_val),full_numerical_passed=final['passed'],
                parameters_passed=parameter_checks['proposal']['passed'],parameters_changed=bool(np.count_nonzero(x)),
                autonomy_passed=all(preserved(oldauto['rhythms'][l][j],auto['rhythms'][l][j]) for l in range(2) for j in range(2)),
                state_reproduced=max(state_errors.values())<=1e-12,panel_reproduced=panel_error<=1e-10)
            selection=json.loads(pin(root/'selection.json').read_text());chosen=decision(gates)
            expected_model=root/'trials/proposal/network.npz' if chosen else Path(m['parent'])
            scores={split:loss(full[2],m['targets'],ids,stats,split) for split in ('train','val','test')}
            if gates!=s['gates'] or scores!=s['candidate_scores'] or chosen!=s['fit_qualified_before_audit'] or selection['candidate_selected']!=chosen or file_sha256(expected_model)!=selection['selected_sha256'] or selection['selected_sha256']!=s['selected_sha256']:
                raise ValueError('Final selection/scores differ')
            if s['independent_full_loss_reproduction'] is not False:raise ValueError('Unperformed full loss replication claimed')
            report=dict(gates=gates,parameter_checks=parameter_checks,candidate_scores=scores,
                state_errors=state_errors,panel_error=panel_error)
    except Exception as exc:
        traceback.print_exc();problems.append(repr(exc))
    for required in ('manifest.json','summary.json','optimizer.npz','selection.json','candidate-autonomy.json'):
        if not (root/required).exists():missing.append('Unavailable: '+required)
    # Pin exact source/environment/config, optimizer, task and scientific outputs.
    for path in root.rglob('*'):
        if path.is_file() and path.name not in ('audit.json','budget_receipt.json') and path.suffix in ('.json','.npz','.py'):
            try:pin(path)
            except Exception as exc:problems.append(repr(exc))
    result=dict(experiment='E117',audit_complete=True,execution_complete=not missing and (root/'summary.json').exists(),
        integrity_passed=not problems,problems=problems,unavailable_evidence=missing,evidence_sha256=evidence,
        recomputed=report,behavioral_qualification=False,
        limits=['Saved-data audit, no independent scientific forward rerun.',
            'Control voltage maximum checked as saved reduction; full duplicate voltage traces unavailable.',
            'Full numerical atlas run once; only fresh states and finest panel independently replayed.'])
    write(root/'audit.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--parent-run',type=Path);p.add_argument('--coordinator',action='store_true');p.add_argument('--audit',action='store_true');p.add_argument('--report',action='store_true')
    a=p.parse_args();root=a.output_dir.resolve()
    if a.report:
        print('Run:',root)
        for name,keys in [('summary',('iterations','accepted_updates','selected_sha256','candidate_scores','candidate_parameters','gates','fit_qualified_before_audit','independent_full_loss_reproduction')),
                          ('audit',('audit_complete','execution_complete','integrity_passed','problems','unavailable_evidence')),
                          ('budget_receipt',('total_minutes','elapsed_seconds','all_steps_finished','timed_out','validation_complete','fit_qualified'))]:
            path=root/(name+'.json');d=json.loads(path.read_text()) if path.exists() else {}
            print(name+':',json.dumps({k:d.get(k) for k in keys}))
        return
    if a.audit:audit(root);return
    if a.coordinator:
        try:coordinator(root)
        except Exception as exc:traceback.print_exc();write(root/'failure.json',dict(error=repr(exc)));raise
        return
    if root==REPO or REPO in root.parents or root in REPO.parents:p.error('External output required')
    start=time.monotonic();end=start+120*60;root.mkdir(parents=True,exist_ok=False)
    parent=a.parent_run.resolve() if a.parent_run else latest_parent('/home/carsonc/r-elegans-results')
    write(root/'deadline.json',dict(parent_run=str(parent),total_minutes=120,coordinator_deadline=end-600,
        deadline_monotonic=end,started_monotonic=start,protocol='One eight-target/two-direction proposal; final audit included'))
    capture_run_record(root,REPO,vars(a),phase='E117-supervisor')
    from scripts.total_job_budget import run_group
    code,timeout=run_group([sys.executable,'-u','-m','scripts.run_panel_atlas_fit','--output-dir',str(root),'--coordinator'],max(.01,end-time.monotonic()-600))
    ac,at=run_group([sys.executable,'-u','-m','scripts.run_panel_atlas_fit','--output-dir',str(root),'--audit'],max(.01,end-time.monotonic()-10))
    def read(name):
        path=root/(name+'.json');return json.loads(path.read_text()) if path.exists() else {}
    au,s=read('audit'),read('summary');elapsed=time.monotonic()-start
    complete=bool(code==0 and not timeout and ac==0 and not at and au.get('audit_complete') and au.get('execution_complete') and elapsed<=120*60)
    receipt=dict(total_minutes=120,elapsed_seconds=elapsed,all_steps_finished=complete,timed_out=timeout,
        exit_code=code,audit_exit_code=ac,audit_timed_out=at,validation_complete=bool(s.get('validation_complete')),
        fit_qualified=bool(complete and au.get('integrity_passed') and s.get('fit_qualified_before_audit')))
    write(root/'budget_receipt.json',receipt);print(json.dumps(receipt),flush=True)
    raise SystemExit(0 if complete else 2)


if __name__=='__main__':main()
