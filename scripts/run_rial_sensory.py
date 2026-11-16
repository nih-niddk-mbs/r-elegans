"""E085-E088 body, RIAL release or relay operating-range fitting with a total wall budget."""
import argparse,concurrent.futures,json,math,multiprocessing,os,random,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from scripts.experiment_records import capture_run_record,file_sha256
from scripts.run_cma_sensorimotor import optimizer_snapshot
from scripts.rial_sensory_worker import initialize,evaluate,setup
from r_elegans.rial_sensory import parameter_groups,RUNGS as BODY_RUNGS,score as body_score

SEED=8500
PARENT='dd42e3f62bc00befc6052fae60c102bc89aaf9266ae88da1f78f3be30c2766a8'
RELAY_PARENT='5b77ec25266a4849b9e843c4fb1891f243ef31beef18c46fc9fd0d84f989c022'
RELAY_CONTINUATION_PARENT='608587d8cc9267674b5828df02d7891ae8940f46cc6981b62518b7f89068de73'
RELEASE_PARENT='f8cd5b401d81d297eb40d44ee3e355d5915ef2a07576077e022e16ca6d464b7f'


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--objective',choices=('body','rial-release','relay-range'),default='body')
    p.add_argument('--continue-relay',action='store_true',help='Fresh optimizer at pinned E087 selected weights, original-anchor bounds unchanged')
    a=p.parse_args();release_objective=a.objective=='rial-release';relay_objective=a.objective=='relay-range'
    if a.continue_relay and not relay_objective:p.error('--continue-relay requires --objective relay-range')
    neural_objective=release_objective or relay_objective
    from r_elegans.rial_release import RUNGS as RELEASE_RUNGS,score as release_score
    from r_elegans.relay_range import RUNGS as RELAY_RUNGS,score as relay_score,parameter_groups as relay_groups,CUE_SCALE,direction_consistent
    RUNGS=RELAY_RUNGS if relay_objective else RELEASE_RUNGS if release_objective else BODY_RUNGS
    score=relay_score if relay_objective else release_score if release_objective else body_score
    experiment='E087' if relay_objective else 'E086' if release_objective else 'E085'
    run_seed=8700 if relay_objective else 8600 if release_objective else SEED
    expected_parent=RELAY_PARENT if relay_objective else RELEASE_PARENT if release_objective else PARENT
    if a.continue_relay:
        experiment='E088';run_seed=8800;expected_parent=RELAY_CONTINUATION_PARENT
    out=a.output_dir.resolve();source=a.source_run.resolve();repo=Path(__file__).resolve().parents[1]
    for root in (repo,source):
        if out==root or root in out.parents or out in root.parents:
            raise ValueError('Separate external output required')
    budget=json.loads((out/'deadline.json').read_text());finish=budget['coordinator_finish_by']
    from importlib import metadata
    try:version=metadata.version('cma')
    except metadata.PackageNotFoundError:version=None
    if version!='4.4.1':
        subprocess.run([sys.executable,'-m','pip','install','cma==4.4.1'],check=True,timeout=min(180,max(1,int(finish-time.monotonic()-60))))
    import cma
    capture_run_record(out,repo,a)
    from scripts.compare_rhythm_worker import verify_inputs
    source_summary=json.loads((source/'summary.json').read_text())
    if a.continue_relay and (source_summary.get('experiment')!='E087' or source_summary.get('selected',{}).get('checkpoint_sha256')!=expected_parent):
        raise ValueError('Continuation requires pinned E087 selected checkpoint; no fallback')
    ordinary=next(r for r in source_summary['validation'] if r['variant']==('nominal' if neural_objective else 'ordinary') and r['grid']=='ordinary' and (not neural_objective or r.get('mode')=='normal'))
    folder=Path(ordinary['output']).resolve()
    if source not in folder.parents or ordinary['checkpoint_sha256']!=expected_parent or file_sha256(folder/'network.npz')!=expected_parent:
        raise ValueError('Pinned parent model required for '+experiment)
    verify_inputs(source/'inputs');inputs=out/'inputs';shutil.copytree(source/'inputs',inputs)
    manifest=json.loads((inputs/'manifest.json').read_text())
    for name,path in [('sensory_parent.npz',folder/'network.npz'),('sensory_parent_coordinates.npz',folder/'coordinates.npz'),('sensory_parent_summary.json',source/'summary.json')]:
        sha=file_sha256(path);shutil.copy2(path,inputs/name)
        if file_sha256(inputs/name)!=sha or file_sha256(path)!=sha:raise ValueError('Parent changed during copy')
        manifest['sha256'][name]=sha;manifest['sources'][name]=str(path)
    reference_name='no_cue' if neural_objective else 'sham'
    reference=folder/(reference_name+'.npz')
    if file_sha256(reference)!=ordinary['trajectory_sha256'][reference_name]:raise ValueError('Reference hash mismatch')
    write(inputs/'sensory_reference.json',dict(path=str(reference),sha256=ordinary['trajectory_sha256'][reference_name]))
    manifest['sha256']['sensory_reference.json']=file_sha256(inputs/'sensory_reference.json');manifest['sources']['sensory_reference.json']=str(reference)
    manifest['consumer_experiment']=experiment;write(inputs/'manifest.json',manifest);verify_inputs(inputs)
    d=setup(inputs);schema=(relay_groups if relay_objective else parameter_groups)(d['spec'],d['model'].neuron_ids,d['origin'],d['lower'],d['upper'])
    write(out/'parameter_groups.json',schema)
    np.savez_compressed(out/'parameter_origin.npz',origin=d['origin'],lower=d['lower'],upper=d['upper'])
    from scripts.benchmark_cpu import physical_core_cpus
    from scripts.benchmark_cpu_parallel import worker_cpu_groups
    cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0)))
    if not known:raise ValueError('Known physical CPU topology required')
    groups=worker_cpu_groups(cpus,4,4)
    os.environ.update(JAX_PLATFORMS='cpu',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    reserve=max(60*60,.4*(finish-time.monotonic())) if neural_objective else max(2*3600,.25*(finish-time.monotonic()));fit_until=finish-reserve
    if fit_until-time.monotonic()<20*60:raise ValueError('Insufficient search and final-validation budget')
    write(out/'configuration.json',dict(experiment=experiment,parent_sha256=expected_parent,seed=run_seed,
        initialization='Fresh CMA at selected weights; local trust region recentered, original-anchor bounds unchanged; not exact optimizer resume',
        optimizer='cma==4.4.1',population=12,dimension=len(schema),cpu_groups=groups,total_minutes=budget['total_minutes'],
        validation_reserve_minutes=reserve/60,rungs=RUNGS,training_margin=1.3,
        objective=('AIYL release headroom2/5/10% plus cue responsiveness and recovery; body gait preserved' if relay_objective else 'Early cue-specific1/2/5% RIAL release-cycle attenuation; body gait preserved' if release_objective else 'Early cue-specific2/5/10% cycle-amplitude attenuation with uncued gait and recovery preserved'),
        frequency_estimator=('parabolic_subsample_median_period' if neural_objective else 'discrete_peak_median_period'),
        trainable=('AWAL/AIYL nonleak densities, existing inputs to AIYL and AIYL-to-RIAL compensation' if relay_objective else 'Explicit AWA/AIY/AIZ/AIB/RIA nonleak densities and existing control-to-control chemical connections'),
        fixed='Motor channels, RIAL-to-SMD strengths, gaps, leak/capacitance/kinetics/reversals, NMJs/sensory gains/physics',
        diagnostic_clamps_in_training=False,full_start_stop_qualified=False,maximum_new_run_GiB=30,minimum_free_GiB=50,
        retention='All model/coordinate/optimizer/RNG/request/metrics; raw trajectories only parent and final tests'))
    random.seed(run_seed);np.random.seed(run_seed)
    es=cma.CMAEvolutionStrategy(np.zeros(len(schema)),.3,{'seed':run_seed,'popsize':12,'bounds':[-1,1],'verbose':-9,'verb_log':0})
    optimizer_snapshot(out/'optimizer/initial',es,0,0)
    selected=None;zbest=np.zeros(len(schema));parent=None;generation=0;history=[];validation=[];failure=None;stage=0;evaluations=[]
    stop_reason='time_budget';recent=[];pending_points=None;baseline=None;qualified=None;milestones=[];restart=0
    def job(z,name,until,variant='nominal',grid='ordinary',retain=False,reference=False,mode='normal',target=None,margin=1.):
        return dict(z=list(map(float,z)),output=str(out/'evaluations'/name),until=until,parent_features=parent,
            variant=variant,grid=grid,retain_traces=retain,reference=reference,mode=mode,objective=a.objective,
            target=RUNGS[min(stage,len(RUNGS)-1)] if target is None else target,margin=margin)
    def valid(r):
        m=r.get('metrics',{})
        return bool(r['status']=='completed' and m.get('safe') and m.get('baseline_and_recovery_preserved') and m.get('useful_response_preserved'))
    def preserve(r,label):
        folder=out/'selected'/label;folder.mkdir(parents=True,exist_ok=False)
        for name in ('network.npz','coordinates.npz','request.json','summary.json'):shutil.copy2(Path(r['output'])/name,folder/name)
        write(folder/'reference.json',dict(evaluation=r['output'],checkpoint_sha256=r['checkpoint_sha256']))
    def batch(pool,jobs):
        futures={pool.submit(evaluate,j):j for j in jobs};rows={};last=time.monotonic()
        while futures:
            ready,_=concurrent.futures.wait(futures,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in ready:
                j=futures.pop(f)
                try:r=f.result()
                except Exception as e:r=dict(status='worker_error',error=repr(e),output=j['output'])
                rows[j['output']]=r
                evaluations.append(r)
                print(json.dumps(dict(evaluation=Path(j['output']).name,status=r['status'],metrics=r.get('metrics'),error=r.get('error'))),flush=True)
            if time.monotonic()-last>30:
                print(json.dumps(dict(generation=generation,pending=len(futures),stage=stage,remaining_total_minutes=(finish-time.monotonic())/60)),flush=True);last=time.monotonic()
        return [rows[j['output']] for j in jobs]
    ctx=multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=ctx,initializer=initialize,
                initargs=(str(inputs),schema,groups,str(out/'workers'))) as pool:
        try:
            baseline=batch(pool,[job(zbest,'parent',fit_until,retain=True,reference=True,margin=1.3)])[0]
            if baseline['status']!='completed':raise ValueError('Parent baseline failed: '+str(baseline))
            parent=baseline['paired_features']['no_cue'];selected=baseline;recent=[baseline['elapsed_seconds']]
            best=baseline['metrics']['loss'];write(out/'parent_features.json',parent);preserve(baseline,'parent')
            while stage<len(RUNGS):
                estimate=3*max(recent[-12:])*1.4
                if time.monotonic()+estimate>fit_until:break
                size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
                if size>15*1024**3 or shutil.disk_usage(out).free<50*1024**3:stop_reason='disk_budget';break
                points=es.ask();pending_points=[list(map(float,z)) for z in points]
                optimizer_snapshot(out/'optimizer'/f'g-{generation:05d}-asked',es,stage,generation,pending_points)
                rows=batch(pool,[job(z,f'g-{generation:05d}-c-{i:02d}',fit_until,margin=1.3) for i,z in enumerate(points)])
                if any(r['status'] in ('time_budget','disk_budget','worker_error') for r in rows):stop_reason='evaluation_failure';break
                losses=[r['metrics']['loss'] if r['status']=='completed' else 1e6 for r in rows]
                es.tell(points,losses);pending_points=None
                for z,r in zip(points,rows):
                    if r['status']=='completed':recent.append(r['elapsed_seconds'])
                    if valid(r) and r['metrics']['loss']<best:
                        selected=r;zbest=np.asarray(z).copy();best=r['metrics']['loss']
                optimizer_snapshot(out/'optimizer'/f'g-{generation:05d}-told',es,stage,generation)
                history.append(dict(generation=generation,target=RUNGS[stage],minimum_loss=min(losses),selected=selected['checkpoint_sha256'],selected_evaluation=selected['output']))
                write(out/'generation_history.json',history);generation+=1
                if not any(r['status']=='completed' for r in rows):stop_reason='all_candidates_failed';break
                current=score(selected['paired_features'],parent,RUNGS[stage],1.3)
                if valid(selected) and current['training_margin_met'] and time.monotonic()+max(recent[-12:])*1.5<fit_until:
                    checks=batch(pool,[job(zbest,f'rung-{stage}-g-{generation}-'+v,fit_until,variant=v) for v in ('low-dose','shift')])
                    passed=all(valid(r) and r['metrics']['partial_attenuation_passed'] and r.get('checkpoint_sha256')==selected['checkpoint_sha256'] for r in checks)
                    if relay_objective and passed:passed=direction_consistent([selected]+checks)
                    milestones.append(dict(target=RUNGS[stage],passed=passed,checks=checks,checkpoint_sha256=selected['checkpoint_sha256']))
                    if passed:
                        qualified=(selected,zbest.copy(),RUNGS[stage]);preserve(selected,f'rung-{stage}')
                        stage+=1
                        if stage==len(RUNGS):stop_reason='curriculum_complete';break
                        es=cma.CMAEvolutionStrategy(zbest,.15,{'seed':run_seed+stage,'popsize':12,'bounds':[-1,1],'verbose':-9,'verb_log':0})
                        best=score(selected['paired_features'],parent,RUNGS[stage],1.3)['loss']
                        optimizer_snapshot(out/'optimizer'/f'stage-{stage}-fresh',es,stage,generation)
                if es.stop():
                    restart+=1;es=cma.CMAEvolutionStrategy(zbest,.2,{'seed':run_seed+1000+restart,'popsize':12,'bounds':[-1,1],'verbose':-9,'verb_log':0})
                    optimizer_snapshot(out/'optimizer'/f'g-{generation:05d}-restart-{restart}',es,stage,generation)
            final_selected,final_z,final_target=qualified if qualified is not None else (selected,zbest.copy(),RUNGS[0])
            tests=[('nominal','nominal','ordinary','normal'),('low-dose','low-dose','ordinary','normal'),('shift','shift','ordinary','normal'),
                   ('repeat','repeat','ordinary','normal'),('redundant','redundant','ordinary','normal'),('long-cue','long-cue','ordinary','normal'),
                   ('neural-half','nominal','neural-half','normal'),('body-half','nominal','body-half','normal'),
                   ('sensory-blocked','nominal','ordinary','sensory-blocked'),('relay-replay' if relay_objective else 'rial-replay','nominal','ordinary','relay-replay' if relay_objective else 'rial-replay')]
            validation=batch(pool,[job(final_z,'final-'+label,finish-90,v,g,True,mode=mode,target=final_target) for label,v,g,mode in tests])
            preserve(final_selected,'final')
        except Exception as e:
            failure=repr(e);print('RIAL_SENSORY_ERROR:',failure,flush=True)
    optimizer_snapshot(out/'optimizer/final',es,stage,generation,pending_points)
    verify_inputs(inputs)
    complete=bool(failure is None and len(validation)==10 and all(r['status']=='completed' for r in validation))
    convergence=[];partial=False;blocked=False;mediated=False;sustained=False;relay_direction=False
    if complete:
        nominal=validation[0]['metrics']
        for row in validation[6:8]:
            error=max(abs(x-y) for x,y in zip(nominal['response_amplitude_ratios'],row['metrics']['response_amplitude_ratios']))
            convergence.append(dict(grid=row['grid'],amplitude_ratio_difference=error,passed=bool(error<=min(.005,.25*final_target)*(CUE_SCALE if relay_objective else 1.))))
        blocked=bool(valid(validation[8]) and all(abs(r-1)<=.25*final_target*(CUE_SCALE if relay_objective else 1.) for r in validation[8]['metrics']['response_amplitude_ratios']))
        mediated=bool(validation[9]['mediation']['cue_attenuation_reduced'])
        same=all(r.get('checkpoint_sha256')==final_selected['checkpoint_sha256'] for r in validation)
        partial=bool(same and all(valid(r) and r['metrics']['partial_attenuation_passed'] for r in validation[:8])
                     and all(r['passed'] for r in convergence) and blocked)
        if relay_objective:
            relay_direction=direction_consistent(validation[:8]);partial=partial and relay_direction
        sustained=bool(all(r<=1-final_target*(CUE_SCALE if relay_objective else 1.) for r in validation[5]['metrics']['late_amplitude_ratios']))
    write(out/'summary.json',dict(experiment=experiment,objective=a.objective,execution_valid=failure is None and stop_reason not in ('evaluation_failure','all_candidates_failed'),
        all_stages_finished=complete,validation_complete=complete,completed_generations=generation,active_stage=stage,
        evaluation_status_counts={status:sum(r['status']==status for r in evaluations) for status in ('completed','failed','worker_error','time_budget','disk_budget')},
        stop_reason=stop_reason,failure=failure,baseline=baseline,selected=final_selected if complete else selected,
        final_target=final_target if complete else None,milestones=milestones,validation=validation,numerical_checks=convergence,
        final_target_meaning='minimum_AIYL_upper_release_headroom' if relay_objective else 'minimum_fractional_amplitude_attenuation',
        partial_sensory_component_qualified=partial and not neural_objective,
        sensory_RIAL_release_component_qualified=partial and release_objective,
        sensory_relay_range_component_qualified=partial and relay_objective,sensory_block_control=blocked,RIAL_chemical_replay_mediation=mediated and not relay_objective,
        relay_response_direction_consistent_across_cases=relay_direction,
        AIYL_to_RIAL_body_effect_mediation=mediated and relay_objective,
        RIAL_mediated_partial_component_qualified=partial and mediated and not neural_objective,
        long_cue_sustained_attenuation=sustained and not neural_objective,
        long_cue_RIAL_release_attenuation=sustained and release_objective,
        long_cue_relay_response=sustained and relay_objective,
        full_start_stop_qualified=False,swimming_qualified=False,
        scope=('AIYL operating range and sensory responsiveness only; no RIAL/body stopping qualification' if relay_objective else 'Sensory-triggered partial RIAL release-cycle attenuation; no body slowing qualification' if release_objective else 'Early sensory-triggered partial amplitude attenuation and recovery; no rest/initiation, full start/stop or navigation qualification')))
    print('RIAL_SENSORY_REPORT:',out/'summary.json',flush=True)


if __name__=='__main__':main()
