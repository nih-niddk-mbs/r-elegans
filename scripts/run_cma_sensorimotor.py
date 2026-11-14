"""E077 bounded CMA-ES curriculum: paired sensory slowing and recovery."""
import argparse,concurrent.futures,json,math,multiprocessing,os,pickle,random,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from scripts.experiment_records import capture_run_record,file_sha256
from r_elegans.cma_curriculum import RUNGS,parameter_groups
from r_elegans.wall_budget import check_deadline
from scripts.cma_sensorimotor_worker import initialize,evaluate

PARENT='8e26b5d0b776edd29c1ded289c1d3aedc2897e4cce1159b08d98e0b75cc0ce2a'
SEED=7700


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def optimizer_snapshot(folder,es,stage,generation,pending=None):
    folder.mkdir(parents=True,exist_ok=False)
    # Local generated state only; never unpickle untrusted artifacts.
    with (folder/'optimizer.pkl').open('wb') as f:pickle.dump(dict(es=es,numpy_rng=np.random.get_state(),python_rng=random.getstate(),stage=stage,generation=generation,pending=pending),f)
    write(folder/'metadata.json',dict(stage=stage,generation=generation,pending=pending,optimizer_sha256=file_sha256(folder/'optimizer.pkl')))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--source-run',type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();source=a.source_run.resolve();repo=Path(__file__).resolve().parents[1]
    for root in (repo,source):
        if out==root or root in out.parents or out in root.parents:raise ValueError('Separate external output required')
    budget=json.loads((out/'deadline.json').read_text());finish=budget['coordinator_finish_by'];check_deadline(finish)
    # Install the pinned pure-Python optimizer only in this dedicated interpreter.
    from importlib import metadata
    try:version=metadata.version('cma')
    except metadata.PackageNotFoundError:version=None
    if version!='4.4.1':subprocess.run([sys.executable,'-m','pip','install','cma==4.4.1'],check=True,timeout=min(180,max(1,int(finish-time.monotonic()-60))))
    import cma
    capture_run_record(out,repo,a)
    from scripts.compare_rhythm_worker import verify_inputs
    m=verify_inputs(source/'control_inputs')
    if m['continuation_parent_sha256']!=PARENT or file_sha256(source/'control_inputs/continuation_parent.npz')!=PARENT:raise ValueError('Qualified E071 motor parent required')
    inputs=out/'inputs';shutil.copytree(source/'control_inputs',inputs);verify_inputs(inputs)
    from scripts.neural_timing_setup import setup
    d=setup(inputs);schema=parameter_groups(d['spec'],d['model'].neuron_ids,d['origin'],d['lower'],d['upper'])
    if not 10<=len(schema)<=30:raise ValueError('Unexpected grouped search dimension')
    write(out/'parameter_groups.json',schema)
    np.savez_compressed(out/'parameter_origin.npz',origin=d['origin'],lower=d['lower'],upper=d['upper'])
    from scripts.benchmark_cpu import physical_core_cpus
    from scripts.benchmark_cpu_parallel import worker_cpu_groups
    cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0)))
    if not known:raise ValueError('Known physical CPU topology required')
    groups=worker_cpu_groups(cpus,4,4)
    env=dict(JAX_PLATFORMS='cpu',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    os.environ.update(env)
    fit_until=finish-30*60
    if fit_until-time.monotonic()<20*60:raise ValueError('Insufficient search and validation budget')
    write(out/'configuration.json',dict(experiment='E077',parent_sha256=PARENT,seed=SEED,optimizer='pycma4.4.1',dimension=len(schema),
        population=12,initial_sigma=.25,rungs=RUNGS,total_minutes=budget['total_minutes'],validation_reserve_minutes=30,
        objective='Cue-specific cadence reduction with retained bends, unchanged uncued movement and START recovery versus STOP-only',
        full_network=True,full_stopping_qualified=False,swimming_blocked=True,manual_diagnostic_input=False))
    random.seed(SEED);np.random.seed(SEED)
    ctx=multiprocessing.get_context('spawn')
    futures_used=[];evaluations=[];milestones=[];stage=0;generation=0;selected=None;selected_z=np.zeros(len(schema));recent=[];failure=None;last_pending=None;attempted_checks=set()
    def job(z,target,name,until,parent=None,variant='nominal',grid='ordinary',reference=False):
        return dict(z=list(map(float,z)),target=target,output=str(out/'evaluations'/name),until=until,parent_features=parent,variant=variant,grid=grid,reference=reference)
    def run_batch(pool,jobs):
        pending={pool.submit(evaluate,j):j for j in jobs};result=[];last=time.monotonic()
        futures_used.extend(pending)
        while pending:
            check_deadline(finish-45)
            ready,_=concurrent.futures.wait(pending,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in ready:
                j=pending.pop(future)
                try:r=future.result()
                except Exception as e:r=dict(status='worker_error',error=str(e),output=j['output'])
                result.append(r);evaluations.append(r)
                print(json.dumps(dict(evaluation=Path(j['output']).name,status=r['status'],target=j['target'],loss=r.get('metrics',{}).get('loss'),passed=r.get('metrics',{}).get('partial_cadence_passed'))),flush=True)
            if time.monotonic()-last>=30:
                print(json.dumps(dict(stage=stage,generation=generation,pending=len(pending),remaining_minutes=(finish-time.monotonic())/60)),flush=True);last=time.monotonic()
        by_path={r['output']:r for r in result}
        return [by_path[j['output']] for j in jobs]
    def valid(r):
        m=r.get('metrics',{})
        return r['status']=='completed' and m.get('safe') and m.get('uncued_movement_preserved') and m.get('pre_movement_preserved') and m.get('bend_amplitude_preserved')
    def preserve(r,label):
        folder=out/'selected'/label;folder.mkdir(parents=True,exist_ok=False)
        # Reference full immutable evaluation directory rather than duplicating large traces.
        for name in ('network.npz','coordinates.npz','request.json','summary.json'):shutil.copy2(Path(r['output'])/name,folder/name)
        write(folder/'reference.json',dict(evaluation=r['output'],checkpoint_sha256=r['checkpoint_sha256'],trajectory_sha256=r['trajectory_sha256']))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=ctx,initializer=initialize,
            initargs=(str(inputs),schema,groups,str(out/'workers'),finish-60)) as pool:
        try:
            baseline=run_batch(pool,[job(selected_z,RUNGS[0],'parent',fit_until,reference=True)])[0]
            if baseline['status']!='completed':raise ValueError('Parent baseline failed: '+str(baseline))
            parent=baseline['paired_features']['no_cue'];write(out/'parent_features.json',parent)
            recent=[baseline['elapsed_seconds']];selected=baseline;preserve(baseline,'parent')
            es=cma.CMAEvolutionStrategy(selected_z,.25,{'seed':SEED,'popsize':12,'bounds':[-1,1],'verbose':-9,'verb_log':0})
            optimizer_snapshot(out/'optimizer'/'initial',es,stage,generation)
            best_loss=baseline['metrics']['loss']
            while stage<len(RUNGS):
                estimate=math.ceil(12/4)*max(recent[-12:])*1.4
                if time.monotonic()+estimate>fit_until:break
                points=es.ask();last_pending=[list(map(float,z)) for z in points]
                optimizer_snapshot(out/'optimizer'/f'g-{generation:05d}-asked',es,stage,generation,last_pending)
                jobs=[job(z,RUNGS[stage],f'g-{generation:05d}-c-{i:02d}',fit_until,parent) for i,z in enumerate(points)]
                results=run_batch(pool,jobs)
                if any(r['status'] in ('time_budget','worker_error') for r in results):break
                losses=[r['metrics']['loss'] if r['status']=='completed' else 1e9 for r in results]
                es.tell(points,losses);last_pending=None
                for z,r in zip(points,results):
                    if r['status']=='completed':recent.append(r['elapsed_seconds'])
                    if valid(r) and r['metrics']['loss']<best_loss:
                        selected=r;selected_z=np.asarray(z).copy();best_loss=r['metrics']['loss'];preserve(r,f'g-{generation:05d}-'+Path(r['output']).name)
                optimizer_snapshot(out/'optimizer'/f'g-{generation:05d}-told',es,stage,generation)
                with (out/'generation_history.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(generation=generation,target=RUNGS[stage],minimum_population_loss=min(losses),selected_loss=best_loss,selected_checkpoint_sha256=selected['checkpoint_sha256'],completed=sum(r['status']=='completed' for r in results),failures=sum(r['status']!='completed' for r in results)))+'\n')
                generation+=1
                from r_elegans.cma_curriculum import score
                selected_score=score(selected['paired_features'],parent,RUNGS[stage])
                key=(stage,selected['checkpoint_sha256'])
                if valid(selected) and selected_score['partial_cadence_passed'] and key not in attempted_checks:
                    # Verify reduced-dose shifted cues and repetition on the exact model.
                    if time.monotonic()+2*max(recent[-12:])*1.4>fit_until:break
                    attempted_checks.add(key)
                    checks=run_batch(pool,[job(selected_z,RUNGS[stage],f'rung-{stage}-g-{generation}-'+v,fit_until,parent,variant=v) for v in ('low_shift','repeat')])
                    same=all(r.get('checkpoint_sha256')==selected['checkpoint_sha256'] for r in checks)
                    passed=same and all(valid(r) and r['metrics']['partial_cadence_passed'] for r in checks)
                    milestones.append(dict(target=RUNGS[stage],checkpoint_sha256=selected['checkpoint_sha256'],checks=checks,passed=passed))
                    if passed:
                        stage+=1
                        if stage==len(RUNGS):break
                        # Objective changed: fresh CMA distribution at selected weights.
                        es=cma.CMAEvolutionStrategy(selected_z,.15,{'seed':SEED+stage,'popsize':12,'bounds':[-1,1],'verbose':-9,'verb_log':0})
                        from r_elegans.cma_curriculum import score
                        best_loss=score(selected['paired_features'],parent,RUNGS[stage])['loss']
                        optimizer_snapshot(out/'optimizer'/f'rung-{stage}-fresh',es,stage,generation)
                if es.stop():break
            refinement=[]
            if selected is not None and finish-time.monotonic()>5*60:
                target=RUNGS[min(stage,len(RUNGS)-1)]
                refinement=run_batch(pool,[job(selected_z,target,'final-'+label,finish-90,parent,variant=v,grid=g) for label,v,g in [('low-shift','low_shift','ordinary'),('repeat','repeat','ordinary'),('neural-half','nominal','neural-half'),('body-half','nominal','body-half')]])
            preserve(selected,'final')
            optimizer_snapshot(out/'optimizer'/'final',es,stage,generation,last_pending)
        except Exception as e:failure=repr(e);print('CMA_PILOT_ERROR:',failure,flush=True);refinement=[]
    from scripts.compare_rhythm_worker import verify_inputs
    verify_inputs(inputs)
    completed=bool(failure is None and len(refinement)==4 and all(r['status']=='completed' for r in refinement))
    from r_elegans.cma_curriculum import score
    target=RUNGS[min(stage,len(RUNGS)-1)]
    nominal_score=score(selected['paired_features'],parent,target) if selected is not None and failure is None else None
    numerical=[]
    if nominal_score is not None:
        for r in refinement:
            if r.get('grid') not in ('neural-half','body-half') or r['status']!='completed':continue
            a=[f for row in nominal_score['cadence_ratios'] for f in row['frequency_ratios']]
            b=[f for row in r['metrics']['cadence_ratios'] for f in row['frequency_ratios']]
            finite=all(x is not None and y is not None for x,y in zip(a,b))
            error=max(abs(x-y) for x,y in zip(a,b)) if finite else None
            amplitude=True
            for c in selected['paired_features']:
                for label in ('on','off'):
                    aa=np.asarray([v['regional_cycles_rad'] for v in selected['paired_features'][c][label]])
                    bb=np.asarray([v['regional_cycles_rad'] for v in r['paired_features'][c][label]])
                    amplitude=amplitude and bool(np.all(abs(aa-bb)<=np.maximum(1e-5,.1*aa)))
            numerical.append(dict(grid=r['grid'],maximum_cadence_ratio_difference=error,amplitudes_consistent=amplitude,passed=bool(finite and error<=.02 and amplitude)))
    final_grid_pass=bool(completed and nominal_score and nominal_score['partial_cadence_passed'] and len(numerical)==2 and all(v['passed'] for v in numerical) and all(valid(r) and r['metrics']['partial_cadence_passed'] and r.get('checkpoint_sha256')==selected['checkpoint_sha256'] for r in refinement))
    summary=dict(experiment='E077',execution_valid=bool(failure is None and not any(r['status'] in ('worker_error','time_budget') for r in evaluations)),
        all_stages_finished=completed,validation_complete=completed,completed_generations=generation,dimension=len(schema),
        evaluation_status_counts={status:sum(r['status']==status for r in evaluations) for status in ('completed','failed','worker_error','time_budget')},
        curriculum_rungs_passed_training_checks=[r['target'] for r in milestones if r['passed']],milestones=milestones,
        selected=selected,active_target=target,selected_rescored_for_active_target=nominal_score,refinement=refinement,numerical_consistency=numerical,final_grid_partial_cadence_pass=final_grid_pass,
        full_stopping_qualified=False,swimming_qualified=False,swimming_blocked=True,failure=failure,
        scope='Partial sensory slowing curriculum; no full start/stop, causal-control or biological qualification')
    write(out/'summary.json',summary);print('CMA_CURRICULUM_REPORT:',out/'summary.json',flush=True)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('selected','refinement','milestones')}),flush=True)


if __name__=='__main__':main()
