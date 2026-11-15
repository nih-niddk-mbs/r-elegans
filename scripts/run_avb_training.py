"""E078 eight-hour-total conditional AVB motor-circuit fitting, not sensory policy learning."""
import argparse,concurrent.futures,json,math,multiprocessing,os,random,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from scripts.experiment_records import capture_run_record,file_sha256
from scripts.run_cma_sensorimotor import optimizer_snapshot,PARENT
from scripts.avb_training_worker import initialize,evaluate
from r_elegans.avb_training import parameter_groups

SEED=7800


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--source-run',type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();source=a.source_run.resolve();repo=Path(__file__).resolve().parents[1]
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
    original=source/'control_inputs';m=verify_inputs(original)
    if m['continuation_parent_sha256']!=PARENT or file_sha256(original/'continuation_parent.npz')!=PARENT:
        raise ValueError('Pinned E071 parent required')
    inputs=out/'inputs';shutil.copytree(original,inputs);verify_inputs(inputs)
    from scripts.neural_timing_setup import setup
    d=setup(inputs);schema=parameter_groups(d['spec'],d['model'].neuron_ids,d['origin'],d['lower'],d['upper'])
    write(out/'parameter_groups.json',schema)
    np.savez_compressed(out/'parameter_origin.npz',origin=d['origin'],lower=d['lower'],upper=d['upper'])
    from scripts.benchmark_cpu import physical_core_cpus
    from scripts.benchmark_cpu_parallel import worker_cpu_groups
    cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0)))
    if not known:raise ValueError('Known physical CPU topology required')
    groups=worker_cpu_groups(cpus,4,4)
    os.environ.update(JAX_PLATFORMS='cpu',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    reserve=max(90*60,.25*(finish-time.monotonic()))
    fit_until=finish-reserve
    if fit_until-time.monotonic()<20*60:raise ValueError('Insufficient total budget')
    write(out/'configuration.json',dict(experiment='E078',parent_sha256=PARENT,seed=SEED,
        optimizer='cma==4.4.1',population=12,dimension=len(schema),cpu_groups=groups,
        total_minutes=budget['total_minutes'],validation_reserve_minutes=reserve/60,
        mechanism='Endogenous baseline rhythm; AVBL/AVBR-only 0.1nS -80mV conductance during off interval; release for recovery',
        objective='Continuous body-range and motor-voltage suppression with baseline water gait preserved',
        sensory_system='Live chemical/proprioceptive pathways; no chemical cue in component training',
        maximum_new_run_GiB=30,minimum_free_GiB=50,search_trajectory_retention='metrics only; every model/coordinate/optimizer retained',
        sensory_start_stop_qualified=False,swimming_qualified=False))
    random.seed(SEED);np.random.seed(SEED)
    es=cma.CMAEvolutionStrategy(np.zeros(len(schema)),.3,{'seed':SEED,'popsize':12,'bounds':[-1,1],'verbose':-9,'verb_log':0})
    optimizer_snapshot(out/'optimizer/initial',es,0,0)
    selected=None;zbest=np.zeros(len(schema));parent=None;generation=0;history=[];validation=[];failure=None
    stop_reason='time_budget';recent=[];pending_points=None;baseline=None
    def job(z,name,until,variant='ordinary',grid='ordinary',retain=False,reference=False):
        return dict(z=list(map(float,z)),output=str(out/'evaluations'/name),until=until,
            parent_features=parent,variant=variant,grid=grid,retain_traces=retain,reference=reference)
    def batch(pool,jobs):
        futures={pool.submit(evaluate,j):j for j in jobs};rows={};last=time.monotonic()
        while futures:
            ready,_=concurrent.futures.wait(futures,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in ready:
                j=futures.pop(f)
                try:r=f.result()
                except Exception as e:r=dict(status='worker_error',error=repr(e),output=j['output'])
                rows[j['output']]=r
                print(json.dumps(dict(evaluation=Path(j['output']).name,status=r['status'],metrics=r.get('metrics'),error=r.get('error'))),flush=True)
            if time.monotonic()-last>30:
                print(json.dumps(dict(generation=generation,pending=len(futures),remaining_total_minutes=(finish-time.monotonic())/60)),flush=True);last=time.monotonic()
        return [rows[j['output']] for j in jobs]
    ctx=multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=ctx,initializer=initialize,
                initargs=(str(inputs),schema,groups,str(out/'workers'),finish-60)) as pool:
        try:
            baseline=batch(pool,[job(zbest,'parent',fit_until,retain=True,reference=True)])[0]
            if baseline['status']!='completed':raise ValueError('Parent baseline failed: '+str(baseline))
            parent=baseline['features']['sham'];selected=baseline;recent=[baseline['elapsed_seconds']]
            best=baseline['metrics']['loss'];write(out/'parent_features.json',parent)
            while True:
                estimate=3*max(recent[-12:])*1.4
                if time.monotonic()+estimate>fit_until:break
                # A total per-run cap includes source/input/optimizer/model records, not just trajectories.
                size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
                if size>15*1024**3 or shutil.disk_usage(out).free<50*1024**3:
                    stop_reason='disk_budget';break
                points=es.ask();pending_points=[list(map(float,z)) for z in points]
                optimizer_snapshot(out/'optimizer'/f'g-{generation:05d}-asked',es,0,generation,pending_points)
                rows=batch(pool,[job(z,f'g-{generation:05d}-c-{i:02d}',fit_until) for i,z in enumerate(points)])
                if any(r['status']!='completed' for r in rows):
                    stop_reason='evaluation_failure';break
                es.tell(points,[r['metrics']['loss'] for r in rows]);pending_points=None
                for z,r in zip(points,rows):
                    recent.append(r['elapsed_seconds'])
                    if r['metrics']['loss']<best:
                        selected=r;zbest=np.asarray(z).copy();best=r['metrics']['loss']
                optimizer_snapshot(out/'optimizer'/f'g-{generation:05d}-told',es,0,generation)
                history.append(dict(generation=generation,minimum_loss=min(r['metrics']['loss'] for r in rows),selected=selected['checkpoint_sha256']))
                write(out/'generation_history.json',history);generation+=1
                if selected['metrics']['avb_component_passed']:
                    stop_reason='training_target_met';break
                if es.stop():
                    # Fresh distribution at the best point; persist the restart as a new optimizer state.
                    es=cma.CMAEvolutionStrategy(zbest,.15,{'seed':SEED+generation,'popsize':12,'bounds':[-1,1],'verbose':-9,'verb_log':0})
                    optimizer_snapshot(out/'optimizer'/f'g-{generation:05d}-restart',es,0,generation)
            # Always repeat the selected model with full traces, even if the parent remains best.
            validation=batch(pool,[job(zbest,'final-'+label,finish-90,variant,grid,True)
                for label,variant,grid in [('ordinary','ordinary','ordinary'),('shift-weak','shift-weak','ordinary'),
                    ('repeat','repeat','ordinary'),('neural-half','ordinary','neural-half'),('body-half','ordinary','body-half')]])
        except Exception as e:
            failure=repr(e);print('AVB_TRAINING_ERROR:',failure,flush=True)
    optimizer_snapshot(out/'optimizer/final',es,0,generation,pending_points)
    if selected:
        final=out/'selected';final.mkdir()
        for name in ('network.npz','coordinates.npz','request.json','summary.json'):
            shutil.copy2(Path(selected['output'])/name,final/name)
        write(final/'reference.json',dict(evaluation=selected['output'],checkpoint_sha256=selected['checkpoint_sha256']))
    verify_inputs(inputs)
    complete=bool(failure is None and len(validation)==5 and all(r['status']=='completed' for r in validation))
    convergence=[]
    if complete:
        ordinary=validation[0]
        for row in validation[3:]:
            a=ordinary['metrics'];b=row['metrics']
            error=max(abs(x-y) for x,y in zip(a['body_range_ratios'],b['body_range_ratios']))
            amp=True
            for condition in ('sham','inhibited'):
                for aa,bb in zip(ordinary['features'][condition]['active'],row['features'][condition]['active']):
                    x=np.asarray(aa['regional_cycles_rad']);y=np.asarray(bb['regional_cycles_rad'])
                    amp=amp and bool(np.all(abs(x-y)<=np.maximum(1e-5,.1*x)))
            convergence.append(dict(grid=row['grid'],suppression_ratio_difference=error,active_amplitudes_consistent=amp,passed=bool(error<=.05 and amp)))
    passed=bool(complete and selected and all(r['checkpoint_sha256']==selected['checkpoint_sha256'] and r['metrics']['avb_component_passed'] for r in validation)
                and len(convergence)==2 and all(r['passed'] for r in convergence))
    write(out/'summary.json',dict(experiment='E078',execution_valid=failure is None and stop_reason!='evaluation_failure',
        all_stages_finished=complete,validation_complete=complete,completed_generations=generation,
        stop_reason=stop_reason,failure=failure,baseline=baseline,selected=selected,validation=validation,
        numerical_checks=convergence,avb_component_qualified=passed,
        sensory_start_stop_qualified=False,swimming_qualified=False,
        scope='Fitted circuit response to imposed AVB inhibition; no learned sensory start/stop or propulsion claim'))
    print('AVB_TRAINING_REPORT:',out/'summary.json',flush=True)


if __name__=='__main__':main()
