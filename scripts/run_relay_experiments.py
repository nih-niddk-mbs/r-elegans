"""E092 matched prior-cap experiments, within an eight-hour total budget."""
import argparse,concurrent.futures,json,multiprocessing,os,random,shutil,time
from pathlib import Path
import numpy as np
from scripts.experiment_records import capture_run_record,file_sha256
from scripts.run_cma_sensorimotor import optimizer_snapshot
from scripts.rial_sensory_worker import setup
from scripts.relay_cap_worker import initialize,evaluate
from r_elegans.relay_range import parameter_groups
from r_elegans.relay_cap_experiments import CAPS,SEEDS,FACTORS,TARGET,edge_coordinate,upper_for,eligible,panel_assessment

PARENT='0020499c6b900183e954a0e577563584c7ffa19e9cd5264f90c584d8b5b116da'


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def new_optimizer(cma,dimension,seed,mean=None,sigma=.3):
    # Per-arm RNG: initializing another optimizer cannot change this arm's stream.
    rng=np.random.RandomState(seed)
    return cma.CMAEvolutionStrategy(np.zeros(dimension) if mean is None else mean,sigma,
        dict(seed=seed,randn=rng.randn,popsize=12,bounds=[-1,1],verbose=-9,verb_log=0))


def prepare(source,out):
    from scripts.compare_rhythm_worker import verify_inputs
    s=json.loads((source/'summary.json').read_text());receipt=json.loads((source/'budget_receipt.json').read_text())
    if not s.get('execution_valid') or s.get('objective')!='relay-range' or s.get('experiment')!='E091' or s.get('selected',{}).get('checkpoint_sha256')!=PARENT or not s.get('validation_complete') or not all(receipt.get(k) for k in ('all_steps_finished','audit_complete','audit_integrity_passed')):
        raise ValueError('Pinned completed, audited E091 required')
    row=next(r for r in s['validation'] if (r['variant'],r['grid'],r['mode'])==('nominal','ordinary','normal'))
    folder=Path(row['output']).resolve()
    if source not in folder.parents or file_sha256(folder/'network.npz')!=PARENT:raise ValueError('Parent model mismatch')
    verify_inputs(source/'inputs');inputs=out/'inputs';shutil.copytree(source/'inputs',inputs)
    manifest=json.loads((inputs/'manifest.json').read_text())
    for name,path in [('sensory_parent.npz',folder/'network.npz'),('sensory_parent_coordinates.npz',folder/'coordinates.npz'),('sensory_parent_summary.json',source/'summary.json'),('source_bounds.npz',source/'parameter_origin.npz')]:
        sha=file_sha256(path);shutil.copy2(path,inputs/name)
        if file_sha256(inputs/name)!=sha or file_sha256(path)!=sha:raise ValueError('Input changed while copying')
        manifest['sha256'][name]=sha;manifest['sources'][name]=str(path)
    reference=folder/'no_cue.npz'
    if file_sha256(reference)!=row['trajectory_sha256']['no_cue']:raise ValueError('Source trace mismatch')
    write(inputs/'sensory_reference.json',dict(path=str(reference),sha256=row['trajectory_sha256']['no_cue']))
    manifest['sha256']['sensory_reference.json']=file_sha256(inputs/'sensory_reference.json')
    manifest['sources']['sensory_reference.json']=str(reference)
    manifest['consumer_experiment']='E092';write(inputs/'manifest.json',manifest)
    d=setup(inputs);edge=edge_coordinate(d['spec'],d['model'].neuron_ids)
    with np.load(inputs/'source_bounds.npz',allow_pickle=False) as z:
        if not np.array_equal(d['lower'],z['lower']) or not np.array_equal(d['upper'],z['upper']):raise ValueError('Original bounds changed')
    np.savez_compressed(inputs/'cap_prior.npz',origin=d['origin'],lower=d['lower'],upper=d['upper'],edge=edge)
    schemas={cap:parameter_groups(d['spec'],d['model'].neuron_ids,d['origin'],d['lower'],upper_for(cap,d['upper'],edge)) for cap in CAPS}
    if [g['name'] for g in schemas[CAPS[0]]]!=[g['name'] for g in schemas[CAPS[1]]]:raise ValueError('Cap policies changed coordinate coverage')
    write(inputs/'cap_groups.json',schemas)
    for name in ('cap_prior.npz','cap_groups.json'):
        manifest['sha256'][name]=file_sha256(inputs/name);manifest['sources'][name]='generated E092 experimental design'
    write(inputs/'manifest.json',manifest);verify_inputs(inputs)
    return inputs,d,schemas,edge


def main():
    import cma
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--source-run',type=Path,required=True);p.add_argument('--generation-limit',type=int)
    a=p.parse_args();out=a.output_dir.resolve();source=a.source_run.resolve();repo=Path(__file__).resolve().parents[1]
    if any(out==root or root in out.parents or out in root.parents for root in (source,repo)):raise ValueError('Separate external output required')
    if a.generation_limit is not None and a.generation_limit<1:p.error('Positive generation limit required')
    from importlib.metadata import version
    if version('cma')!='4.4.1':raise ValueError('Use existing dedicated environment with cma==4.4.1')
    capture_run_record(out,repo,a)
    budget=json.loads((out/'deadline.json').read_text());finish=budget['coordinator_finish_by'];train_until=finish-120*60
    inputs,d,schemas,edge=prepare(source,out)
    from scripts.benchmark_cpu import physical_core_cpus
    from scripts.benchmark_cpu_parallel import worker_cpu_groups
    cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0)))
    if not known or len(cpus)<16:raise ValueError('At least16known physical cores required')
    workers=8 if len(cpus)>=32 else 4;cpugroups=worker_cpu_groups(cpus,workers,4)
    dimension=len(schemas['original']);zero=np.zeros(dimension);all_results=[];scan=[];scan_controls=[];panels=[];parent=None
    failure=None;generation=0;restarts=0;history=[];stop='search_time_budget';last_progress=time.monotonic()
    random.seed(9200);np.random.seed(9200)
    write(out/'configuration.json',dict(experiment='E092',parent_sha256=PARENT,workers=workers,cpu_groups=cpugroups,
        caps=dict(original=2.,aiyr_2p5=2.5),only_extended_coordinate=edge,seeds=SEEDS,scan_factors=FACTORS,
        target=TARGET,training_variant='redundant',response_aggregation='weakest',population=12,
        comparison='Equal complete cohorts, paired seeds, independent per-arm random streams',
        raw_retention='Parent; selected single-edge scans/refinements; two cap champions and optional scan champion final panels. Other traces deliberately metrics-only.',
        diagnostic_only=True,full_start_stop_qualified=False,automatic_parent_promotion=False))
    arms={}
    for cap in CAPS:
        for seed in SEEDS:
            name=f'{cap}-seed-{seed}';es=new_optimizer(cma,dimension,seed)
            arms[name]=dict(cap=cap,seed=seed,es=es,primary=None,all_best=None,z_primary=zero.copy(),z_all=zero.copy())
            optimizer_snapshot(out/'optimizer'/name/'initial',es,0,0)
    def job(z,name,cap,kind,until,variant='redundant',grid='ordinary',mode='normal',retain=False,reference=False,margin=1.3):
        return dict(z=list(map(float,z)),output=str(out/'evaluations'/name),cap=cap,kind=kind,until=until,
            objective='relay-range',response_aggregation='weakest',target=TARGET,margin=margin,
            variant=variant,grid=grid,mode=mode,retain_traces=retain,reference=reference,parent_features=parent)
    def batch(pool,jobs,phase):
        nonlocal last_progress
        futures={pool.submit(evaluate,j):j for j in jobs};rows={}
        while futures:
            ready,_=concurrent.futures.wait(futures,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in ready:
                j=futures.pop(f)
                try:r=f.result()
                except Exception as e:r=dict(status='worker_error',error=repr(e),output=j['output'])
                rows[j['output']]=r;all_results.append(r)
                print(json.dumps(dict(phase=phase,case=Path(j['output']).name,status=r['status'],metrics=r.get('metrics'),error=r.get('error'))),flush=True)
            if time.monotonic()-last_progress>30:
                print(json.dumps(dict(phase=phase,generation=generation,pending=len(futures),remaining_total_minutes=(finish-time.monotonic())/60)),flush=True);last_progress=time.monotonic()
        return [rows[j['output']] for j in jobs]
    ctx=multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=initialize,
            initargs=(str(inputs),schemas,cpugroups,str(out/'workers'))) as pool:
        try:
            baseline=batch(pool,[job(zero,'parent','original','reference',train_until,variant='nominal',retain=True,reference=True)],'baseline')[0]
            if baseline['status']!='completed' or baseline['checkpoint_sha256']!=PARENT:raise ValueError('Pinned baseline reproduction failed')
            parent=baseline['paired_features']['no_cue'];write(out/'parent_features.json',parent)
            jobs=[]
            for factor in FACTORS:
                cap='original' if np.log(factor)<=d['upper'][edge]-d['origin'][edge]+1e-12 else 'aiyr_2p5'
                z=zero.copy();g=next((i,g) for i,g in enumerate(schemas[cap]) if g['indices']==[edge]);delta=np.log(factor)
                z[g[0]]=delta/(g[1]['upper'] if delta>=0 else -g[1]['lower'])
                jobs.append(job(z,f'factor-{factor:g}',cap,'single_edge',train_until,retain=factor in (1.,1.10,1.25)))
            scan=batch(pool,jobs,'single-edge-scan')
            control=scan[FACTORS.index(1.)]
            if control['status']!='completed' or control['checkpoint_sha256']!=PARENT:raise ValueError('Unit-factor control failed')
            for arm in arms.values():arm.update(primary=control,all_best=control)
            max_case=scan[-1];boundary=[]
            if max_case['status']=='completed':
                with np.load(Path(max_case['output'])/'coordinates.npz',allow_pickle=False) as z:max_z=z['z']
                for label,z,cap in [('control',zero,'original'),('max',max_z,'aiyr_2p5')]:
                    for grid in ('neural-half','body-half'):
                        boundary.append(job(z,f'scan-{label}-{grid}',cap,'single_edge',train_until,grid=grid,retain=True))
                boundary.append(job(max_z,'scan-max-blocked','aiyr_2p5','single_edge',train_until,mode='sensory-blocked',retain=True))
            if boundary:scan_controls=batch(pool,boundary,'scan-controls')
            while time.monotonic()<train_until and (a.generation_limit is None or generation<a.generation_limit):
                if shutil.disk_usage(out).free<50*1024**3 or sum(f.stat().st_size for f in out.rglob('*') if f.is_file())>15*1024**3:
                    stop='disk_budget';break
                points={name:arm['es'].ask() for name,arm in arms.items()}
                for name,arm in arms.items():optimizer_snapshot(out/'optimizer'/name/f'g-{generation:05d}-asked',arm['es'],0,generation,[list(map(float,x)) for x in points[name]])
                names=list(arms);jobs=[job(points[name][i],f'g-{generation:05d}-{name}-c-{i:02d}',arms[name]['cap'],'fit',train_until) for i in range(12) for name in names]
                rows=batch(pool,jobs,'matched-optimization');by_output={r['output']:r for r in rows}
                complete=not any(r['status'] in ('time_budget','disk_budget','worker_error') for r in rows)
                for name,arm in arms.items():
                    arm_rows=[by_output[str(out/'evaluations'/f'g-{generation:05d}-{name}-c-{i:02d}')] for i in range(12)]
                    for z,r in zip(points[name],arm_rows):
                        if eligible(r) and r['metrics']['loss']<arm['all_best']['metrics']['loss']:
                            arm['all_best']=r;arm['z_all']=np.asarray(z).copy()
                    if complete:
                        arm['es'].tell(points[name],[r['metrics']['loss'] if r['status']=='completed' else 1e6 for r in arm_rows])
                        arm['primary']=arm['all_best'];arm['z_primary']=arm['z_all'].copy()
                        optimizer_snapshot(out/'optimizer'/name/f'g-{generation:05d}-told',arm['es'],0,generation)
                if not complete:
                    write(out/'partial_cohort.json',dict(generation=generation,results=rows,primary_comparison_uses_only_prior_complete_cohorts=True))
                    stop='search_time_budget' if all(r['status'] not in ('disk_budget','worker_error') for r in rows) else 'evaluation_failure';break
                generation+=1
                history.append(dict(generation=generation,arms={name:dict(loss=a['primary']['metrics']['loss'],checkpoint_sha256=a['primary']['checkpoint_sha256'],output=a['primary']['output']) for name,a in arms.items()}))
                write(out/'generation_history.json',history)
                if generation%8==0 or any(a['es'].stop() for a in arms.values()):
                    restarts+=1
                    for name,arm in arms.items():
                        arm['es']=new_optimizer(cma,dimension,arm['seed']+1000*restarts,arm['z_primary'],.15)
                        optimizer_snapshot(out/'optimizer'/name/f'restart-{restarts}',arm['es'],0,generation)
            for name,arm in arms.items():optimizer_snapshot(out/'optimizer'/name/'final',arm['es'],0,generation,
                [list(map(float,x)) for x in points[name]] if 'points' in locals() and stop in ('search_time_budget','evaluation_failure') and (out/'partial_cohort.json').exists() else None)
            # Pick raw-trace champions using training data only, before final tests.
            champions={min((a for a in arms.values() if a['cap']==cap),key=lambda a:a['primary']['metrics']['loss'])['primary']['checkpoint_sha256'] for cap in CAPS}
            profiles={}
            def profile(row,cap,label,raw):
                sha=row['checkpoint_sha256']
                if sha not in profiles:profiles[sha]=dict(row=row,cap=cap,labels=[],retain=raw)
                profiles[sha]['labels'].append(label);profiles[sha]['retain']|=raw
            for name,arm in arms.items():profile(arm['primary'],arm['cap'],name,arm['primary']['checkpoint_sha256'] in champions)
            good_scan=[r for r in scan if eligible(r)]
            if good_scan:
                best_scan=min(good_scan,key=lambda r:r['metrics']['loss']);profile(best_scan,best_scan['cap'],'best-single-edge',True)
            exploratory=min((a for a in arms.values()),key=lambda a:a['all_best']['metrics']['loss'])
            profile(exploratory['all_best'],exploratory['cap'],'best-all-completed-exploratory',False)
            variants=[('nominal','ordinary','normal'),('low-dose','ordinary','normal'),('shift','ordinary','normal'),
                ('repeat','ordinary','normal'),('redundant','ordinary','normal'),('long-cue','ordinary','normal'),
                ('nominal','neural-half','normal'),('nominal','body-half','normal'),('nominal','ordinary','sensory-blocked'),('nominal','ordinary','relay-replay')]
            final_jobs=[];ordered=list(profiles.items())
            for i,(sha,pr) in enumerate(ordered):
                with np.load(Path(pr['row']['output'])/'coordinates.npz',allow_pickle=False) as z:point=z['z']
                for j,(v,g,m) in enumerate(variants):final_jobs.append(job(point,f'final-p{i:02d}-{j:02d}',pr['cap'],'final',finish-90,v,g,m,pr['retain'],margin=1.))
            final=batch(pool,final_jobs,'final-validation')
            for i,(sha,pr) in enumerate(ordered):
                rows=final[i*10:(i+1)*10]
                panels.append(dict(checkpoint_sha256=sha,cap=pr['cap'],labels=pr['labels'],training_output=pr['row']['output'],
                    full_raw_retained=pr['retain'],validation=rows,assessment=panel_assessment(rows)))
        except Exception as e:
            failure=repr(e);print('E092_ERROR:',failure,flush=True)
    for name,arm in arms.items():
        if not (out/'optimizer'/name/'final').exists():
            optimizer_snapshot(out/'optimizer'/name/'final',arm['es'],0,generation,
                [list(map(float,x)) for x in points[name]] if 'points' in locals() else None)
    complete=bool(panels and all(p['assessment']['complete'] for p in panels))
    write(out/'summary.json',dict(experiment='E092',execution_valid=failure is None and stop!='evaluation_failure',
        validation_complete=complete,all_stages_finished=bool(complete and len(scan)==len(FACTORS) and len(scan_controls)==5 and all(r['status']=='completed' for r in scan+scan_controls)),stop_reason=stop,failure=failure,
        completed_matched_generations=generation,scan=scan,scan_controls=scan_controls,panels=panels,
        arms={name:dict(cap=a['cap'],seed=a['seed'],primary=a['primary'],best_all_completed=a['all_best']) for name,a in arms.items()},
        evaluation_count=len(all_results),sensory_relay_range_component_qualified=False,full_start_stop_qualified=False,
        scope='Diagnostic prior comparison; expanded-prior models explicitly labeled; no automatic fitting-parent or biological promotion'))
    print('E092_REPORT:',out/'summary.json',flush=True)
    return 0 if failure is None else 2


if __name__=='__main__':raise SystemExit(main())
