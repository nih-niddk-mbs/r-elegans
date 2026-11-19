"""E093/E094 four-hour total sensory movement-suppression curriculum."""
import argparse,concurrent.futures,json,multiprocessing,os,shutil,time
from pathlib import Path
import numpy as np
from scripts.experiment_records import capture_run_record,file_sha256
from scripts.run_relay_experiments import write,new_optimizer,PARENT as E091_PARENT
from scripts.run_cma_sensorimotor import optimizer_snapshot
from scripts.sensory_motion_worker import setup,initialize,evaluate
from r_elegans.relay_cap_experiments import edge_coordinate,eligible
from r_elegans.sensory_motion import RUNGS,parameter_groups,score,panel_assessment
from r_elegans.pathway import NODE_TARGET,pathway_groups,hop1_with_edges


def prepare(source,out,experiment='E093',bilateral=False,pathway=False):
    from scripts.compare_rhythm_worker import verify_inputs
    from scripts.report_relay_experiments import audit_path
    from scripts.audit_rial_sensory import frozen_network_fields
    s=json.loads((source/'summary.json').read_text());ap=audit_path(source)
    audit=json.loads(ap.read_text());config=json.loads((source/'configuration.json').read_text())
    if s.get('experiment')!='E092' or not s.get('execution_valid') or not s.get('validation_complete') or config.get('parent_sha256')!=E091_PARENT:
        raise ValueError('Pinned completed E092 source required')
    if not audit.get('audit_complete') or not audit.get('integrity_passed') or audit.get('problems'):
        raise ValueError('Latest corrected E092 audit must pass; no fallback')
    panels=[p for p in s['panels'] if 'best-single-edge' in p['labels']]
    if len(panels)!=1 or not panels[0]['assessment'].get('engineering_criteria_met'):raise ValueError('Unique passing isolated-edge panel required')
    panel=panels[0]
    if audit.get('panels')!=[dict(labels=p['labels'],cap=p['cap'],assessment=p['assessment']) for p in s['panels']]:raise ValueError('Audit/source panel mismatch')
    if file_sha256(source/'inputs/sensory_parent.npz')!=E091_PARENT:raise ValueError('E092 original parent hash mismatch')
    row=panel['validation'][0];folder=Path(row['output']).resolve()
    if source not in folder.parents or row['variant']!='nominal' or row['grid']!='ordinary' or row['mode']!='normal':raise ValueError('Invalid selected source reference')
    sha=panel['checkpoint_sha256']
    if row['checkpoint_sha256']!=sha or file_sha256(folder/'network.npz')!=sha:raise ValueError('Selected single-edge model hash mismatch')
    scan=next(r for r in s['scan'] if Path(r['output']).name=='factor-1.25')
    if scan['checkpoint_sha256']!=sha:raise ValueError('Parent must be the isolated 1.25 factor; no optimizer fallback')
    with np.load(source/'inputs/cap_prior.npz',allow_pickle=False) as z:
        old_origin,lower,upper,edge=z['origin'],z['lower'],z['upper'],int(z['edge'])
    with np.load(folder/'coordinates.npz',allow_pickle=False) as z:origin=z['offsets']
    delta=origin-old_origin;expected=np.zeros_like(delta);expected[edge]=np.log(1.25)
    if not np.allclose(delta,expected,rtol=0,atol=1e-12) or not frozen_network_fields(source/'inputs/sensory_parent.npz',folder/'network.npz',delta!=0):
        raise ValueError('Single-edge source changed other parameters')
    evidence=json.loads((ap.parent/'evidence.json').read_text())
    hashes={str(Path(e['path']).resolve()):e['sha256'] for e in evidence}
    for name in ('network.npz','coordinates.npz','no_cue.npz'):
        path=folder/name
        if hashes.get(str(path))!=file_sha256(path):raise ValueError('Selected source differs from corrected audit evidence: '+name)
    verify_inputs(source/'inputs');inputs=out/'inputs';shutil.copytree(source/'inputs',inputs)
    manifest=json.loads((inputs/'manifest.json').read_text())
    paths=[('motion_parent.npz',folder/'network.npz'),('motion_parent_coordinates.npz',folder/'coordinates.npz'),
        ('motion_parent_summary.json',source/'summary.json'),('motion_parent_audit.json',ap),('motion_parent_audit_evidence.json',ap.parent/'evidence.json')]
    for name,path in paths:
        digest=file_sha256(path);shutil.copy2(path,inputs/name)
        if file_sha256(inputs/name)!=digest or file_sha256(path)!=digest:raise ValueError('Source changed during copy')
        manifest['sha256'][name]=digest;manifest['sources'][name]=str(path)
    reference=folder/'no_cue.npz'
    if file_sha256(reference)!=row['trajectory_sha256']['no_cue']:raise ValueError('Parent no-cue trace mismatch')
    write(inputs/'sensory_reference.json',dict(path=str(reference),sha256=row['trajectory_sha256']['no_cue']))
    np.savez_compressed(inputs/'cap_prior.npz',origin=origin,lower=lower,upper=upper,edge=edge)
    for name in ('sensory_reference.json','cap_prior.npz'):
        manifest['sha256'][name]=file_sha256(inputs/name);manifest['sources'][name]=str(reference) if name=='sensory_reference.json' else experiment+' selected E092 coordinates; unchanged E092 original bounds and one-edge policy'
    manifest['consumer_experiment']=experiment;write(inputs/'manifest.json',manifest)
    d=setup(inputs)
    if edge_coordinate(d['spec'],d['model'].neuron_ids)!=edge:raise ValueError('Prior exception edge changed')
    schemas={'pathway':pathway_groups(d['spec'],d['model'].neuron_ids,origin,d['lower'],d['upper'])} if pathway else {'aiyr_2p5':parameter_groups(d['spec'],d['model'].neuron_ids,origin,d['lower'],d['upper'],bilateral=bilateral)}
    write(inputs/'cap_groups.json',schemas);manifest['sha256']['cap_groups.json']=file_sha256(inputs/'cap_groups.json')
    manifest['sources']['cap_groups.json']=experiment+' focused sensory/relay output groups';write(inputs/'manifest.json',manifest);verify_inputs(inputs)
    return inputs,d,schemas,edge,sha,str(ap)


def main():
    import cma
    from importlib.metadata import version
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--source-run',type=Path,required=True);p.add_argument('--generation-limit',type=int)
    p.add_argument('--profile',choices=('e093','e094','e098'),default='e093',help='e094: bilateral sensory/relay groups and worst-region loss term; e098: cue-to-RIA-to-motor pathway with a RIA node-level term')
    a=p.parse_args();out=a.output_dir.resolve();source=a.source_run.resolve();repo=Path(__file__).resolve().parents[1]
    experiment=a.profile.upper();bilateral=worst_region=a.profile in ('e094','e098');pathway=a.profile=='e098'
    node_target=NODE_TARGET if pathway else None;seed=9800 if pathway else 9300;population=21 if pathway else 12;capname='pathway' if pathway else 'aiyr_2p5'
    if any(out==r or r in out.parents or out in r.parents for r in (source,repo)):raise ValueError('External separate output required')
    if a.generation_limit is not None and a.generation_limit<1:p.error('Positive generation limit required')
    if version('cma')!='4.4.1':raise ValueError('Dedicated cma==4.4.1 required')
    capture_run_record(out,repo,a);deadline=json.loads((out/'deadline.json').read_text());finish=deadline['coordinator_finish_by'];fit_until=finish-(30 if pathway else 70)*60
    inputs,d,schemas,edge,parent_sha,source_audit=prepare(source,out,experiment,bilateral,pathway)
    from scripts.benchmark_cpu import physical_core_cpus
    from scripts.benchmark_cpu_parallel import worker_cpu_groups
    cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0)))
    if not known or len(cpus)<16:raise ValueError('Known topology and >=16physical cores required')
    workers=min(21,len(cpus)//4) if pathway else 8 if len(cpus)>=32 else 4;cpugroups=worker_cpu_groups(cpus,workers,4)
    hop1=hop1_with_edges(d['spec'],d['model'].neuron_ids) if pathway else None
    write(out/'configuration.json',dict(experiment=experiment,bilateral=bilateral,worst_region_term=worst_region,pathway=pathway,node_target=node_target,pathway_hop1=hop1,parent_sha256=parent_sha,parent_filename='motion_parent.npz',
        source_audit=source_audit,only_extended_coordinate=edge,caps={'aiyr_2p5':2.5},seed=seed,workers=workers,cpu_groups=cpugroups,
        population=population,rungs=RUNGS,training_variant='redundant',initial_sigma=.3,training_margin=1.05,
        prior='Same E092 exception: only AIYR->AIYL original upper factor2 becomes2.5; all other bounds fixed',
        objective='Cue-relative regional joint angular-speed RMS reduction; preserve no-cue/recovery gait and relay component',
        fixed='Water mechanics, sensory gains, NMJs, topology/signs, fixed electrophysiology, other neural parameters',
        qualification='No initial sensory initiation from rest tested; no full start/stop or swimming promotion'))
    dimension=len(schemas.get('pathway') or schemas['aiyr_2p5']);zero=np.zeros(dimension);zbest=zero.copy();parent=None;best=None;baseline=None
    stage=0;generation=0;restart=0;rows_all=[];validation=[];history=[];winners=[];failure=None;stop='search_time_budget';pending=None
    es=new_optimizer(cma,dimension,seed,popsize=population);name=f'motion-seed-{seed}'
    optimizer_snapshot(out/'optimizer'/name/'initial',es,stage,generation)
    def job(z,label,until,variant='redundant',grid='ordinary',mode='normal',retain=False,reference=False,kind='fit',target=None):
        return dict(z=list(map(float,z)),output=str(out/'evaluations'/label),cap=capname,kind=kind,objective='sensory-motion',
            until=until,variant=variant,grid=grid,mode=mode,retain_traces=retain,reference=reference,bilateral=bilateral,worst_region=worst_region,
            **(dict(node_target=node_target,pathway_hop1=hop1) if pathway else {}),
            target=RUNGS[stage] if target is None else target,margin=1. if kind=='final' else 1.05,parent_features=parent)
    def batch(pool,jobs,phase):
        futures={pool.submit(evaluate,j):j for j in jobs};results={};last=time.monotonic()
        while futures:
            ready,_=concurrent.futures.wait(futures,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in ready:
                j=futures.pop(f)
                try:r=f.result()
                except Exception as e:r=dict(status='worker_error',output=j['output'],error=repr(e))
                results[j['output']]=r;rows_all.append(r)
                print(json.dumps(dict(phase=phase,case=Path(j['output']).name,status=r['status'],metrics=r.get('metrics'),error=r.get('error'))),flush=True)
            if time.monotonic()-last>30:
                print(json.dumps(dict(phase=phase,generation=generation,target=RUNGS[stage],pending=len(futures),remaining_total_minutes=(finish-time.monotonic())/60)),flush=True);last=time.monotonic()
        return [results[j['output']] for j in jobs]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,
            initargs=(str(inputs),schemas,cpugroups,str(out/'workers'))) as pool:
        try:
            baseline=batch(pool,[job(zero,'parent',fit_until,variant='nominal',retain=True,reference=True,kind='reference')],'parent-reference')[0]
            if baseline['status']!='completed' or baseline['checkpoint_sha256']!=parent_sha:raise ValueError('Pinned parent reproduction failed')
            parent=baseline['paired_features']['no_cue'];write(out/'parent_features.json',parent)
            best=batch(pool,[job(zero,'parent-redundant',fit_until,retain=True,kind='reference')],'parent-redundant')[0]
            if not eligible(best) or not best['metrics']['relay_component_preserved']:raise ValueError('Parent movement or relay preservation failed')
            while time.monotonic()<fit_until and (a.generation_limit is None or generation<a.generation_limit):
                if shutil.disk_usage(out).free<50*1024**3 or sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>15*1024**3:stop='disk_budget';break
                points=es.ask();pending=[list(map(float,x)) for x in points]
                optimizer_snapshot(out/'optimizer'/name/f'g-{generation:05d}-asked',es,stage,generation,pending)
                rows=batch(pool,[job(z,f'g-{generation:05d}-c-{i:02d}',fit_until) for i,z in enumerate(points)],'optimization')
                current=score(best['paired_features'],parent,RUNGS[stage],1.05,worst_region,node_target)['loss']
                for z,r in zip(points,rows):
                    if eligible(r) and r['metrics']['relay_component_preserved'] and r['metrics']['loss']<current:
                        best=r;zbest=np.asarray(z).copy();current=r['metrics']['loss']
                complete=all(r['status'] not in ('time_budget','disk_budget','worker_error') for r in rows)
                if not complete:
                    write(out/'partial_cohort.json',dict(generation=generation,results=rows,selection='Completed eligible points retained; incomplete population never told'))
                    stop='search_time_budget' if all(r['status'] not in ('disk_budget','worker_error') for r in rows) else 'evaluation_failure';break
                es.tell(points,[r['metrics']['loss'] if r['status']=='completed' else 1e6 for r in rows]);pending=None
                optimizer_snapshot(out/'optimizer'/name/f'g-{generation:05d}-told',es,stage,generation);generation+=1
                m=score(best['paired_features'],parent,RUNGS[stage],1.05,worst_region,node_target)
                history.append(dict(generation=generation,target=RUNGS[stage],selected=best['checkpoint_sha256'],selected_evaluation=best['output'],rescored_loss=m['loss']))
                write(out/'generation_history.json',history)
                advance=m['training_margin_met'] and stage<len(RUNGS)-1
                if advance:
                    winners.append(dict(target=RUNGS[stage],output=best['output'],checkpoint_sha256=best['checkpoint_sha256']))
                    write(out/'stage_winners.json',winners);stage+=1
                if advance or es.stop() or generation%8==0:
                    restart+=1;es=new_optimizer(cma,dimension,seed+restart,zbest,.15,popsize=population)
                    optimizer_snapshot(out/'optimizer'/name/f'restart-{restart}',es,stage,generation)
            optimizer_snapshot(out/'optimizer'/name/'final',es,stage,generation,pending)
            # Validate the highest motion rung actually achieved in training; if none,
            # assess the first rung and report failure without weakening criteria.
            attained=[t for t in RUNGS if score(best['paired_features'],parent,t,1.)['partial_attenuation_passed']]
            target=max(attained) if attained else RUNGS[0]
            variants=[('nominal','ordinary','normal'),('low-dose','ordinary','normal'),('shift','ordinary','normal'),
                ('repeat','ordinary','normal'),('redundant','ordinary','normal'),('long-cue','ordinary','normal'),
                ('nominal','neural-half','normal'),('nominal','body-half','normal'),('nominal','ordinary','sensory-blocked'),
                ('nominal','ordinary','relay-replay'),('nominal','ordinary','rial-replay')]
            validation=batch(pool,[job(zbest,f'final-{i:02d}',finish-90,v,g,m,True,kind='final',target=target) for i,(v,g,m) in enumerate(variants)],'final-validation')
        except Exception as e:failure=repr(e);print(experiment+'_ERROR:',failure,flush=True)
    if not (out/'optimizer'/name/'final').exists():optimizer_snapshot(out/'optimizer'/name/'final',es,stage,generation,pending)
    target=target if 'target' in locals() else RUNGS[0];assessment=panel_assessment(validation,target)
    panels=[dict(checkpoint_sha256=best['checkpoint_sha256'],cap=capname,labels=[name],training_output=best['output'],
        full_raw_retained=True,target=target,validation=validation,assessment=assessment)] if best else []
    write(out/'summary.json',dict(experiment=experiment,execution_valid=failure is None and stop!='evaluation_failure',failure=failure,stop_reason=stop,
        validation_complete=assessment['complete'],all_stages_finished=assessment['complete'],completed_generations=generation,
        active_stage=stage,target=target,scan=[],scan_controls=[],panels=panels,evaluation_count=len(rows_all),
        arms={name:dict(seed=seed,cap=capname,primary=best,best_all_completed=best)},
        sensory_motion_component_qualified=assessment.get('engineering_criteria_met',False),
        rest_restart_component_qualified=assessment.get('rest_restart_component_qualified',False),full_start_stop_qualified=False,swimming_qualified=False))
    print(experiment+'_REPORT:',out/'summary.json',flush=True);return 0 if failure is None else 2


if __name__=='__main__':raise SystemExit(main())
