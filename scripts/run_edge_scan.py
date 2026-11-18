"""E095 single-edge strength dose-response of cue-on motion suppression (diagnostic, no fitting)."""
import argparse,concurrent.futures,json,multiprocessing,os,time
from pathlib import Path
import numpy as np
from scripts.experiment_records import capture_run_record,file_sha256
from scripts.run_relay_experiments import write
from scripts.run_sensory_motion import prepare
from scripts.sensory_motion_worker import initialize,evaluate
from r_elegans.edge_scan import FACTORS,MAX_FACTOR,TARGET,EDGES,FINALISTS,scan_groups,planned_cases,eligible,finalists,reduction_percent

CONTROLS=(('ordinary','normal'),('neural-half','normal'),('body-half','normal'),('ordinary','sensory-blocked'))


def main():
    from scripts.compare_rhythm_worker import verify_inputs
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--source-run',type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();source=a.source_run.resolve();repo=Path(__file__).resolve().parents[1]
    if any(out==r or r in out.parents or out in r.parents for r in (source,repo)):raise ValueError('External separate output required')
    capture_run_record(out,repo,a);deadline=json.loads((out/'deadline.json').read_text());finish=deadline['coordinator_finish_by'];scan_until=finish-25*60
    inputs,d,_,edge,parent_sha,source_audit=prepare(source,out,'E095',False)
    with np.load(inputs/'cap_prior.npz',allow_pickle=False) as z:origin,lower,upper=z['origin'],z['lower'],z['upper']
    groups=scan_groups(d['spec'],d['model'].neuron_ids,origin,lower,upper)
    write(inputs/'cap_groups.json',dict(scan=groups))
    write(inputs/'scan_prior.json',dict(edges=[f'{a}->{b}' for a,b in EDGES],factors=FACTORS,max_factor=MAX_FACTOR,
        note='Engineering prior: scanned edges may exceed their original upper bound up to max_factor times the parent strength; lower bounds are never widened.'))
    manifest=json.loads((inputs/'manifest.json').read_text())
    for name,text in (('cap_groups.json','E095 single-edge scan groups'),('scan_prior.json','E095 declared widened-prior policy')):
        manifest['sha256'][name]=file_sha256(inputs/name);manifest['sources'][name]=text
    write(inputs/'manifest.json',manifest);verify_inputs(inputs)
    from scripts.benchmark_cpu import physical_core_cpus
    from scripts.benchmark_cpu_parallel import worker_cpu_groups
    cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0)))
    if not known or len(cpus)<16:raise ValueError('Known topology and >=16physical cores required')
    workers=8 if len(cpus)>=32 else 4;cpugroups=worker_cpu_groups(cpus,workers,4)
    cases=planned_cases(groups)
    write(out/'configuration.json',dict(experiment='E095',parent_sha256=parent_sha,parent_filename='motion_parent.npz',source_audit=source_audit,
        only_extended_coordinate=edge,caps={'aiyr_2p5':2.5},workers=workers,cpu_groups=cpugroups,target=TARGET,variant='nominal',
        edges=[f'{a}->{b}' for a,b in EDGES],factors=FACTORS,max_factor=MAX_FACTOR,finalists=FINALISTS,planned=cases,
        prior='E092 one-edge exception (AIYR->AIYL) retained; each scanned edge may exceed its original upper bound up to max_factor x parent; lower bounds not widened',
        fixed='Everything except the single scanned chemical edge per evaluation; water mechanics, sensory gains, NMJs, topology/signs',
        qualification='Diagnostic dose-response only; no fitting, no component qualification, no start/stop or swimming claim'))
    rows_all=[];meta={};failure=None;stop='complete';baseline=None;parent=None;scan=[];controls=[];chosen=[]
    zero=[0.]*len(groups)
    def job(z,label,until,grid='ordinary',mode='normal',retain=False,reference=False,kind='scan',**extra):
        output=str(out/'evaluations'/label);meta[output]=dict(label=label,**extra)
        return dict(z=list(map(float,z)),output=output,cap='scan',kind=kind,objective='sensory-motion',until=until,variant='nominal',grid=grid,mode=mode,
            retain_traces=retain,reference=reference,bilateral=False,worst_region=False,target=TARGET,margin=1.,parent_features=parent,**extra)
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
                print(json.dumps(dict(phase=phase,pending=len(futures),remaining_total_minutes=(finish-time.monotonic())/60)),flush=True);last=time.monotonic()
        return [results[j['output']] for j in jobs]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,
            initargs=(str(inputs),dict(scan=groups),cpugroups,str(out/'workers'))) as pool:
        try:
            baseline=batch(pool,[job(zero,'parent',scan_until,retain=True,reference=True,kind='reference')],'parent-reference')[0]
            if baseline['status']!='completed' or baseline['checkpoint_sha256']!=parent_sha:raise ValueError('Pinned parent reproduction failed')
            parent=baseline['paired_features']['no_cue'];write(out/'parent_features.json',parent)
            runnable=[c for c in cases if c['runnable']]
            scan=batch(pool,[job(c['z'],f"scan-{c['group']:02d}-f{c['factor']:g}",scan_until,edge=c['edge'],factor=c['factor']) for c in runnable],'single-edge-scan')
            if any(r['status']=='disk_budget' for r in scan):stop='disk_budget'
            elif any(r['status'] in ('time_budget','worker_error','failed') for r in scan):stop='search_time_budget'
            chosen=finalists(scan)
            for i,row in enumerate(chosen):
                with np.load(Path(row['output'])/'coordinates.npz',allow_pickle=False) as zz:z=zz['z']
                m=meta[row['output']]
                controls+=batch(pool,[job(z,f'final-{i}-{k}',finish-90,grid=g,mode=md,retain=True,kind='final',edge=m['edge'],factor=m['factor'],source_scan=row['output'])
                    for k,(g,md) in enumerate(CONTROLS)],'finalist-controls')
        except Exception as e:failure=repr(e);print('E095_ERROR:',failure,flush=True)
    table=[]
    for r in scan:
        m=meta.get(r['output'],{});row=dict(output=r['output'],edge=m.get('edge'),factor=m.get('factor'),status=r['status'])
        if r['status']=='completed':
            x=r['metrics'];row.update(reduction_percent=reduction_percent(x),regional_motion_ratios=x['regional_motion_ratios'][0],safe=x['safe'],
                baseline_and_recovery_preserved=x['baseline_and_recovery_preserved'],relay_component_preserved=x['relay_component_preserved'],
                motion_target_reached=bool(eligible(r) and np.all(np.asarray(x['regional_motion_ratios'])<=1-TARGET)))
        table.append(row)
    checks=[]
    for i,row in enumerate(chosen):
        cs=controls[i*len(CONTROLS):(i+1)*len(CONTROLS)]
        if len(cs)<len(CONTROLS) or any(c['status']!='completed' for c in cs):checks.append(dict(scan=row['output'],complete=False));continue
        ratios=np.asarray(cs[0]['metrics']['regional_motion_ratios']);diffs=[float(np.max(abs(ratios-np.asarray(c['metrics']['regional_motion_ratios'])))) for c in cs[1:3]]
        blocked=cs[3]['metrics'];meta_row=meta[row['output']]
        checks.append(dict(scan=row['output'],edge=meta_row['edge'],factor=meta_row['factor'],complete=True,
            same_checkpoint=len({c['checkpoint_sha256'] for c in cs}|{row['checkpoint_sha256']})==1,half_grid_differences=diffs,half_grid_passed=all(x<=.25*TARGET for x in diffs),
            sensory_block_passed=bool(np.max(abs(np.asarray(blocked['regional_motion_ratios'])-1))<=.25*TARGET and max(abs(x) for x in blocked['relay_response_mean_changes'])<=5e-6),
            motion_target_reached=bool(eligible(cs[0]) and np.all(ratios<=1-TARGET))))
    complete=bool(failure is None and stop=='complete' and len(scan)==sum(c['runnable'] for c in cases) and all(r['status']=='completed' for r in scan+controls)
        and len(controls)==len(CONTROLS)*len(chosen))
    write(out/'summary.json',dict(experiment='E095',execution_valid=failure is None and stop!='disk_budget',failure=failure,stop_reason=stop,
        validation_complete=complete,all_stages_finished=complete,completed_generations=0,target=TARGET,baseline=baseline,scan=scan,scan_table=table,finalist_checks=checks,
        skipped=[dict(edge=c['edge'],factor=c['factor'],reason='below original lower bound; not widened') for c in cases if not c['runnable']],
        finalists=[r['output'] for r in chosen],controls=controls,panels=[],arms={},evaluation_count=len(rows_all),
        any_motion_target_reached=any(t.get('motion_target_reached') for t in table),
        sensory_motion_component_qualified=False,rest_restart_component_qualified=False,full_start_stop_qualified=False,swimming_qualified=False,
        scope='Single-edge dose-response diagnostic under an explicit engineering prior; no fitting, qualification, start/stop, swimming or biological claim'))
    print('E095_REPORT:',out/'summary.json',flush=True);return 0 if failure is None else 2


if __name__=='__main__':raise SystemExit(main())
