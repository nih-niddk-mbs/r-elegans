"""E096 imposed-conductance dose-response at single nodes (diagnostic, no fitting)."""
import argparse,concurrent.futures,json,multiprocessing,os,time
from pathlib import Path
import numpy as np
from scripts.experiment_records import capture_run_record,file_sha256
from scripts.run_relay_experiments import write
from scripts.run_sensory_motion import prepare
from scripts.node_probe_worker import initialize,evaluate
from r_elegans.node_probe import cases,TARGET,RETAIN_DOSE,reduction_percent,reaches_target,minimum_doses
from r_elegans.sensory_motion import score

MAX_WORKERS=21


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--source-run',type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();source=a.source_run.resolve();repo=Path(__file__).resolve().parents[1]
    if any(out==r or r in out.parents or out in r.parents for r in (source,repo)):raise ValueError('External separate output required')
    capture_run_record(out,repo,a);deadline=json.loads((out/'deadline.json').read_text());finish=deadline['coordinator_finish_by']
    inputs,d,_,edge,parent_sha,source_audit=prepare(source,out,'E096',False)
    from scripts.benchmark_cpu import physical_core_cpus
    from scripts.benchmark_cpu_parallel import worker_cpu_groups
    cpus,known=physical_core_cpus(sorted(os.sched_getaffinity(0)))
    if not known or len(cpus)<16:raise ValueError('Known topology and >=16physical cores required')
    workers=min(MAX_WORKERS,len(cpus)//4);cpugroups=worker_cpu_groups(cpus,workers,4);planned=cases()
    write(out/'configuration.json',dict(experiment='E096',parent_sha256=parent_sha,parent_filename='motion_parent.npz',source_audit=source_audit,
        only_extended_coordinate=edge,caps={'aiyr_2p5':2.5},workers=workers,cpu_groups=cpugroups,target=TARGET,variant='nominal',probe_window_s=[40.,50.],
        planned=planned,retain_dose_nS=RETAIN_DOSE,
        fixed='Parent weights and physics unchanged; the only change is an imposed membrane conductance on named neurons during the nominal cue window',
        qualification='Diagnostic dose-response only; no fitting, no component qualification, no learned stop, start/stop or swimming claim'))
    meta={};rows_all=[];failure=None
    def job(label,probe,retain):
        output=str(out/'evaluations'/label);meta[output]=dict(label=label,probe=probe)
        return dict(output=output,probe=probe,retain_traces=retain,until=finish-60)
    def batch(pool,jobs):
        futures={pool.submit(evaluate,j):j for j in jobs};results={};last=time.monotonic()
        while futures:
            ready,_=concurrent.futures.wait(futures,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in ready:
                j=futures.pop(f)
                try:r=f.result()
                except Exception as e:r=dict(status='worker_error',output=j['output'],error=repr(e),probe=j['probe'])
                results[j['output']]=r;rows_all.append(r)
                print(json.dumps(dict(case=Path(j['output']).name,status=r['status'],error=r.get('error'))),flush=True)
            if time.monotonic()-last>30:
                print(json.dumps(dict(pending=len(futures),remaining_total_minutes=(finish-time.monotonic())/60)),flush=True);last=time.monotonic()
        return [results[j['output']] for j in jobs]
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,
            initargs=(str(inputs),cpugroups,str(out/'workers'))) as pool:
        try:
            jobs=[job('baseline',None,True)]+[job(c['name'],c,c['conductance_nS']==RETAIN_DOSE) for c in planned]
            results=batch(pool,jobs)
        except Exception as e:failure=repr(e);print('E096_ERROR:',failure,flush=True)
    baseline=results[0] if results else None;table=[]
    if baseline and baseline['status']!='completed':failure=failure or 'Baseline did not complete'
    for c,r in zip(planned,results[1:]):
        row=dict(output=r['output'],name=c['name'],group=c['group'],sign='excite' if c['reversal_mV']==0 else 'inhibit',dose_nS=c['conductance_nS'],status=r['status'])
        if r['status']=='completed' and baseline and baseline['status']=='completed':
            m=score(dict(no_cue=baseline['features'],cue=r['features']),baseline['features'],TARGET,1.)
            row.update(reduction_percent=reduction_percent(m),regional_motion_ratios=m['regional_motion_ratios'][0],safe=m['safe'],
                baseline_and_recovery_preserved=m['baseline_and_recovery_preserved'],regional_cycle_ratios=m['regional_cycle_ratios'][0],target_reached=reaches_target(m))
        table.append(row)
    complete=bool(failure is None and baseline and len(results)==1+len(planned) and all(r['status']=='completed' for r in results))
    write(out/'summary.json',dict(experiment='E096',execution_valid=failure is None,failure=failure,stop_reason='complete' if complete else 'incomplete',
        validation_complete=complete,all_stages_finished=complete,completed_generations=0,target=TARGET,baseline=baseline,probes=results[1:],probe_table=table,
        minimum_doses_reaching_target_nS=minimum_doses(table),evaluation_count=len(rows_all),panels=[],arms={},scan=[],
        sensory_motion_component_qualified=False,rest_restart_component_qualified=False,full_start_stop_qualified=False,swimming_qualified=False,
        scope='Imposed-conductance dose-response diagnostic; no fitting, qualification, learned stop or biological claim'))
    print('E096_REPORT:',out/'summary.json',flush=True);return 0 if failure is None else 2


if __name__=='__main__':raise SystemExit(main())
