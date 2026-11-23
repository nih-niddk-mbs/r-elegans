"""E119 read-only two-radius bidirectional check of the rejected E118 step."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from scripts.run_stable_atlas_fit import Pool, task, replay_tasks, analyze_results, REPO
from scripts.run_panel_atlas_fit import expensive_first
from scripts.stable_atlas_policy import Coordinates, prior
from scripts.stable_atlas_worker import atomic_json as write
from scripts.selected_atlas_audit_checks import saved_arrays
from scripts.experiment_records import file_sha256, capture_run_record

SOURCE='/home/carsonc/r-elegans-results/extended-panel-atlas-fit-20261123T040511Z-890436'
CASES={'zero':0.,'minus-small':-.01,'plus-small':.01,'minus-large':-.1,'plus-large':.1}
FIELDS=('voltage_mV','gates','calcium_uM','indicator')


def prepare(source,root):
    source=Path(source).resolve();evidence={}
    def pin(path,expected=None):
        path=Path(path);sha=file_sha256(path)
        if expected is not None and sha!=expected:raise ValueError('Source evidence changed: '+str(path))
        evidence[str(path)]=sha;return path
    def doc(name):return json.loads(pin(source/(name+'.json')).read_text())
    m,a,b,s=doc('manifest'),doc('audit'),doc('budget_receipt'),doc('summary')
    if m['experiment']!='E118' or not all(a.get(k) for k in ('audit_complete','execution_complete','integrity_passed')) or a.get('problems') or a.get('unavailable_evidence'):
        raise ValueError('Complete clean E118 audit required')
    if not b.get('all_steps_finished') or b.get('timed_out') or not b.get('validation_complete'):raise ValueError('E118 execution incomplete')
    for d in (m,a):
        for path,sha in d['evidence_sha256'].items():pin(path,sha)
    inputs=root/'inputs';inputs.mkdir()
    def copy(path,name,expected=None):
        src=pin(path,expected);dest=inputs/name;shutil.copy2(src,dest);pin(dest,evidence[str(src)]);return str(dest)
    parent=copy(m['parent'],'parent.npz',m['parent_sha256'])
    optimizer=copy(source/'optimizer.npz','source-optimizer.npz')
    panel=doc('panel')['targets'];proposal=saved_arrays(optimizer)['proposal']
    coordinates=saved_arrays(pin(source/'trials/proposal/coordinates.npz'))['offsets']
    if not np.array_equal(proposal,coordinates):raise ValueError('Source proposal identity differs')
    codec=Coordinates(saved_arrays(parent));candidate=codec.apply(proposal);original=saved_arrays(pin(source/'trials/proposal/network.npz'))
    if set(candidate)!=set(original) or any(not np.array_equal(candidate[k],original[k]) for k in candidate):raise ValueError('Source candidate reconstruction differs')
    result=dict(m,experiment='E119',source=str(source),parent=parent,parent_sha256=file_sha256(parent),
        statistics=copy(m['statistics'],'statistics.json'),
        parent_absolute_offsets=copy(m['parent_absolute_offsets'],'absolute.npz',m['parent_absolute_offsets_sha256']),
        optimizer=optimizer,panel=panel,cases=CASES,evidence_sha256=evidence,
        cached_endpoints=dict(parent=s['panel_metrics']['current']['data_loss'],proposal=s['proposal_panel']['data_loss']),
        limits=['Panel derivative diagnostic only; no fitting/promotion/full-atlas or long-autonomy qualification.'])
    result['parent_absolute_offsets_sha256']=file_sha256(result['parent_absolute_offsets'])
    write(root/'manifest.json',result);return result


def scaled_offsets(direction,alpha,absolute):
    x=float(alpha)*np.asarray(direction)
    if not np.isfinite(x).all() or np.max(abs(x))>.1+1e-12 or np.max(abs(absolute+x))>.1+1e-12:
        raise ValueError('Direction violates bounds; refusing clipping')
    return x


def curve(rows,levels):
    out={}
    for level in levels:
        key=str(level);zero=rows['zero']['losses'][key];radii=[]
        for label,radius in [('small',.01),('large',.1)]:
            plus=rows['plus-'+label]['losses'][key];minus=rows['minus-'+label]['losses'][key]
            radii.append(dict(radius=radius,central_slope=(plus-minus)/(2*radius),
                forward_slope=(plus-zero)/radius,backward_slope=(zero-minus)/radius,
                plus_change=plus-zero,minus_change=minus-zero))
        out[key]=radii
    return out


def analyze(root,m,results,write_outputs=False):
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    stats=json.loads(Path(m['statistics']).read_text());ids=saved_arrays(m['parent'])['neuron_ids'].tolist()
    opt=saved_arrays(m['optimizer']);direction=opt['proposal'];absolute=saved_arrays(m['parent_absolute_offsets'])['offsets']
    metrics={};states={};matrices={};parameters={};models={}
    codec=Coordinates(saved_arrays(m['parent']))
    for name,alpha in CASES.items():
        model=root/'trials'/name/'network.npz';x=saved_arrays(model.parent/'coordinates.npz')['offsets']
        if not np.array_equal(x,scaled_offsets(direction,alpha,absolute)):raise ValueError('Trial direction altered')
        parameters[name]=codec.audit(saved_arrays(model),x)
        if not parameters[name]['passed']:raise ValueError('Frozen/parameter projection failed')
        models[name]=file_sha256(model)
        anchor=results['anchors-'+name]
        if anchor['task']['model_sha256']!=models[name]:raise ValueError('State generation model differs')
        a=saved_arrays(anchor['anchors'])
        if str(a['model_sha256'])!=models[name]:raise ValueError('State provenance differs')
        states[name]=anchor['anchors']
        rows=[r for label,r in results.items() if label.startswith('panel-'+name+'-')]
        check,mat=analyze_results(rows,m,stats,ids,m['panel'],[1,2],'train')
        # Retain raw data loss alongside regularized objective. No validation/test scores used.
        check['data_losses']=dict(check['losses']);check['losses']={k:v+prior(x) for k,v in check['losses'].items()}
        metrics[name]=check;matrices[name]=mat
    fresh=results['anchors-zero-repeat']
    if fresh['task']['model_sha256']!=models['zero']:raise ValueError('Zero repeat used different model')
    a=saved_arrays(states['zero']);b=saved_arrays(fresh['anchors'])
    state_errors={k:float(abs(a['start_'+k]-b['start_'+k]).max()) for k in FIELDS}
    repeated=[r for label,r in results.items() if label.startswith('zero-repeat-')]
    repeat_check,repeat=analyze_results(repeated,m,stats,ids,m['panel'],[2],'train')
    replay_error=float(abs(repeat[2]-matrices['zero'][2]).max())
    endpoint_error=abs(metrics['zero']['data_losses']['2']-m['cached_endpoints']['parent'])
    result=dict(experiment='E119',execution_complete=True,validation_complete=True,
        curve=curve(metrics,[1,2]),case_metrics=metrics,parameter_checks=parameters,
        zero_state_errors=state_errors,zero_panel_replay_error=replay_error,parent_cached_loss_difference=endpoint_error,
        parent_reproduced=bool(endpoint_error<=1e-10),repeat_passed=bool(max(state_errors.values())<=1e-12 and replay_error<=1e-10 and repeat_check['passed']),
        all_panel_screens_passed=all(v['passed'] for v in metrics.values()),
        source_raw_linear_slope=float(opt['averaged_gradient']@direction),cached_endpoints=m['cached_endpoints'],
        model_promoted=False,fit_qualified=False,
        scope='Read-only local panel diagnostic; slopes descriptive, not certified derivative/error bounds. No full atlas or long-autonomy requalification.')
    if write_outputs:write(root/'summary.json',result)
    return result


def coordinator(root):
    budget=json.loads((root/'deadline.json').read_text());deadline=budget['coordinator_deadline']
    m=prepare(Path(budget['source']),root);opt=saved_arrays(m['optimizer'])
    absolute=saved_arrays(m['parent_absolute_offsets'])['offsets'];codec=Coordinates(saved_arrays(m['parent']))
    models={}
    for name,alpha in CASES.items():
        folder=root/'trials'/name;folder.mkdir(parents=True);x=scaled_offsets(opt['proposal'],alpha,absolute)
        model=folder/'network.npz';np.savez_compressed(model,**codec.apply(x));np.savez_compressed(folder/'coordinates.npz',offsets=x)
        write(folder/'identity.json',dict(sha256=file_sha256(model),coordinates_sha256=file_sha256(folder/'coordinates.npz')));models[name]=model
    pool=Pool(root)
    try:
        specs=[]
        for name,model in models.items():
            spec=task(root,'anchors-'+name,model);spec['kind']='anchors';specs.append(spec)
        rows=pool.tasks(specs,deadline);anchors={r['task']['name'].removeprefix('anchors-'):r['anchors'] for r in rows}
        specs=[]
        for name,model in models.items():
            trial=replay_tasks(root,m,'panel-'+name,model,anchors[name],m['panel'],[1,2])
            for spec in trial:spec['require_own_state']=True
            specs.extend(trial)
        pool.tasks(expensive_first(specs),deadline)
        spec=task(root,'anchors-zero-repeat',models['zero']);spec['kind']='anchors'
        fresh=pool.tasks([spec],deadline)[0]
        specs=replay_tasks(root,m,'zero-repeat',models['zero'],fresh['anchors'],m['panel'],[2])
        for spec in specs:spec['require_own_state']=True
        pool.tasks(expensive_first(specs),deadline)
        results={}
        for path in (root/'tasks').glob('*/result.json'):
            r=json.loads(path.read_text());results[r['task']['name']]=r
        analyze(root,m,results,write_outputs=True)
    finally:pool.close()


def audit(root):
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    evidence={};problems=[];missing=[];computed=None
    def pin(path,expected=None):
        path=Path(path);sha=file_sha256(path);evidence[str(path)]=sha
        if expected is not None and sha!=expected:raise ValueError('Hash mismatch: '+str(path))
        return path
    try:
        m=json.loads(pin(root/'manifest.json').read_text())
        for path,sha in m['evidence_sha256'].items():pin(path,sha)
        results={}
        for folder in (root/'trials').iterdir():
            identity=json.loads(pin(folder/'identity.json').read_text());pin(folder/'network.npz',identity['sha256']);pin(folder/'coordinates.npz',identity['coordinates_sha256'])
        for path in (root/'tasks').glob('*/task.json'):
            spec=json.loads(pin(path).read_text());pin(spec['model'],spec['model_sha256'])
            if not Path(spec['result']).exists():missing.append(spec['name']);continue
            r=json.loads(pin(spec['result']).read_text());results[spec['name']]=r
            if r['task']!=spec or r['status']!='completed':raise ValueError('Task failed/identity changed')
            if spec['kind']=='anchors':
                a=saved_arrays(pin(r['anchors'],r['anchors_sha256']))
                if str(a['model_sha256'])!=spec['model_sha256']:raise ValueError('Anchor provenance differs')
                for key in FIELDS:
                    if not np.isfinite(a['start_'+key]).all():raise ValueError('Nonfinite initial state')
                continue
            pin(spec['anchors'],spec['anchors_sha256']);a=saved_arrays(pin(r['artifact'],r['sha256']))
            if str(saved_arrays(spec['anchors'])['model_sha256'])!=spec['model_sha256']:raise ValueError('Replay own-state mismatch')
            n=len(spec['targets']);ids=saved_arrays(m['parent'])['neuron_ids'].tolist()
            controls=2*((n+7)//8)+(-n)%8
            if a['response'].shape!=(4,n,len(ids)) or a['control_means'].shape!=(controls,4,len(ids)) or a['neuron_ids'].tolist()!=ids:
                raise ValueError('Incomplete response/control/neuron coverage')
            labels=[f'batch-{begin//8}-condition-{k}' for begin in range(0,n,8) for k in range(min(8,n-begin),10)]
            if a['targets'].tolist()!=spec['targets'] or a['control_labels'].tolist()!=labels or float(a['neural_step_ms'])!=spec['step_ms'] or str(a['anchors_sha256'])!=spec['anchors_sha256']:
                raise ValueError('Replay grid/target/control identity differs')
            for key in ('response','control_means','reference_sham_rmd_mV'):
                if a[key].dtype!=np.float64 or not np.isfinite(a[key]).all():raise ValueError('Nonfinite/nonfloat64 output')
            mean=float(abs(a['control_means']-a['control_means'][0]).max());voltage=float(a['maximum_control_voltage_delta_mV'])
            rhythms=[[rhythm_metrics(a['reference_sham_rmd_mV'][p,:,j],.001) for j in range(2)] for p in range(4)]
            if mean!=r['duplicate_mean_error'] or voltage!=r['duplicate_trace_error'] or rhythms!=r['rhythms'] or r['controls_passed']!=(mean<=1e-12 and voltage<=1e-12):raise ValueError('Signal/control score differs')
        if missing:raise ValueError('Incomplete tasks')
        computed=analyze(root,m,results)
        if computed!=json.loads(pin(root/'summary.json').read_text()):raise ValueError('Loss/slope/replay arithmetic differs')
    except Exception as exc:traceback.print_exc();problems.append(repr(exc))
    for required in ('manifest.json','summary.json'):
        if not (root/required).exists():missing.append(required)
    for path in root.rglob('*'):
        if path.is_file() and path.name not in ('audit.json','budget_receipt.json') and path.suffix in ('.json','.npz','.py'):
            try:pin(path)
            except Exception as exc:problems.append(repr(exc))
    result=dict(experiment='E119',audit_complete=True,execution_complete=bool(computed and not missing),integrity_passed=not problems,
        problems=problems,unavailable_evidence=missing,evidence_sha256=evidence,model_promoted=False,
        scope='Saved-data panel-direction audit; no independent forward rerun, full-atlas or behavioral qualification.',
        limits=['Duplicate voltage errors retained as reductions; full duplicate traces unavailable.','Grid screens do not certify derivative accuracy.'])
    write(root/'audit.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--source',type=Path,default=Path(SOURCE))
    p.add_argument('--coordinator',action='store_true');p.add_argument('--audit',action='store_true');p.add_argument('--report',action='store_true')
    a=p.parse_args();root=a.output_dir.resolve()
    if a.report:
        print('Run:',root)
        for name in ('summary','audit','budget_receipt'):
            path=root/(name+'.json');s=json.loads(path.read_text()) if path.exists() else {}
            keys={'summary':('curve','source_raw_linear_slope','cached_endpoints','all_panel_screens_passed','parent_reproduced','repeat_passed'),
                  'audit':('audit_complete','execution_complete','integrity_passed','problems','unavailable_evidence'),
                  'budget_receipt':('total_minutes','elapsed_seconds','all_steps_finished','timed_out','diagnostic_checks_passed')}[name]
            print(name+':',json.dumps({k:s.get(k) for k in keys}))
        return
    if a.audit:audit(root);return
    if a.coordinator:
        try:coordinator(root)
        except Exception as exc:traceback.print_exc();write(root/'failure.json',dict(error=repr(exc)));raise
        return
    if root==REPO or REPO in root.parents or root in REPO.parents:p.error('External output required')
    start=time.monotonic();end=start+120*60;root.mkdir(parents=True,exist_ok=False)
    write(root/'deadline.json',dict(source=str(a.source.resolve()),coordinator_deadline=end-600,total_minutes=120,deadline_monotonic=end))
    capture_run_record(root,REPO,vars(a),phase='E119-supervisor')
    from scripts.total_job_budget import run_group
    code,timeout=run_group([sys.executable,'-u','-m','scripts.run_atlas_direction_check','--output-dir',str(root),'--coordinator'],max(.01,end-time.monotonic()-600))
    ac,at=run_group([sys.executable,'-u','-m','scripts.run_atlas_direction_check','--output-dir',str(root),'--audit'],max(.01,end-time.monotonic()-10))
    def read(name):
        path=root/(name+'.json');return json.loads(path.read_text()) if path.exists() else {}
    au,s=read('audit'),read('summary');elapsed=time.monotonic()-start
    complete=bool(code==0 and not timeout and ac==0 and not at and au.get('execution_complete') and au.get('audit_complete') and elapsed<=120*60)
    receipt=dict(total_minutes=120,elapsed_seconds=elapsed,all_steps_finished=complete,timed_out=timeout,audit_timed_out=at,
        exit_code=code,audit_exit_code=ac,diagnostic_checks_passed=bool(complete and au.get('integrity_passed') and s.get('all_panel_screens_passed') and s.get('parent_reproduced') and s.get('repeat_passed')),
        fit_qualified=False,model_promoted=False)
    write(root/'budget_receipt.json',receipt);print(json.dumps(receipt),flush=True);raise SystemExit(0 if complete else 2)


if __name__=='__main__':main()
