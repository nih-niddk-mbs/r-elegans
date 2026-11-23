"""E120 evaluate exact saved E119 minus-large weights; no optimization."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from scripts.run_stable_atlas_fit import Pool,task,replay_tasks,analyze_results,REPO
from scripts.run_panel_atlas_fit import expensive_first
from scripts.atlas_fit_continuation import prepare_continuation
from scripts.stable_atlas_policy import Coordinates,loss,prior,preserved
from scripts.stable_atlas_worker import atomic_json as write
from scripts.selected_atlas_audit_checks import saved_arrays
from scripts.experiment_records import capture_run_record,file_sha256
from scripts.atlas_task_audit import audit_tasks

SOURCE='/home/carsonc/r-elegans-results/atlas-direction-check-20261123T151107Z-958304'


def prepare(source,root):
    source=Path(source).resolve();docs={name:json.loads((source/(name+'.json')).read_text()) for name in ('manifest','summary','audit','budget_receipt')}
    old,s,a,b=[docs[k] for k in ('manifest','summary','audit','budget_receipt')]
    if old['experiment']!='E119' or not all(a.get(k) for k in ('audit_complete','execution_complete','integrity_passed')) or a.get('problems') or a.get('unavailable_evidence') or not b.get('diagnostic_checks_passed') or not b.get('all_steps_finished') or b.get('timed_out'):
        raise ValueError('Complete clean E119 diagnostic required')
    m=prepare_continuation(Path(old['continuation']['parent_run']),root)
    evidence=m['evidence_sha256']
    def pin(path,expected=None):
        path=Path(path);sha=file_sha256(path)
        if expected is not None and sha!=expected:raise ValueError('Source hash mismatch: '+str(path))
        evidence[str(path)]=sha;return path
    for name in docs:pin(source/(name+'.json'))
    for d in (old,a):
        for path,sha in d['evidence_sha256'].items():pin(path,sha)
    if old['parent_sha256']!=m['parent_sha256']:raise ValueError('E119 parent differs from qualified E115')
    folder=source/'trials/minus-large';identity=json.loads(pin(folder/'identity.json').read_text())
    model=pin(folder/'network.npz',identity['sha256']);coordinates=pin(folder/'coordinates.npz',identity['coordinates_sha256'])
    x=saved_arrays(coordinates)['offsets'];direction=saved_arrays(pin(old['optimizer']))['proposal']
    if not np.array_equal(x,-.1*direction):raise ValueError('Not exact minus10% trial')
    codec=Coordinates(saved_arrays(m['parent']))
    if not codec.audit(saved_arrays(model),x)['passed']:raise ValueError('Candidate reconstruction/frozen fields differ')
    absolute=saved_arrays(m['parent_absolute_offsets'])['offsets']
    if not np.isfinite(absolute+x).all() or np.max(abs(absolute+x))>.1+1e-12:raise ValueError('Cumulative bounds exceeded')
    dest=root/'inputs/candidate.npz';shutil.copy2(model,dest);pin(dest,identity['sha256'])
    coord=root/'inputs/candidate-coordinates.npz';shutil.copy2(coordinates,coord);pin(coord,identity['coordinates_sha256'])
    result=dict(m,experiment='E120',candidate=str(dest),candidate_sha256=identity['sha256'],coordinates=str(coord),
        panel=old['panel'],direction_source=str(source),panel_cached_loss=s['case_metrics']['minus-large']['data_losses']['2'],
        source_parent_run=m['continuation']['parent_run'],evidence_sha256=evidence,method='Fixed saved candidate evaluation only; no optimizer')
    write(root/'manifest.json',result);return result


def assess(root,m,results):
    stats=json.loads(Path(m['statistics']).read_text());ids=saved_arrays(m['parent'])['neuron_ids'].tolist()
    x=saved_arrays(m['coordinates'])['offsets'];codec=Coordinates(saved_arrays(m['parent']));absolute=saved_arrays(m['parent_absolute_offsets'])['offsets']
    params=codec.audit(saved_arrays(m['candidate']),x)
    params['cumulative_bound_passed']=bool(np.max(abs(absolute+x))<=.1+1e-12)
    rows=[r for name,r in results.items() if name.startswith('candidate-final-')]
    final,full=analyze_results(rows,m,stats,ids,m['targets'],[0,1,2],'val',reference=False,final=True)
    auto=results['candidate-autonomy'];initial=results['candidate-anchors']
    a=saved_arrays(initial['anchors']);b=saved_arrays(auto['anchors'])
    for r in (initial,auto):
        if r['task']['model_sha256']!=m['candidate_sha256']:raise ValueError('Own-state candidate identity differs')
    state_errors={k:float(abs(a['start_'+k]-b['start_'+k]).max()) for k in ('voltage_mV','gates','calcium_uM','indicator')}
    repeated=[r for name,r in results.items() if name.startswith('repeat-panel-')]
    rep,mat=analyze_results(repeated,m,stats,ids,m['panel'],[2],'train',reference=False)
    error=float(abs(mat[2]-full[2][:,[m['targets'].index(t) for t in m['panel']]]).max())
    parent_auto=json.loads((Path(m['source_parent_run'])/'autonomy.json').read_text())['selected']
    parent=saved_arrays(m['baselines'][2])['response']
    # Test scores are calculated only after the fixed candidate decision.
    parent_losses={split:loss(parent,m['targets'],ids,stats,split) for split in ('train','val')}
    losses={split:loss(full[2],m['targets'],ids,stats,split) for split in ('train','val')}
    panel_error=abs(rep['data_loss']-m['panel_cached_loss'])
    gates=dict(full_numerical_passed=final['passed'],parameters_passed=params['passed'] and params['cumulative_bound_passed'],
        parameters_changed=params['changed_coordinates']>0,
        train_improved=losses['train']+prior(x)<parent_losses['train'],
        validation_improved=losses['val']+prior(x)<parent_losses['val'],
        autonomy_passed=all(preserved(parent_auto['rhythms'][i][j],auto['rhythms'][i][j]) for i in range(2) for j in range(2)),
        state_reproduced=max(state_errors.values())<=1e-12,panel_reproduced=bool(error<=1e-10 and panel_error<=1e-10 and rep['passed']))
    qualified=bool(all(gates.values()))
    selected=m['candidate'] if qualified else m['parent'];selected_sha=file_sha256(selected)
    scores=dict(losses,test=loss(full[2],m['targets'],ids,stats,'test'))
    return dict(experiment='E120',execution_complete=True,validation_complete=True,evaluation_only=True,
        candidate_qualified_before_audit=qualified,selected_sha256=selected_sha,selected=str(selected),candidate_sha256=m['candidate_sha256'],
        candidate_scores=scores,parent_losses=parent_losses,parameters=params,gates=gates,final_metrics=final,
        strict_cycle_diagnostics=auto['strict_cycle_checks'],state_errors=state_errors,panel_response_error=error,cached_panel_loss_error=panel_error,
        independent_full_loss_reproduction=False,model_promoted=False,sensory_start_stop_qualified=False,swimming_qualified=False,
        scope='One fixed candidate; full atlas once, own-state/finest-panel repeat, long RMD preservation. No new fitting or biological validation.')


def coordinator(root):
    budget=json.loads((root/'deadline.json').read_text());deadline=budget['coordinator_deadline']
    m=prepare(Path(budget['source']),root);pool=Pool(root)
    try:
        spec=task(root,'candidate-anchors',m['candidate']);spec['kind']='anchors'
        initial=pool.tasks([spec],deadline)[0]
        specs=replay_tasks(root,m,'candidate-final',m['candidate'],initial['anchors'],m['targets'],[0,1,2])
        for spec in specs:spec['require_own_state']=True
        # Autonomy uses the second GPU when replay dispatch allows it; no shared state mutation.
        specs=expensive_first(specs)+[task(root,'candidate-autonomy',m['candidate'])]
        rows=pool.tasks(specs,deadline);auto=next(r for r in rows if r['task']['name']=='candidate-autonomy')
        specs=replay_tasks(root,m,'repeat-panel',m['candidate'],auto['anchors'],m['panel'],[2])
        for spec in specs:spec['require_own_state']=True
        pool.tasks(expensive_first(specs),deadline)
        results={}
        for path in (root/'tasks').glob('*/result.json'):
            r=json.loads(path.read_text());results[r['task']['name']]=r
        result=assess(root,m,results);write(root/'summary.json',result)
        write(root/'selection.json',{k:result[k] for k in ('selected','selected_sha256','candidate_qualified_before_audit','gates')})
    finally:pool.close()


def audit(root, *, output_dir=None):
    root=Path(root).resolve()
    destination=root
    if output_dir is not None:
        destination=Path(output_dir).resolve()
        if destination==root or root in destination.parents:raise ValueError('Independent audit output must be outside source run')
        destination.mkdir(parents=True,exist_ok=False)
    hashes={};problems=[];missing=[];computed=None
    def pin(path,expected=None):
        path=Path(path);sha=file_sha256(path);hashes[str(path)]=sha
        if expected is not None and sha!=expected:raise ValueError('Changed artifact: '+str(path))
        return path
    try:
        m=json.loads(pin(root/'manifest.json').read_text())
        for path,sha in m['evidence_sha256'].items():pin(path,sha)
        pin(m['candidate'],m['candidate_sha256']);pin(m['parent'],m['parent_sha256'])
        ids=saved_arrays(m['parent'])['neuron_ids'].tolist();results=audit_tasks(root,ids,pin)
        computed=assess(root,m,results)
        if computed!=json.loads(pin(root/'summary.json').read_text()):raise ValueError('Candidate metrics/decision differs')
        selected=json.loads(pin(root/'selection.json').read_text())
        if selected!={k:computed[k] for k in selected} or file_sha256(selected['selected'])!=selected['selected_sha256']:raise ValueError('Selection identity differs')
    except Exception as exc:traceback.print_exc();problems.append(repr(exc))
    for name in ('manifest.json','summary.json','selection.json'):
        if not (root/name).exists():missing.append(name)
    for path in root.rglob('*'):
        if path.is_file() and path.name not in ('audit.json','budget_receipt.json') and path.suffix in ('.json','.npz','.py'):
            try:pin(path)
            except Exception as exc:problems.append(repr(exc))
    result=dict(experiment='E120',audit_complete=True,execution_complete=bool(computed and not missing),integrity_passed=not problems,
        problems=problems,unavailable_evidence=missing,evidence_sha256=hashes,model_promoted=False,
        limits=['Saved-data audit; no independent full-atlas loss rerun.','Duplicate voltage errors saved as reductions, not full duplicate traces.','Long RMD preservation is not body movement or behavior qualification.'])
    if output_dir is not None:result.update(source_run=str(root),independent_saved_data_recheck=True)
    write(destination/'audit.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--source',type=Path,default=Path(SOURCE))
    p.add_argument('--coordinator',action='store_true');p.add_argument('--audit',action='store_true');p.add_argument('--report',action='store_true')
    a=p.parse_args();root=a.output_dir.resolve()
    if a.report:
        print('Run:',root)
        for name,keys in [('summary',('candidate_qualified_before_audit','candidate_sha256','selected_sha256','candidate_scores','parent_losses','parameters','gates','independent_full_loss_reproduction')),('audit',('audit_complete','execution_complete','integrity_passed','problems','unavailable_evidence')),('budget_receipt',('total_minutes','elapsed_seconds','all_steps_finished','timed_out','candidate_qualified'))]:
            path=root/(name+'.json');d=json.loads(path.read_text()) if path.exists() else {};print(name+':',json.dumps({k:d.get(k) for k in keys}))
        return
    if a.audit:audit(root);return
    if a.coordinator:
        try:coordinator(root)
        except Exception as exc:traceback.print_exc();write(root/'failure.json',dict(error=repr(exc)));raise
        return
    if root==REPO or REPO in root.parents or root in REPO.parents:p.error('External output required')
    start=time.monotonic();end=start+120*60;root.mkdir(parents=True,exist_ok=False)
    write(root/'deadline.json',dict(source=str(a.source.resolve()),coordinator_deadline=end-600,total_minutes=120))
    capture_run_record(root,REPO,vars(a),phase='E120-supervisor')
    from scripts.total_job_budget import run_group
    code,timeout=run_group([sys.executable,'-u','-m','scripts.run_saved_atlas_candidate','--output-dir',str(root),'--coordinator'],max(.01,end-time.monotonic()-600))
    ac,at=run_group([sys.executable,'-u','-m','scripts.run_saved_atlas_candidate','--output-dir',str(root),'--audit'],max(.01,end-time.monotonic()-10))
    def read(name):
        path=root/(name+'.json');return json.loads(path.read_text()) if path.exists() else {}
    s,au=read('summary'),read('audit');elapsed=time.monotonic()-start
    complete=bool(code==0 and not timeout and ac==0 and not at and au.get('audit_complete') and au.get('execution_complete') and elapsed<=120*60)
    receipt=dict(total_minutes=120,elapsed_seconds=elapsed,all_steps_finished=complete,timed_out=timeout,
        exit_code=code,audit_exit_code=ac,audit_timed_out=at,candidate_qualified=bool(complete and au.get('integrity_passed') and s.get('candidate_qualified_before_audit')),
        model_promoted=False)
    write(root/'budget_receipt.json',receipt);print(json.dumps(receipt),flush=True);raise SystemExit(0 if complete else 2)


if __name__=='__main__':main()
