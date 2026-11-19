"""E095 saved-data audit: coordinates, widened-prior use, hashes, retained-trace scores and finalist selection."""
import argparse,json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from r_elegans.cma_curriculum import decode_groups
from r_elegans.edge_scan import scan_groups,extend_upper,planned_cases,finalists
from r_elegans.relay_cap_experiments import upper_for
from r_elegans.sensory_motion import features,score
from scripts.audit_rial_sensory import frozen_network_fields
from scripts.audit_relay_experiments import write,close,coordinate_check
from scripts.experiment_records import capture_run_record,file_sha256


def audit(run,out):
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,Path(__file__).resolve().parents[1],dict(run=str(run)))
    problems=[];evidence=[];counts=Counter();extended=Counter()
    def check(ok,label):
        if not ok:problems.append(label)
    def verify(path,sha):
        actual=file_sha256(path);check(actual==sha,'Hash mismatch: '+str(path));evidence.append(dict(path=str(path),sha256=actual,expected=sha))
    s=json.loads((run/'summary.json').read_text());config=json.loads((run/'configuration.json').read_text())
    check(s.get('experiment')=='E095','Not an E095 run')
    manifest=json.loads((run/'inputs/manifest.json').read_text())
    for name,sha in manifest['sha256'].items():verify(run/'inputs'/name,sha)
    ref=json.loads((run/'inputs/sensory_reference.json').read_text());verify(Path(ref['path']),ref['sha256'])
    parent=run/'inputs/motion_parent.npz';verify(parent,config['parent_sha256'])
    with np.load(run/'inputs/cap_prior.npz',allow_pickle=False) as z:origin,lower,upper,edge=z['origin'],z['lower'],z['upper'],int(z['edge'])
    with np.load(run/'inputs/source_bounds.npz',allow_pickle=False) as z:
        check(np.array_equal(lower,z['lower']) and np.array_equal(upper,z['upper']),'Original bound policy changed')
    with np.load(run/'inputs/motion_parent_coordinates.npz',allow_pickle=False) as z:check(np.array_equal(origin,z['offsets']),'Parent coordinates changed')
    with np.load(parent,allow_pickle=False) as z:
        ids=list(z['neuron_ids']);mask=z['fixed_chemical_mask'].astype(bool);channels=z['cells_conductances_nS']>0;channels[:,16]=False
        threshold=z['fixed_release_threshold_mV'];slope=z['fixed_release_slope_mV']
    spec=SimpleNamespace(channels=channels,chemical=mask)
    groups=json.loads((run/'inputs/cap_groups.json').read_text())['scan']
    check(groups==scan_groups(spec,ids,origin,lower,upper),'Unexpected scan group policy')
    cases=planned_cases(groups);check(close(config['planned'],cases),'Planned case table changed')
    allowed=extend_upper(upper_for('aiyr_2p5',upper,edge),origin,groups)
    parent_features=json.loads((run/'parent_features.json').read_text());saved={};requests={}
    for request in sorted((run/'evaluations').glob('*/request.json')):
        folder=request.parent;j=json.loads(request.read_text());requests[str(folder)]=j
        try:
            r=json.loads((folder/'summary.json').read_text());saved[str(folder)]=r;counts[r['status']]+=1
            check(Path(r['output']).resolve()==folder and Path(j['output']).resolve()==folder,'Evaluation path mismatch: '+folder.name)
            x=decode_groups(j['z'],groups,origin,lower,allowed)
            scanned={groups[i]['indices'][0] for i,v in enumerate(j['z']) if v!=0}
            if (folder/'network.npz').exists():
                verify(folder/'network.npz',r['checkpoint_sha256']);verify(folder/'coordinates.npz',r['coordinates_sha256'])
                with np.load(folder/'coordinates.npz',allow_pickle=False) as z:check(np.array_equal(x,z['offsets']) and np.array_equal(z['z'],j['z']),'Coordinate decode mismatch: '+folder.name)
                used=x!=origin
                check(set(np.flatnonzero(used).tolist())==scanned,'Changed coordinates differ from the single scanned edge: '+folder.name)
                check(frozen_network_fields(parent,folder/'network.npz',used),'Frozen model changed: '+folder.name)
                check(coordinate_check(parent,folder/'network.npz',origin,x),'Physical coordinate mismatch: '+folder.name)
                outside=set(np.flatnonzero(x>upper+1e-10).tolist())
                check(outside<=scanned|{edge},'Undeclared original-bound violation: '+folder.name)
                extended[folder.name.split('-')[0]]+=int(bool(outside&scanned))
            elif r['status']=='completed':check(False,'Completed evaluation missing model: '+folder.name)
            hashes=r.get('trajectory_sha256',{});paired={};traces={}
            for condition,sha in hashes.items():
                path=folder/(condition+'.npz');verify(path,sha)
                with np.load(path,allow_pickle=False) as z:trace={k:z[k] for k in z.files}
                check(all(np.isfinite(v).all() for v in trace.values() if np.issubdtype(v.dtype,np.number)),'Nonfinite raw trace: '+str(path))
                paired[condition]=features(trace,ids,j['variant'],float(threshold[ids.index('AIYL')]),float(slope[ids.index('AIYL')]));traces[condition]=trace
            if r['status']=='completed':
                check(set(hashes)==({'no_cue','cue'} if j['retain_traces'] else set()),'Raw retention policy mismatch: '+folder.name)
                for key,value in paired.items():check(close(value,r['paired_features'][key]),'Raw features mismatch: '+folder.name+'/'+key)
                metrics=score(r['paired_features'],j.get('parent_features') or r['paired_features']['no_cue'],j['target'],j['margin'])
                check(close(metrics,r['metrics']),'Recorded score mismatch: '+folder.name)
                if traces:
                    from r_elegans.rial_sensory import protocol
                    base=traces['no_cue'];before=base['dense_time_s']<protocol(j['variant'])['events'][0][0]
                    for trace in traces.values():
                        check(np.array_equal(trace['dense_time_s'],base['dense_time_s']),'Clock mismatch: '+folder.name)
                        check(all(np.max(abs(trace[k][before]-base[k][before]))<=1e-8 for k in ('dense_voltage_mV','dense_joints')),'Pre-cue mismatch: '+folder.name)
        except Exception as e:problems.append(folder.name+': '+repr(e))
    check(sum(counts.values())==s['evaluation_count'],'Evaluation count mismatch')
    for r in s['scan']+s['controls']+([s['baseline']] if s.get('baseline') else []):check(saved.get(r['output'])==r,'Summary/saved record mismatch: '+r['output'])
    runnable=[c for c in cases if c['runnable']]
    done={tuple(requests[r['output']]['z']) for r in s['scan'] if r['status']=='completed' and r['output'] in requests}
    check(all(tuple(c['z']) in done for c in runnable) or not s.get('validation_complete'),'Planned runnable case missing from completed scan')
    check(s['finalists']==[r['output'] for r in finalists(s['scan'])],'Finalist selection differs from recomputed ranking')
    repeat={}
    for row in s['controls']:
        if row.get('status')=='completed' and row['grid']=='ordinary' and row['mode']=='normal':
            src=requests[row['output']]['source_scan'];base=saved.get(src)
            repeat[src]=dict(same_checkpoint=bool(base and base['checkpoint_sha256']==row['checkpoint_sha256']),metrics_match=bool(base and close(base['metrics'],row['metrics'])))
            check(repeat[src]['same_checkpoint'],'Finalist rerun checkpoint differs from scan: '+src)
    write(out/'evidence.json',evidence);write(out/'repeat_consistency.json',repeat)
    result=dict(audit_complete=True,integrity_passed=not problems,problems=problems,status_counts=dict(counts),
        widened_prior_evaluations_by_phase=dict(extended),finalist_rerun_consistency=repeat,
        scope='Saved-data integrity, independent coordinate/bound and score checks; scan cases are metrics-only (no raw replay); no forward simulation, causal proof or qualification.')
    write(out/'summary.json',result);print(json.dumps(result,indent=2),flush=True)
    return 0 if not problems else 2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    run,out=a.run.resolve(),a.output_dir.resolve()
    if run==out or run in out.parents or out in run.parents:raise ValueError('Separate audit output required')
    return audit(run,out)


if __name__=='__main__':raise SystemExit(main())
