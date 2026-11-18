"""E092 saved-data audit, with one explicit experimental prior exception."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from r_elegans.cma_curriculum import decode_groups
from r_elegans.relay_cap_experiments import upper_for,panel_assessment
from scripts.audit_rial_sensory import frozen_network_fields,gait_failures,relay_failures,signal_windows
from scripts.experiment_records import capture_run_record,file_sha256


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def close(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(close(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and not isinstance(a,bool):return bool(np.isclose(a,b,rtol=1e-9,atol=1e-12))
    return a==b


def coordinate_check(parent,model,origin,x):
    """Check the physical parameters independently of the JAX decoder."""
    with np.load(parent,allow_pickle=False) as a,np.load(model,allow_pickle=False) as b:
        channels=a['cells_conductances_nS']>0;channels[:,16]=False
        chemical=a['fixed_chemical_mask'].astype(bool);nc=int(channels.sum());ne=int(chemical.sum())
        delta=x-origin
        return bool(np.allclose(b['params_log_channel_scale'][channels],a['params_log_channel_scale'][channels]+delta[:nc],rtol=0,atol=1e-12)
            and np.allclose(np.logaddexp(0,b['params_raw_chemical'][chemical]),
                np.logaddexp(0,a['params_raw_chemical'][chemical])*np.exp(delta[nc:nc+ne]),rtol=1e-10,atol=1e-12))


def audit(run,out):
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,Path(__file__).resolve().parents[1],dict(run=str(run)))
    problems=[];evidence=[];details=[];counts=Counter();violations=Counter()
    def check(ok,label):
        if not ok:problems.append(label)
    def verify(path,sha):
        actual=file_sha256(path);check(actual==sha,'Hash mismatch: '+str(path))
        evidence.append(dict(path=str(path),sha256=actual,expected=sha))
    s=json.loads((run/'summary.json').read_text());config=json.loads((run/'configuration.json').read_text())
    manifest=json.loads((run/'inputs/manifest.json').read_text())
    for name,sha in manifest['sha256'].items():verify(run/'inputs'/name,sha)
    ref=json.loads((run/'inputs/sensory_reference.json').read_text());verify(Path(ref['path']),ref['sha256'])
    parent=run/'inputs/sensory_parent.npz';verify(parent,config['parent_sha256'])
    with np.load(run/'inputs/cap_prior.npz',allow_pickle=False) as z:
        origin,lower,upper,edge=z['origin'],z['lower'],z['upper'],int(z['edge'])
    with np.load(run/'inputs/source_bounds.npz',allow_pickle=False) as z:
        check(np.array_equal(lower,z['lower']) and np.array_equal(upper,z['upper']),'Original bound policy changed')
    with np.load(run/'inputs/sensory_parent_coordinates.npz',allow_pickle=False) as z:check(np.array_equal(origin,z['offsets']),'Parent coordinates changed')
    with np.load(parent,allow_pickle=False) as z:
        ids=list(z['neuron_ids']);mask=z['fixed_chemical_mask'].astype(bool);post,pre=np.nonzero(mask)
        channels=z['cells_conductances_nS']>0;channels[:,16]=False
        expected=int(channels.sum())+int(np.flatnonzero((post==ids.index('AIYL'))&(pre==ids.index('AIYR')))[0])
        threshold=z['fixed_release_threshold_mV'];slope=z['fixed_release_slope_mV']
    check(edge==expected==config['only_extended_coordinate'],'Extended edge is not AIYR->AIYL')
    schemas=json.loads((run/'inputs/cap_groups.json').read_text())
    # Reconstruct the declared focused groups from unchanged anatomical masks.
    from types import SimpleNamespace
    from r_elegans.relay_range import parameter_groups,features,score
    spec=SimpleNamespace(channels=channels,chemical=mask)
    for cap in schemas:check(schemas[cap]==parameter_groups(spec,ids,origin,lower,upper_for(cap,upper,edge)),'Unexpected group policy: '+cap)
    parent_features=json.loads((run/'parent_features.json').read_text());saved={}
    for request in sorted((run/'evaluations').glob('*/request.json')):
        folder=request.parent;j=json.loads(request.read_text())
        if j.get('retain_traces'):print('Auditing retained signals:',folder.name,flush=True)
        try:
            r=json.loads((folder/'summary.json').read_text());saved[str(folder)]=r;counts[r['status']]+=1
            check(Path(r['output']).resolve()==folder and Path(j['output']).resolve()==folder,'Evaluation path mismatch: '+folder.name)
            x=decode_groups(j['z'],schemas[j['cap']],origin,lower,upper_for(j['cap'],upper,edge))
            if (folder/'network.npz').exists():
                verify(folder/'network.npz',r['checkpoint_sha256']);verify(folder/'coordinates.npz',r['coordinates_sha256'])
                with np.load(folder/'coordinates.npz',allow_pickle=False) as z:check(np.array_equal(x,z['offsets']) and np.array_equal(z['z'],j['z']),'Coordinate decode mismatch: '+folder.name)
                used=x!=origin
                check(frozen_network_fields(parent,folder/'network.npz',used),'Frozen model changed: '+folder.name)
                check(coordinate_check(parent,folder/'network.npz',origin,x),'Physical coordinate mismatch: '+folder.name)
                outside=np.flatnonzero(x>upper+1e-10)
                check(not len(outside) or (j['cap']=='aiyr_2p5' and np.array_equal(outside,[edge])),'Undeclared original-bound violation: '+folder.name)
                violations[j['cap']]+=int(bool(len(outside)))
                if j['kind']=='single_edge':check(not np.any(used[np.arange(len(x))!=edge]),'Single-edge test changed other coordinates: '+folder.name)
            elif r['status']=='completed':check(False,'Completed evaluation missing model: '+folder.name)
            hashes=r.get('trajectory_sha256',{});paired={};traces={}
            for condition,sha in hashes.items():
                path=folder/(condition+'.npz');verify(path,sha)
                with np.load(path,allow_pickle=False) as z:trace={k:z[k] for k in z.files}
                check(all(np.isfinite(v).all() for v in trace.values() if np.issubdtype(v.dtype,np.number)),'Nonfinite raw trace: '+str(path))
                paired[condition]=features(trace,ids,j['variant'],float(threshold[ids.index('AIYL')]),float(slope[ids.index('AIYL')]))
                traces[condition]=trace
            if r['status']=='completed':
                expected_conditions={'no_cue','cue'}|({'no_cue_replay','cue_replay'} if j['mode']=='relay-replay' else set())
                check(set(hashes)==(expected_conditions if j['retain_traces'] else set()),'Raw retention policy mismatch: '+folder.name)
                for key,value in paired.items():check(close(value,r['paired_features'][key]),'Raw features mismatch: '+folder.name+'/'+key)
                metrics=score(r['paired_features'],j.get('parent_features') or r['paired_features']['no_cue'],j['target'],j['margin'],response_aggregation=j['response_aggregation'])
                check(close(metrics,r['metrics']),'Recorded score mismatch: '+folder.name)
                row=dict(output=str(folder),cap=j['cap'],kind=j['kind'],raw_retained=bool(hashes),
                    gait_failures=gait_failures(r['paired_features'],j.get('parent_features') or r['paired_features']['no_cue']),
                    relay_failures=relay_failures(r['paired_features'],j['target']))
                if traces:
                    from r_elegans.rial_sensory import protocol
                    onset=protocol(j['variant'])['events'][0][0]
                    base=traces['no_cue'];before=base['dense_time_s']<onset
                    for trace in traces.values():
                        check(np.array_equal(trace['dense_time_s'],base['dense_time_s']),'Clock mismatch: '+folder.name)
                        check(all(np.max(abs(trace[k][before]-base[k][before]))<=1e-8 for k in ('dense_voltage_mV','dense_joints')),'Pre-cue mismatch: '+folder.name)
                    windows=[feature_row['window'] for category in ('active','response','late')
                             for feature_row in r['paired_features']['no_cue'][category]]
                    write(out/(folder.name+'-signals.json'),dict(windows=signal_windows(base,traces['cue'],ids,threshold,slope,windows)))
                details.append(row)
        except Exception as e:problems.append(folder.name+': '+repr(e))
    check(sum(counts.values())==s['evaluation_count'],'Evaluation count mismatch')
    for r in s['scan']+s.get('scan_controls',[])+[r for arm in s['arms'].values() for r in (arm.get('primary'),arm.get('best_all_completed')) if r]:
        check(saved.get(r['output'])==r,'Selected/scan record mismatch: '+r['output'])
    history=json.loads((run/'generation_history.json').read_text()) if (run/'generation_history.json').exists() else []
    check(len(history)==s['completed_matched_generations'],'Completed cohort history mismatch')
    for generation in history:
        for name,arm in s['arms'].items():
            check((run/'optimizer'/name/f"g-{generation['generation']-1:05d}-told/metadata.json").is_file(),'Missing matched optimizer update: '+name)
    for panel in s['panels']:
        rows=[]
        for r in panel['validation']:
            actual=saved.get(r['output']);check(actual==r,'Panel result mismatch: '+r['output']);rows.append(actual or r)
        check(panel_assessment(rows)==panel['assessment'],'Panel acceptance mismatch: '+panel['checkpoint_sha256'])
        check(all(r.get('checkpoint_sha256')==panel['checkpoint_sha256'] for r in rows if r['status']=='completed'),'Panel checkpoint mismatch')
    for metadata in (run/'optimizer').glob('*/*/metadata.json'):
        m=json.loads(metadata.read_text());verify(metadata.parent/'optimizer.pkl',m['optimizer_sha256'])
    for name in s['arms']:
        check((run/'optimizer'/name/'final/metadata.json').is_file(),'Missing final optimizer: '+name)
    write(out/'evidence.json',evidence);write(out/'cases.json',details)
    result=dict(audit_complete=True,integrity_passed=not problems,problems=problems,status_counts=dict(counts),
        deliberate_original_upper_bound_exceptions=dict(violations),
        completed_matched_generations=s['completed_matched_generations'],
        panels=[dict(labels=p['labels'],cap=p['cap'],assessment=p['assessment']) for p in s['panels']],
        scope='Saved-data integrity, independent coordinate/bound and score checks; planned metrics-only cases lack raw replay. No forward simulation, causal proof or automatic qualification.')
    write(out/'summary.json',result);print(json.dumps(result,indent=2),flush=True)
    return 0 if not problems else 2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    run,out=a.run.resolve(),a.output_dir.resolve()
    if run==out or run in out.parents or out in run.parents:raise ValueError('Separate audit output required')
    return audit(run,out)


if __name__=='__main__':raise SystemExit(main())
