"""E096 saved-data audit: inputs, probe specs, score recomputation, retained-trace features and probe sign checks."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from r_elegans.node_probe import cases,GROUPS,TARGET,reduction_percent,reaches_target,minimum_doses
from r_elegans.sensory_motion import features,score
from scripts.audit_relay_experiments import write,close
from scripts.experiment_records import capture_run_record,file_sha256


def audit(run,out):
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,Path(__file__).resolve().parents[1],dict(run=str(run)))
    problems=[];evidence=[];counts=Counter()
    def check(ok,label):
        if not ok:problems.append(label)
    def verify(path,sha):
        actual=file_sha256(path);check(actual==sha,'Hash mismatch: '+str(path));evidence.append(dict(path=str(path),sha256=actual,expected=sha))
    s=json.loads((run/'summary.json').read_text());config=json.loads((run/'configuration.json').read_text())
    check(s.get('experiment')=='E096','Not an E096 run')
    manifest=json.loads((run/'inputs/manifest.json').read_text())
    for name,sha in manifest['sha256'].items():verify(run/'inputs'/name,sha)
    ref=json.loads((run/'inputs/sensory_reference.json').read_text());verify(Path(ref['path']),ref['sha256'])
    parent=run/'inputs/motion_parent.npz';verify(parent,config['parent_sha256'])
    with np.load(parent,allow_pickle=False) as z:
        ids=[str(x) for x in z['neuron_ids']];threshold=z['fixed_release_threshold_mV'];slope=z['fixed_release_slope_mV']
    planned=cases();check(close(config['planned'],planned),'Planned probe table changed')
    saved={};requests={};traces={}
    for request in sorted((run/'evaluations').glob('*/request.json')):
        folder=request.parent;j=json.loads(request.read_text());requests[str(folder)]=j
        try:
            r=json.loads((folder/'summary.json').read_text());saved[str(folder)]=r;counts[r['status']]+=1
            check(Path(r['output']).resolve()==folder and Path(j['output']).resolve()==folder,'Evaluation path mismatch: '+folder.name)
            check(r['probe']==j['probe'],'Probe differs between request and result: '+folder.name)
            if r['status']!='completed':continue
            check(r['checkpoint_sha256']==config['parent_sha256'],'Checkpoint is not the pinned parent: '+folder.name)
            hashes=r.get('trajectory_sha256',{});check(set(hashes)==({'trace'} if j['retain_traces'] else set()),'Raw retention policy mismatch: '+folder.name)
            if hashes:
                verify(folder/'trace.npz',hashes['trace'])
                with np.load(folder/'trace.npz',allow_pickle=False) as z:trace={k:z[k] for k in z.files}
                check(all(np.isfinite(v).all() for v in trace.values() if np.issubdtype(v.dtype,np.number)),'Nonfinite raw trace: '+folder.name)
                check(close(features(trace,ids,'nominal',float(threshold[ids.index('AIYL')]),float(slope[ids.index('AIYL')])),r['features']),'Raw features mismatch: '+folder.name)
                traces[str(folder)]=trace
        except Exception as e:problems.append(folder.name+': '+repr(e))
    check(sum(counts.values())==s['evaluation_count'],'Evaluation count mismatch')
    base_path=next((k for k,j in requests.items() if j['probe'] is None),None);check(base_path is not None and saved.get(base_path)==s.get('baseline'),'Baseline record mismatch')
    by_name={j['probe']['name']:k for k,j in requests.items() if j['probe']}
    check(set(by_name)<=({c['name'] for c in planned}),'Unplanned probe evaluated')
    check(all(saved.get(by_name.get(c['name'],''),{}).get('status') for c in planned) or not s.get('validation_complete'),'Planned probe missing from a complete run')
    check([r['output'] for r in s['probes']]==[by_name.get(c['name']) for c in planned] or not s.get('validation_complete'),'Probe order or paths differ from the plan')
    if base_path and saved[base_path]['status']=='completed':
        base=saved[base_path]['features'];table={row['name']:row for row in s['probe_table']}
        for c in planned:
            path=by_name.get(c['name']);row=table.get(c['name'])
            if path is None or row is None or saved[path]['status']!='completed':continue
            m=score(dict(no_cue=base,cue=saved[path]['features']),base,TARGET,1.)
            expected=dict(reduction_percent=reduction_percent(m),regional_motion_ratios=m['regional_motion_ratios'][0],safe=m['safe'],
                baseline_and_recovery_preserved=m['baseline_and_recovery_preserved'],regional_cycle_ratios=m['regional_cycle_ratios'][0],target_reached=reaches_target(m))
            check(close(expected,{k:row.get(k) for k in expected}),'Recorded score mismatch: '+c['name'])
            if path in traces and base_path in traces:
                t=traces[path]['dense_time_s'];b=traces[base_path];x=traces[path]
                check(np.array_equal(t,b['dense_time_s']),'Clock mismatch: '+c['name'])
                before=t<config['probe_window_s'][0];window=(t>=41.)&(t<46.)
                check(all(np.max(abs(x[k][before]-b[k][before]))<=1e-8 for k in ('dense_voltage_mV','dense_joints')),'Pre-probe mismatch: '+c['name'])
                idx=[ids.index(n) for n in GROUPS[c['group']]]
                change=float(np.mean(x['dense_voltage_mV'][window][:,idx]-b['dense_voltage_mV'][window][:,idx]))
                check(change>0 if c['reversal_mV']==0 else change<0,'Probe voltage change has the wrong sign: '+c['name'])
        check(close(s['minimum_doses_reaching_target_nS'],minimum_doses(s['probe_table'])),'Minimum-dose summary mismatch')
    write(out/'evidence.json',evidence)
    result=dict(audit_complete=True,integrity_passed=not problems,problems=problems,status_counts=dict(counts),retained_probe_traces=len(traces),
        scope='Saved-data integrity, independent score recomputation and retained-trace sign checks; no forward simulation, causal proof or qualification.')
    write(out/'summary.json',result);print(json.dumps(result,indent=2),flush=True)
    return 0 if not problems else 2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    run,out=a.run.resolve(),a.output_dir.resolve()
    if run==out or run in out.parents or out in run.parents:raise ValueError('Separate audit output required')
    return audit(run,out)


if __name__=='__main__':raise SystemExit(main())
