"""Read-only E097 calibration: motion reduction against the node's own release change, compared with the cue run."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.special import expit
from r_elegans.node_probe import ALL_GROUPS,CALIBRATION_GROUPS,predict_at
from scripts.experiment_records import capture_run_record,file_sha256

WINDOW=(41.,46.)


def load(folder,expected):
    path=Path(folder)/'trace.npz'
    if file_sha256(path)!=expected:raise ValueError('Trace hash mismatch: '+str(path))
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in ('dense_time_s','dense_voltage_mV')}


def analyze(run,out=None):
    run=Path(run).resolve();repo=Path(__file__).resolve().parents[1]
    out=Path(out).resolve() if out else run.with_name(run.name+'-calibration-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if any(out==r or r in out.parents or out in r.parents for r in (run,repo)):raise ValueError('Separate external output required')
    s=json.loads((run/'summary.json').read_text());receipt=json.loads((run/'budget_receipt.json').read_text())
    audit=json.loads(run.with_name(run.name+'-audit').joinpath('summary.json').read_text())
    if s.get('experiment')!='E097' or not s.get('validation_complete') or not receipt.get('all_steps_finished') or not audit.get('integrity_passed'):
        raise ValueError('Completed E097 with a clean saved-data audit required')
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,repo,dict(run=str(run),method='saved-data-only'))
    with np.load(run/'inputs/motion_parent.npz',allow_pickle=False) as z:
        ids=[str(x) for x in z['neuron_ids']];th=z['fixed_release_threshold_mV'];sl=z['fixed_release_slope_mV']
    def release(trace,names):
        t=trace['dense_time_s'];use=(t>=WINDOW[0])&(t<WINDOW[1]);i=[ids.index(n) for n in names]
        return expit((trace['dense_voltage_mV'][use][:,i]-th[i])/sl[i]).mean(axis=0),trace['dense_voltage_mV'][use][:,i].mean(axis=0)
    base=load(Path(s['baseline']['output']),s['baseline']['trajectory_sha256']['trace'])
    cue=load(Path(s['cue_run']['output']),s['cue_run']['trajectory_sha256']['trace'])
    table={r['name']:r for r in s['probe_table'] if r['status']=='completed' and not r.get('grid')}
    result=dict(run=str(run),window_s=list(WINDOW),cue_reduction_percent=s['cue_reduction_percent'],groups={})
    for group in CALIBRATION_GROUPS:
        names=ALL_GROUPS[group];rb,vb=release(base,names);rc,vc=release(cue,names)
        cue_delta=float(np.mean(rc-rb));sign='excite' if cue_delta>=0 else 'inhibit'
        curve=[]
        for r in table.values():
            if r['group']!=group or r['sign']!=sign:continue
            trace=load(r['output'],next(p for p in s['probes'] if p['output']==r['output'])['trajectory_sha256']['trace'])
            rp,vp=release(trace,names);curve.append(dict(dose_nS=r['dose_nS'],node_release_change=float(np.mean(rp-rb)),node_voltage_change_mV=float(np.mean(vp-vb)),
                reduction_percent=r['reduction_percent'],preserved=r['baseline_and_recovery_preserved']))
        curve.sort(key=lambda c:c['dose_nS']);x=abs(cue_delta);pairs=[(abs(c['node_release_change']),c) for c in curve]
        predicted={};status=None
        for k,label in enumerate(('tail','middle','head')):
            value,status=predict_at([(a,c['reduction_percent'][k]) for a,c in pairs],x);predicted[label]=value
        dose,_=predict_at([(a,c['dose_nS']) for a,c in pairs],x)
        result['groups'][group]=dict(cue_node_release_change=cue_delta,cue_node_voltage_change_mV=float(np.mean(vc-vb)),matching_probe_sign=sign,
            cue_per_neuron_release_change={n:float(a-b) for n,a,b in zip(names,rc,rb)},probe_curve=curve,
            probe_equivalent_dose_nS=dose,predicted_reduction_percent_at_cue_node_change=predicted,prediction_status=status)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Calibration report:',out/'summary.json');print('Actual cue reduction % (tail/middle/head):',[round(x,4) for x in s['cue_reduction_percent']])
    print('group, cue release change, matching sign, probe-equivalent dose nS, predicted reduction % tail/middle/head, status')
    for g,r in result['groups'].items():
        p=r['predicted_reduction_percent_at_cue_node_change']
        print(g,f"{r['cue_node_release_change']:.3g}",r['matching_probe_sign'],f"{r['probe_equivalent_dose_nS']:.3g}",[round(p[k],3) for k in ('tail','middle','head')],r['prediction_status'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output-dir',type=Path);a=p.parse_args();analyze(a.run,a.output_dir)


if __name__=='__main__':main()
