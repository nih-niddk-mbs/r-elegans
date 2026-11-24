"""Read-only localization of E126 cycle-sampling variability; no simulation or fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.experiment_records import file_sha256


def summarize(response,targets,ids,measured=None):
    y=np.asarray(response)
    if y.shape!=(16,len(targets),len(ids)) or not np.isfinite(y).all():raise ValueError('Expected finite four-cycle/four-phase response array')
    cycles=y.reshape(4,4,len(targets),len(ids)).mean(1);mean=cycles.mean(0)
    delta=np.stack([abs(np.delete(cycles,i,0).mean(0)-mean) for i in range(4)])
    c,t,r=np.unravel_index(np.argmax(delta),delta.shape)
    error=float(delta.max());tol=float(1e-7+.1*abs(mean).max());fail=delta.max(0)>tol
    if measured is not None and measured.shape!=fail.shape:raise ValueError('Measured support shape')
    return dict(original_screen=dict(leave_one_cycle_out_max_difference=error,tolerance=tol,passed=error<=tol),
        pairs_above_tolerance=int(fail.sum()),pairs_total=int(fail.size),
        measured_pairs_above_tolerance=None if measured is None else int((fail&measured).sum()),
        measured_pairs_total=None if measured is None else int(measured.sum()),
        worst=dict(omitted_cycle=int(c),target=targets[t],receiver=ids[r],self_response=targets[t]==ids[r],
            measured_in_development_pool=None if measured is None else bool(measured[t,r]),
            cycle_means=cycles[:,t,r].tolist(),four_cycle_mean=float(mean[t,r]),
            phase_responses_by_cycle=y[:,t,r].reshape(4,4).tolist()),
        scope='Deterministic variability localization; no new gate, CI, biology or model qualification')


def diagnose(root):
    root=Path(root).resolve();hashes={}
    def read_json(p):hashes[str(p)]=file_sha256(p);return json.loads(p.read_text())
    def array(p,key,expected=None):
        hashes[str(p)]=file_sha256(p)
        if expected is not None and hashes[str(p)]!=expected:raise ValueError('Artifact hash '+str(p))
        with np.load(p,allow_pickle=False) as z:return z[key]
    audit=read_json(root/'audit.json');m=read_json(root/'manifest.json')
    if not all(audit.get(k) for k in ('audit_complete','execution_complete','integrity_passed')):raise ValueError('Completed integrity-passed source audit required')
    folds_path=Path(m['source'])/'folds.json';support=None;unavailable=[]
    if folds_path.exists():
        original=read_json(Path(m['source'])/'manifest.json');folds=read_json(folds_path)
        if hashes[str(folds_path)]!=original['input_sha256'][str(folds_path)]:raise ValueError('Development support input hash')
        support={tuple(pair) for fold in folds for pair in fold['stats']['train']['pairs'] if pair[0]!=pair[1]}
    else:unavailable.append('Development measured-pair support unavailable')
    rows=[]
    for arm in ('network','peptide'):
        ids=array(root/arm/'parent-model.npz','neuron_ids',m['models'][arm]['parent_sha256']).tolist()
        mask=None if support is None else np.array([[(t,r) in support for r in ids] for t in m['full_targets']])
        for folder in sorted((root/arm).glob('full-*')):
            if not (folder/'metrics.json').exists():continue
            metrics=read_json(folder/'metrics.json');path=folder/'response.npz'
            values=array(path,'response',audit['artifact_sha256'][str(path)])
            result=summarize(values,m['full_targets'],ids,mask)
            if result['original_screen']!={k:v for k,v in metrics['sampling'].items() if k!='scope'}:raise ValueError('Original sampling arithmetic mismatch')
            phases=read_json(folder/'phases.json');result['sampled_cycles']=phases['selected_cycles']
            rows.append(dict(arm=arm,level=int(folder.name.split('-')[-1]),**result))
    if len(rows)!=4:raise ValueError('Four completed arm/clock grids required')
    if not all(file_sha256(p)==h for p,h in hashes.items()):raise ValueError('Source changed during read-only analysis')
    return dict(run=str(root),source_unchanged=True,source_sha256=hashes,grids=rows,unavailable=unavailable,
        acceptance_gates_changed=False,model_promoted=False,simulation_performed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output-file',type=Path);a=p.parse_args()
    result=diagnose(a.run)
    for row in result['grids']:
        print(json.dumps({k:row[k] for k in ('arm','level','original_screen','pairs_above_tolerance','measured_pairs_above_tolerance','worst')},sort_keys=True))
    if a.output_file:
        path=a.output_file.resolve();root=a.run.resolve()
        if path==root or root in path.parents:raise ValueError('New report must be outside source run')
        with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print('Report:',path)
    print('Source unchanged; original gates unchanged; no simulation or fitting')


if __name__=='__main__':main()
