"""Read saved E124 response grids; no simulation, fitting or gate reclassification."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def compare(coarse,fine,targets,ids,training_pairs):
    a=np.asarray(coarse);b=np.asarray(fine)
    if a.shape!=b.shape or a.shape!=(4,len(targets),len(ids)) or not np.isfinite([a,b]).all():
        raise ValueError('Expected matching finite four-phase response grids')
    error=abs(a-b);index=np.unravel_index(np.argmax(error),error.shape)
    lookup={n:i for i,n in enumerate(targets)};neurons={n:i for i,n in enumerate(ids)}
    measured=np.zeros(a.shape[1:],dtype=bool)
    for target,receiver in training_pairs:
        if target in lookup and receiver in neurons and target!=receiver:measured[lookup[target],neurons[receiver]]=True
    mean_error=abs(a.mean(axis=0)-b.mean(axis=0))
    def region(mask):
        if not mask.any():return dict(pairs=0,available=False)
        return dict(pairs=int(mask.sum()),available=True,
            phasewise_max_difference=float(error[:,mask].max()),
            phase_mean_max_difference=float(mean_error[mask].max()),
            fine_phase_mean_max_response=float(abs(b.mean(axis=0)[mask]).max()))
    maximum=float(error.max());tol=float(1e-7+.1*abs(b).max())
    return dict(phasewise_max_difference=maximum,original_tolerance=tol,
        original_tolerance_ratio=maximum/tol,
        phase_mean_max_difference=float(mean_error.max()),
        fine_phase_mean_max_response=float(abs(b.mean(axis=0)).max()),
        per_phase_max_difference=error.max(axis=(1,2)).tolist(),
        worst=dict(phase=int(index[0]),target=targets[index[1]],receiver=ids[index[2]],
            in_training_objective=bool(measured[index[1],index[2]]),
            coarse=float(a[index]),fine=float(b[index])),
        training_observed=region(measured),outside_training_objective=region(~measured),
        scope='Descriptive localization only; phase-mean values introduce no replacement gate')


def diagnose(root):
    root=Path(root).resolve();hashes={};rows=[]
    def read_json(path):
        hashes[str(path)]=sha256(path);return json.loads(path.read_text())
    def read_npz(path,key):
        hashes[str(path)]=sha256(path)
        with np.load(path,allow_pickle=False) as z:return z[key]
    manifest=read_json(root/'manifest.json');folds=read_json(root/'folds.json')
    ids=read_npz(root/'parent.npz','neuron_ids').tolist()
    for arm in ('network','peptide'):
        for i,fold in enumerate(folds):
            folder=root/arm/f'fold-{i}';s=read_json(folder/'summary.json')
            paths=[Path(p) for p in s['prediction_paths']]+[Path(s['quarter_path'])]
            data=[];metadata=[]
            for path in paths:
                metadata.append(read_json(path/'metrics.json'))
                data.append(read_npz(path/'prediction.npz','response'))
            for pair,(lo,hi) in enumerate(((0,1),(1,2))):
                targets=metadata[hi]['targets'];source=metadata[lo]['targets']
                a=data[lo][:,[source.index(t) for t in targets]]
                row=dict(arm=arm,fold=i,grids=[metadata[lo]['level'],metadata[hi]['level']],
                    original_check=s['response_checks'][pair],
                    diagnostics=compare(a,data[hi],targets,ids,fold['stats']['train']['pairs']))
                if not np.isclose(row['diagnostics']['phasewise_max_difference'],row['original_check']['maximum_difference'],rtol=1e-12,atol=1e-15):
                    raise ValueError('Saved response maximum does not reproduce original: '+str(folder))
                rows.append(row)
    unchanged=all(sha256(p)==h for p,h in hashes.items())
    if not unchanged:raise ValueError('Source changed during read-only analysis')
    return dict(run=str(root),experiment=manifest['experiment'],source_sha256=hashes,source_unchanged=unchanged,
        comparisons=rows,acceptance_gates_changed=False,model_promoted=False,
        scope='Saved-data response localization only; no new forward, solver attribution, fitting or qualification')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path)
    p.add_argument('--output-file',type=Path,help='Optional new JSON report outside source run; refuses overwrite');a=p.parse_args()
    result=diagnose(a.run)
    for row in result['comparisons']:
        if row['original_check']['passed']:continue
        d=row['diagnostics']
        print(json.dumps(dict(arm=row['arm'],fold=row['fold'],grids=row['grids'],
            tolerance_ratio=d['original_tolerance_ratio'],phasewise=d['phasewise_max_difference'],
            phase_mean=d['phase_mean_max_difference'],worst=d['worst'],
            training_observed=d['training_observed']),sort_keys=True))
    if a.output_file:
        out=a.output_file.resolve();root=a.run.resolve()
        if out==root or root in out.parents:raise ValueError('Report must be outside source run')
        with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print('Report:',out)
    print('Source unchanged:',result['source_unchanged'],'; original gates unchanged; no simulation or fitting')


if __name__=='__main__':main()
