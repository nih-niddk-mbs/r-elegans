"""One E069 fixed-body replay, without neural evolution or fitting."""
import argparse
import json
from pathlib import Path

import numpy as np

from r_elegans.body_timing import replay,summarize
from scripts.experiment_records import capture_run_record,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--speed',type=float,required=True)
    p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--held',action='store_true')
    a=p.parse_args();root=a.inputs.resolve();out=a.output_dir.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    before={name:file_sha256(root/name) for name in manifest['sha256']}
    if before!=manifest['sha256']:raise ValueError('Input capsule mismatch')
    repo=Path(__file__).resolve().parents[1]
    if out==repo or repo in out.parents or out==root or root in out.parents: p.error('Separate external output required')
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,repo,a)
    with np.load(root/'segment.npz',allow_pickle=False) as z:segment={k:z[k] for k in z.files}
    body=json.loads((root/'body.json').read_text())
    trace=replay(segment,body,a.speed,a.dt,held=a.held,
                 progress=lambda done,total:print(json.dumps(dict(steps=done,total=total)),flush=True))
    metrics=summarize(trace);match=None
    if a.speed==1 and a.dt==.002 and not a.held:
        match=dict(maximum_joint_error_rad=float(np.max(np.abs(trace['joints']-segment['joints']))),
                   maximum_filter_error=float(np.max(np.abs(trace['drive']-segment['drive']))))
        match['passed']=match['maximum_joint_error_rad']<=1e-7 and match['maximum_filter_error']<=1e-10
        if not match['passed']:raise RuntimeError('Normal-speed replay does not reproduce the source: '+str(match))
    after={name:file_sha256(root/name) for name in before}
    if before!=after:raise RuntimeError('Input changed during replay')
    np.savez_compressed(out/'trajectory.npz',**trace)
    summary=dict(experiment='E069',execution_valid=True,speed=a.speed,dt=a.dt,held=a.held,
                 metrics=metrics,source_match=match,input_manifest_sha256=file_sha256(root/'manifest.json'),
                 scope='Fixed-body timing diagnostic; externally replayed saved commands, no autonomous skill claim')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(execution_valid=True,speed=a.speed,dt=a.dt,held=a.held,
                          cycles=metrics['minimum_regional_joint_cycles'],source_match=match)),flush=True)


if __name__=='__main__':main()
