"""One frozen E080 closed-loop NMJ-output intervention, no optimization."""
import argparse,json,time,shutil
from pathlib import Path
import numpy as np
from scripts.experiment_records import capture_run_record,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--grid',default='ordinary',choices=('ordinary','neural-half','body-half'))
    p.add_argument('--until',type=float,required=True);a=p.parse_args()
    from scripts.neural_timing_setup import setup
    from scripts.compare_rhythm_worker import verify_inputs
    from r_elegans.brain.conductance_network import load_network
    from r_elegans.rhythm_comparison import assert_fixed_equal
    from r_elegans.nmj_source import release_hold,CASES
    from r_elegans.behavior_control import bath_schedule,ControlProtocol
    from r_elegans.wall_budget import check_deadline
    if a.case not in CASES:raise ValueError('Unknown case')
    root=a.inputs.resolve();out=a.output_dir.resolve();repo=Path(__file__).resolve().parents[1]
    for source in (root,repo):
        if out==source or source in out.parents or out in source.parents:raise ValueError('Separate output required')
    check_deadline(a.until);d=setup(root);verify_inputs(root)
    with np.load(root/'diagnostic_coordinates.npz',allow_pickle=False) as z:params=d['decode'](z['offsets'])
    actual,fixed,config=load_network(root/'diagnostic_model.npz');assert_fixed_equal(d['fixed'],fixed,d['config'],config)
    if any(not np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(params,actual)):raise ValueError('Selected model reconstruction mismatch')
    with np.load(root/'baseline_release.npz',allow_pickle=False) as z:mean=z['mean']
    hook=None if a.case=='sham' else release_hold(d['model'].neuron_ids,d['fixed'].release_threshold_mV,d['fixed'].release_slope_mV,mean,a.case)
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,repo,a)
    dt=.001 if a.grid=='body-half' else .002;brain=.05 if a.grid=='neural-half' else .1
    rollout=d['factory'](dt,brain,wall_deadline=a.until,world_override=bath_schedule(ControlProtocol(sham=True)),diagnostic_nmj_release=hook)
    started=time.monotonic();trace=rollout(params,np.linspace(0,60,121),d['gains'])
    source=json.loads((root/'reference.json').read_text());reference=Path(source['path'])
    if file_sha256(reference)!=source['sha256']:raise ValueError('Source trajectory changed')
    with np.load(reference,allow_pickle=False) as z:
        clock=trace['dense_time_s'];refclock=z['dense_time_s']
        use=clock<(60.01 if a.case=='sham' and a.grid=='ordinary' else 30.)
        errors={}
        for k in ('dense_voltage_mV','dense_joints','dense_muscles'):
            reference_values=z[k]
            interp=np.column_stack([np.interp(clock[use],refclock,reference_values[:,i]) for i in range(reference_values.shape[1])])
            errors[k]=float(np.max(abs(trace[k][use]-interp)))
        if a.grid=='ordinary' and max(errors.values())>1e-8:raise ValueError('Original ordinary/prefix no longer matches')
    required=50*1024**3+2*sum(v.nbytes for v in trace.values())
    if shutil.disk_usage(out).free<required:raise ValueError('50GiB disk reserve required')
    path=out/'trajectory.npz';np.savez_compressed(path,**trace);verify_inputs(root)
    result=dict(case=a.case,grid=a.grid,checkpoint_sha256=file_sha256(root/'diagnostic_model.npz'),
        trajectory_sha256=file_sha256(path),source_reference_errors=errors,completed=True,elapsed_seconds=time.monotonic()-started,
        no_training=True,learned_stopping_qualified=False,intervention='NMJ release held to own sham20..30s mean at30..40s; recurrent outputs stay live')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
