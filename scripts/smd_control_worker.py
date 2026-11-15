"""One frozen E082 SMD membrane inhibition/release trial, no optimization."""
import argparse,json,time,shutil
from pathlib import Path
import numpy as np
from scripts.experiment_records import capture_run_record,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--grid',default='ordinary',choices=('ordinary','neural-half','body-half'))
    p.add_argument('--variant',choices=('nominal','shift','repeat','long-hold'),default='nominal')
    p.add_argument('--until',type=float,required=True);a=p.parse_args()
    from scripts.neural_timing_setup import setup
    from scripts.compare_rhythm_worker import verify_inputs
    from r_elegans.brain.conductance_network import load_network
    from r_elegans.rhythm_comparison import assert_fixed_equal
    from r_elegans.smd_control import cases,protocol,membrane_probe
    from r_elegans.behavior_control import bath_schedule,ControlProtocol
    from r_elegans.wall_budget import check_deadline
    case=next(c for c in cases() if c['name']==a.case)
    duration,intervals,_=protocol(a.variant)
    root=a.inputs.resolve();out=a.output_dir.resolve();repo=Path(__file__).resolve().parents[1]
    for source in (root,repo):
        if out==source or source in out.parents or out in source.parents:raise ValueError('Separate output required')
    check_deadline(a.until);d=setup(root);verify_inputs(root)
    with np.load(root/'diagnostic_coordinates.npz',allow_pickle=False) as z:params=d['decode'](z['offsets'])
    actual,fixed,config=load_network(root/'diagnostic_model.npz');assert_fixed_equal(d['fixed'],fixed,d['config'],config)
    if any(not np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(params,actual)):raise ValueError('Selected model reconstruction mismatch')
    hook=None if a.case=='sham' else membrane_probe(d['model'].neuron_ids,case,a.variant)
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,repo,a)
    dt=.001 if a.grid=='body-half' else .002;brain=.05 if a.grid=='neural-half' else .1
    rollout=d['factory'](dt,brain,wall_deadline=a.until,world_override=bath_schedule(ControlProtocol(sham=True)),diagnostic_membrane=hook)
    started=time.monotonic();trace=rollout(params,np.linspace(0,duration,121),d['gains'])
    source=json.loads((root/'reference.json').read_text());reference=Path(source['path'])
    if file_sha256(reference)!=source['sha256']:raise ValueError('Source trajectory changed')
    with np.load(reference,allow_pickle=False) as z:
        clock=trace['dense_time_s'];refclock=z['dense_time_s']
        use=clock<=(min(60.,float(refclock[-1])) if a.case=='sham' and a.grid=='ordinary' else intervals[0][0]-.5*dt)
        errors={}
        for k in ('dense_voltage_mV','dense_joints','dense_muscles'):
            reference_values=z[k]
            interp=np.column_stack([np.interp(clock[use],refclock,reference_values[:,i]) for i in range(reference_values.shape[1])])
            errors[k]=float(np.max(abs(trace[k][use]-interp)))
        if a.grid=='ordinary' and max(errors.values())>1e-8:raise ValueError('Original ordinary/prefix no longer matches')
    estimated=2*sum(v.nbytes for v in trace.values())
    if sum(p.stat().st_size for p in root.parent.rglob('*') if p.is_file())+estimated>25*1024**3:
        raise ValueError('25GiB new-run trajectory cap')
    required=50*1024**3+estimated
    if shutil.disk_usage(out).free<required:raise ValueError('50GiB disk reserve required')
    path=out/'trajectory.npz';np.savez_compressed(path,**trace);verify_inputs(root)
    result=dict(experiment='E082',case=case,variant=a.variant,simulation_seconds=duration,grid=a.grid,checkpoint_sha256=file_sha256(root/'diagnostic_model.npz'),
        trajectory_sha256=file_sha256(path),source_reference_errors=errors,completed=True,elapsed_seconds=time.monotonic()-started,
        no_training=True,learned_stopping_qualified=False,intervention='Added SMD-only membrane conductance at-80mV; live voltages/recurrence/NMJ/body; no weights fit')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
