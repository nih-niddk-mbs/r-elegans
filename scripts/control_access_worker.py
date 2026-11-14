"""One E076 diagnostic forward rollout; never updates model weights."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scripts.experiment_records import capture_run_record,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--grid',choices=['ordinary','neural-half','body-half'],default='ordinary')
    p.add_argument('--until',type=float,required=True);a=p.parse_args();root=a.inputs.resolve();out=a.output_dir.resolve();repo=Path(__file__).resolve().parents[1]
    for source in (repo,root):
        if out==source or source in out.parents or out in source.parents:raise ValueError('Separate external output required')
    from scripts.neural_timing_setup import setup
    from scripts.compare_rhythm_worker import verify_inputs
    from r_elegans.control_access import cases,membrane_probe
    from r_elegans.brain.conductance_network import load_network
    from r_elegans.rhythm_comparison import assert_fixed_equal
    from r_elegans.behavior_control import ControlProtocol,bath_schedule,CONTROL_GROUPS
    from r_elegans.wall_budget import check_deadline
    check_deadline(a.until);d=setup(root);case=next(c for c in cases() if c['name']==a.case)
    with np.load(root/'access_coordinates.npz') as z:params=d['decode'](z['offsets'])
    actual,fixed,config=load_network(root/'access_model.npz');assert_fixed_equal(d['fixed'],fixed,d['config'],config)
    if any(not np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(params,actual)):raise ValueError('Probe source coordinate mismatch')
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,repo,a)
    dt=.001 if a.grid=='body-half' else .002;brain=.05 if a.grid=='neural-half' else .1
    configuration=dict(experiment='E076',case=case,grid=a.grid,dt=dt,brain_step_ms=brain,onset_s=30.,offset_s=40.,simulation_seconds=60.,
        targets=list(CONTROL_GROUPS.get(case['group'],())),
        checkpoint_sha256=file_sha256(root/'access_model.npz'),input_manifest_sha256=file_sha256(root/'manifest.json'),
        no_training=True,scope='Imposed conductance diagnostic, not learned stopping')
    (out/'configuration.json').write_text(json.dumps(configuration,indent=2)+'\n')
    probe=None if case['name']=='sham' else membrane_probe(d['model'].neuron_ids,case)
    rollout=d['factory'](dt,brain,wall_deadline=a.until,world_override=bath_schedule(ControlProtocol(sham=True)),diagnostic_membrane=probe)
    started=time.monotonic();trace=rollout(params,np.linspace(0,60,121),d['gains'])
    reference_check=None
    if case['name']=='sham' and a.grid=='ordinary':
        with np.load(root/'access_reference.npz') as z:
            use=z['dense_time_s']<=trace['dense_time_s'][-1]+1e-10
            if not np.array_equal(z['dense_time_s'][use],trace['dense_time_s']):raise ValueError('Sham source time mismatch')
            reference_check={k:float(np.max(abs(trace[k]-z[k][use]))) for k in ('dense_voltage_mV','dense_calcium_uM','dense_muscles','dense_filtered_drive','dense_joints')}
            if any(v>1e-8 for v in reference_check.values()):raise ValueError('Sham no longer reconstructs E074 reference')
    np.savez_compressed(out/'trajectory.npz',**trace);verify_inputs(root)
    result=dict(**configuration,completed=True,source_reference_maximum_errors=reference_check,elapsed_seconds=time.monotonic()-started,
        trajectory_sha256=file_sha256(out/'trajectory.npz'))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
