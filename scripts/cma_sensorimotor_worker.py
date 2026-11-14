"""Persistent CPU worker for E077 paired sensory cadence evaluations."""
import json,os,time
from pathlib import Path
import numpy as np
from scripts.experiment_records import capture_run_record,file_sha256
from r_elegans.cma_curriculum import CONDITIONS,condition_protocol,features,score,decode_groups

_DATA=None
_CACHE={}


def initialize(inputs,schema,cpus,records,deadline):
    global _DATA
    import multiprocessing
    slot=(multiprocessing.current_process()._identity[-1]-1)%len(cpus)
    os.sched_setaffinity(0,cpus[slot])
    from scripts.neural_timing_setup import setup
    _DATA=setup(Path(inputs));_DATA['root']=Path(inputs);_DATA['groups']=schema
    _DATA['deadline']=deadline
    folder=Path(records)/f'worker-{slot}';folder.mkdir(parents=True,exist_ok=True)
    capture_run_record(folder,Path(__file__).resolve().parents[1],dict(inputs=inputs,cpus=cpus[slot],deadline=deadline,scope='Persistent E077 worker; tasks saved in evaluation request.json'))


def evaluate(job):
    from r_elegans.behavior_control import bath_schedule
    from r_elegans.brain.conductance_network import save_network
    from scripts.compare_rhythm_worker import verify_inputs
    from r_elegans.wall_budget import WallDeadline,check_deadline
    d=_DATA;out=Path(job['output']);out.mkdir(parents=True,exist_ok=False)
    (out/'request.json').write_text(json.dumps(job,indent=2)+'\n')
    started=time.monotonic();x=decode_groups(job['z'],d['groups'],d['origin'],d['lower'],d['upper']);params=d['decode'](x)
    np.savez_compressed(out/'coordinates.npz',offsets=x,z=job['z'])
    save_network(out/'network.npz',params,d['fixed'],d['config'])
    variant=job.get('variant','nominal');grid=job.get('grid','ordinary');result=None
    try:
        paired={};traces={};prefix={};reference_check=None
        for condition in CONDITIONS:
            check_deadline(job['until']);protocol=condition_protocol(condition,variant)
            key=(condition,variant,grid,job['until'])
            if key not in _CACHE:
                dt=.001 if grid=='body-half' else .002;brain=.05 if grid=='neural-half' else .1
                _CACHE[key]=d['factory'](dt,brain,wall_deadline=job['until'],world_override=bath_schedule(protocol))
            trace=_CACHE[key](params,np.linspace(0,protocol.duration,121),d['gains'])
            paired[condition]=features(trace,protocol)
            before=trace['dense_time_s']<40+protocol.shift
            prefix[condition]=(trace['dense_voltage_mV'][before],trace['dense_joints'][before])
            path=out/(condition+'.npz');np.savez_compressed(path,**trace);traces[condition]=file_sha256(path)
            if job.get('reference') and condition=='no_cue':
                with np.load(d['root']/'parent_training_trace.npz') as z:
                    use=trace['dense_time_s']<=z['dense_time_s'][-1]+1e-10
                    if not np.array_equal(trace['dense_time_s'][use],z['dense_time_s']):raise ValueError('Parent source time grid mismatch')
                    reference_check={k:float(np.max(abs(trace[k][use]-z[k]))) for k in ('dense_voltage_mV','dense_calcium_uM','dense_muscles','dense_filtered_drive','dense_joints')}
                    if any(v>1e-8 for v in reference_check.values()):raise ValueError('Parent trajectory does not reconstruct')
        errors={c:max(float(np.max(abs(a-b))) for a,b in zip(prefix[c],prefix['no_cue'])) for c in CONDITIONS}
        if max(errors.values())>1e-7:raise ValueError('Pre-cue state differs across paired episodes')
        parent=job.get('parent_features') or paired['no_cue']
        m=score(paired,parent,job['target'])
        result=dict(status='completed',metrics=m,paired_features=paired,pre_cue_errors=errors,reference_check=reference_check,
            checkpoint_sha256=file_sha256(out/'network.npz'),coordinate_sha256=file_sha256(out/'coordinates.npz'),trajectory_sha256=traces)
        verify_inputs(d['root'])
    except WallDeadline as e:result=dict(status='time_budget',error=str(e))
    except (ValueError,RuntimeError,FloatingPointError) as e:result=dict(status='failed',error=str(e))
    result.update(output=str(out),elapsed_seconds=time.monotonic()-started,variant=variant,grid=grid,target=job['target'])
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
