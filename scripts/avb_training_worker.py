"""Persistent E078 workers; fitting traces stay in memory, validation is retained."""
import json,time,shutil
from pathlib import Path
import numpy as np
from scripts.cma_sensorimotor_worker import initialize
from scripts.experiment_records import file_sha256

_CACHE={}


def evaluate(job):
    from scripts import cma_sensorimotor_worker as common
    from scripts.compare_rhythm_worker import verify_inputs
    from r_elegans.avb_training import protocol,membrane_probe,features,score
    from r_elegans.cma_curriculum import decode_groups
    from r_elegans.brain.conductance_network import save_network
    from r_elegans.behavior_control import bath_schedule,ControlProtocol
    from r_elegans.wall_budget import check_deadline,WallDeadline
    d=common._DATA;out=Path(job['output']);out.mkdir(parents=True,exist_ok=False)
    (out/'request.json').write_text(json.dumps(job,indent=2)+'\n')
    started=time.monotonic();result={}
    try:
        check_deadline(job['until'])
        if shutil.disk_usage(out).free<50*1024**3:
            raise common.DiskBudget('Less than 50 GiB free')
        x=decode_groups(job['z'],d['groups'],d['origin'],d['lower'],d['upper']);params=d['decode'](x)
        np.savez_compressed(out/'coordinates.npz',offsets=x,z=job['z'])
        save_network(out/'network.npz',params,d['fixed'],d['config'])
        variant=job.get('variant','ordinary');grid=job.get('grid','ordinary')
        duration,intervals,_=protocol(variant);rows={};hashes={};prefix=None
        dt=.001 if grid=='body-half' else .002
        brain=.05 if grid=='neural-half' else .1
        for condition in ('sham','inhibited'):
            key=(condition,variant,grid,job['until'])
            if key not in _CACHE:
                _CACHE[key]=d['factory'](dt,brain,wall_deadline=job['until'],
                    world_override=bath_schedule(ControlProtocol(sham=True)),
                    diagnostic_membrane=None if condition=='sham' else membrane_probe(d['model'].neuron_ids,variant))
            trace=_CACHE[key](params,np.linspace(0,duration,121),d['gains'])
            if not all(np.isfinite(v).all() for v in trace.values()):
                raise ValueError('Nonfinite state')
            rows[condition]=features(trace,d['model'].neuron_ids,variant)
            use=trace['dense_time_s']<intervals[0][0]
            before=tuple(trace[k][use].copy() for k in ('dense_voltage_mV','dense_joints'))
            if prefix is None:
                prefix=before
            elif any(np.max(abs(a-b))>1e-8 for a,b in zip(prefix,before)):
                raise ValueError('Pre-intervention states differ')
            if job.get('reference') and condition=='sham':
                with np.load(d['root']/'parent_training_trace.npz') as z:
                    clock=trace['dense_time_s'];end=min(clock[-1],z['dense_time_s'][-1])
                    a=clock<=end+1e-10;b=z['dense_time_s']<=end+1e-10
                    if not np.array_equal(clock[a],z['dense_time_s'][b]):
                        raise ValueError('Parent reference clock mismatch')
                    if any(np.max(abs(trace[k][a]-z[k][b]))>1e-8 for k in
                           ('dense_voltage_mV','dense_calcium_uM','dense_muscles','dense_filtered_drive','dense_joints')):
                        raise ValueError('Parent reference mismatch')
            if job.get('retain_traces'):
                estimated=2*sum(v.nbytes for v in trace.values())
                run=d['root'].parent
                used=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
                if used+estimated>30*1024**3:
                    raise common.DiskBudget('30 GiB total new-run cap would be breached')
                if shutil.disk_usage(out).free<50*1024**3+estimated:
                    raise common.DiskBudget('Trajectory would breach reserve')
                path=out/(condition+'.npz');np.savez_compressed(path,**trace)
                hashes[condition]=file_sha256(path)
            del trace
        parent=job.get('parent_features') or rows['sham']
        result=dict(status='completed',metrics=score(rows['sham'],rows['inhibited'],parent),features=rows,
            checkpoint_sha256=file_sha256(out/'network.npz'),trajectory_sha256=hashes,
            pre_intervention_match=True,training_trajectories_retained=bool(job.get('retain_traces')))
        verify_inputs(d['root'])
    except common.DiskBudget as e:result=dict(status='disk_budget',error=str(e))
    except WallDeadline as e:result=dict(status='time_budget',error=str(e))
    except (ValueError,RuntimeError,FloatingPointError) as e:result=dict(status='failed',error=str(e))
    result.update(output=str(out),elapsed_seconds=time.monotonic()-started,
                  variant=job.get('variant','ordinary'),grid=job.get('grid','ordinary'))
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
