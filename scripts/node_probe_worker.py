"""E096 worker: one imposed-conductance rollout from the pinned parent; never updates model weights."""
import json,shutil,time
from pathlib import Path
import numpy as np
from scripts import rial_sensory_worker as core
from scripts.sensory_motion_worker import initialize as motion_initialize
from scripts.experiment_records import file_sha256

_REFERENCE=None


def initialize(inputs,cpus,records):
    motion_initialize(inputs,dict(aiyr_2p5=[]),cpus,records)


def reference_trace(root):
    global _REFERENCE
    if _REFERENCE is None:
        ref=json.loads((root/'sensory_reference.json').read_text());path=Path(ref['path'])
        if file_sha256(path)!=ref['sha256']:raise ValueError('Parent reference changed')
        with np.load(path,allow_pickle=False) as z:_REFERENCE={k:z[k] for k in ('dense_time_s','dense_voltage_mV','dense_joints')}
    return _REFERENCE


def evaluate(job):
    from r_elegans.rial_sensory import protocol,world
    from r_elegans.sensory_motion import features
    from r_elegans.node_probe import membrane_probe
    from r_elegans.wall_budget import WallDeadline,check_deadline
    from scripts.compare_rhythm_worker import verify_inputs
    d=core._DATA;out=Path(job['output']);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    (out/'request.json').write_text(json.dumps(job,indent=2)+'\n');result=dict(status='failed');hashes={}
    try:
        check_deadline(job['until']);ids=d['model'].neuron_ids;p=protocol('nominal');params=d['decode'](d['origin'])
        grid=(job['probe'] or {}).get('grid','ordinary');cued=bool(job.get('cue'));dt=.001 if grid=='body-half' else .002;brain=.05 if grid=='neural-half' else .1
        probe=None if job['probe'] is None else membrane_probe(ids,job['probe'],p['events'][0])
        kwargs=dict(wall_deadline=job['until'],world_override=world('nominal',cued))
        if probe is not None:kwargs['diagnostic_membrane']=probe
        trace=d['factory'](dt,brain,**kwargs)(params,np.linspace(0,p['duration'],121),d['gains'].copy())
        ai=ids.index('AIYL')
        if grid=='ordinary':
            ref=reference_trace(d['root']);t=trace['dense_time_s'];end=min(t[-1],ref['dense_time_s'][-1])
            a=t<=end+1e-10;b=ref['dense_time_s']<=end+1e-10;cut=p['events'][0][0]
            if not np.array_equal(t[a],ref['dense_time_s'][b]):raise ValueError('Parent clock mismatch')
            whole=probe is None and not cued
            use=a if whole else a&(t<cut);use_ref=b if whole else b&(ref['dense_time_s']<cut)
            if any(np.max(abs(trace[k][use]-ref[k][use_ref]))>1e-8 for k in ('dense_voltage_mV','dense_joints')):
                raise ValueError('Trajectory differs from the parent reference where it must match')
        feature=features(trace,ids,'nominal',float(d['fixed'].release_threshold_mV[ai]),float(d['fixed'].release_slope_mV[ai]))
        if job['retain_traces']:
            if shutil.disk_usage(out).free<20*1024**3:raise RuntimeError('disk_budget: less than 20GiB free')
            path=out/'trace.npz';np.savez_compressed(path,**trace);hashes['trace']=file_sha256(path)
        verify_inputs(d['root'])
        result=dict(status='completed',features=feature,checkpoint_sha256=file_sha256(d['root']/'motion_parent.npz'),trajectory_sha256=hashes)
    except WallDeadline as e:result=dict(status='time_budget',error=str(e))
    except (ValueError,RuntimeError,FloatingPointError) as e:result=dict(status='failed',error=str(e))
    result.update(output=str(out),elapsed_seconds=time.monotonic()-started,probe=job['probe'],cue=bool(job.get('cue')))
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result
