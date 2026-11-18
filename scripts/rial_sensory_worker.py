"""E085/E086 persistent fitting workers; diagnostic clamps absent from training."""
import json,os,time,shutil
from pathlib import Path
import numpy as np
from scripts.experiment_records import capture_run_record,file_sha256

_DATA=None
_CACHE={}


def setup(inputs):
    from scripts.neural_timing_setup import setup as original
    from r_elegans.brain.conductance_network import load_network
    from r_elegans.rhythm_comparison import assert_fixed_equal
    d=original(Path(inputs))
    with np.load(Path(inputs)/'sensory_parent_coordinates.npz',allow_pickle=False) as z:x=z['offsets']
    parent,fixed,config=load_network(Path(inputs)/'sensory_parent.npz');assert_fixed_equal(d['fixed'],fixed,d['config'],config)
    rebuilt=d['decode'](x)
    if any(not np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(rebuilt,parent)):raise ValueError('E078 selected parent reconstruction mismatch')
    d.update(origin=x,base=parent,root=Path(inputs));return d


def initialize(inputs,schema,cpus,records):
    global _DATA
    import multiprocessing
    slot=(multiprocessing.current_process()._identity[-1]-1)%len(cpus);os.sched_setaffinity(0,cpus[slot])
    _DATA=setup(Path(inputs));_DATA['groups']=schema
    experiment=json.loads((Path(inputs)/'manifest.json').read_text()).get('consumer_experiment','E085')
    capture_run_record(Path(records)/f'worker-{slot}',Path(__file__).resolve().parents[1],dict(inputs=inputs,cpus=cpus[slot],experiment=experiment))


def evaluate(job):
    from scipy.special import expit
    from r_elegans.rial_sensory import protocol,world,features,score,replay_probe
    from r_elegans.cma_curriculum import decode_groups
    from r_elegans.brain.conductance_network import save_network
    from r_elegans.wall_budget import WallDeadline,check_deadline
    from scripts.compare_rhythm_worker import verify_inputs
    relay_objective=job.get('objective')=='relay-range'
    release_objective=job.get('objective')=='rial-release'
    if release_objective:
        from r_elegans.rial_release import features as release_features,score
    if relay_objective:
        from r_elegans.relay_range import features as relay_features,score as relay_score
        def score(paired,parent,target,margin=1.):
            return relay_score(paired,parent,target,margin,response_aggregation=job.get('response_aggregation','mean'))
    d=_DATA;out=Path(job['output']);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    (out/'request.json').write_text(json.dumps(job,indent=2)+'\n');mode=job.get('mode','normal')
    artifacts={};hashes={}
    try:
        if mode not in ('normal','sensory-blocked','rial-replay','relay-replay'):raise ValueError('Unknown evaluation mode')
        if shutil.disk_usage(out).free<50*1024**3:raise RuntimeError('disk_budget: less than50GiB free')
        x=decode_groups(job['z'],d['groups'],d['origin'],d['lower'],d['upper']);params=d['decode'](x)
        np.savez_compressed(out/'coordinates.npz',offsets=x,z=job['z']);save_network(out/'network.npz',params,d['fixed'],d['config'])
        modelsha=file_sha256(out/'network.npz')
        artifacts=dict(checkpoint_sha256=modelsha,coordinates_sha256=file_sha256(out/'coordinates.npz'))
        variant=job.get('variant','nominal');grid=job.get('grid','ordinary');p=protocol(variant)
        dt=.001 if grid=='body-half' else .002;brain=.05 if grid=='neural-half' else .1
        paired={};prefix=None;hashes={};reference_release=None;replay=None
        conditions=('no_cue','cue','no_cue_replay','cue_replay') if mode in ('rial-replay','relay-replay') else ('no_cue','cue')
        gains=d['gains'].copy()
        if mode=='sensory-blocked':gains[0]=0.
        for condition in conditions:
            check_deadline(job['until']);cued=condition in ('cue','cue_replay')
            kwargs=dict(wall_deadline=job['until'],world_override=world(variant,cued))
            if condition.endswith('_replay'):kwargs['diagnostic_chemical_release']=replay
            key=(condition,variant,grid,job['until'],modelsha if condition.endswith('_replay') else None)
            if key not in _CACHE:_CACHE[key]=d['factory'](dt,brain,**kwargs)
            trace=_CACHE[key](params,np.linspace(0,p['duration'],121),gains)
            if relay_objective:
                ai=d['model'].neuron_ids.index('AIYL')
                paired[condition]=relay_features(trace,d['model'].neuron_ids,variant,
                    float(d['fixed'].release_threshold_mV[ai]),float(d['fixed'].release_slope_mV[ai]))
            elif release_objective:
                ri=d['model'].neuron_ids.index('RIAL')
                paired[condition]=release_features(trace,d['model'].neuron_ids,variant,
                    float(d['fixed'].release_threshold_mV[ri]),float(d['fixed'].release_slope_mV[ri]))
            else:paired[condition]=features(trace,d['model'].neuron_ids,variant)
            before=trace['dense_time_s']<p['events'][0][0]
            current=tuple(trace[k][before].copy() for k in ('dense_voltage_mV','dense_joints'))
            if prefix is None:prefix=current
            elif any(np.max(abs(a-b))>1e-8 for a,b in zip(prefix,current)):raise ValueError('Pre-cue trajectory mismatch')
            if condition=='no_cue':
                if job.get('reference'):
                    ref=json.loads((d['root']/'sensory_reference.json').read_text());path=Path(ref['path'])
                    if file_sha256(path)!=ref['sha256']:raise ValueError('Parent reference changed')
                    with np.load(path,allow_pickle=False) as z:
                        end=min(z['dense_time_s'][-1],trace['dense_time_s'][-1]);a=trace['dense_time_s']<=end+1e-10;b=z['dense_time_s']<=end+1e-10
                        if not np.array_equal(trace['dense_time_s'][a],z['dense_time_s'][b]):raise ValueError('Parent clock mismatch')
                        if any(np.max(abs(trace[k][a]-z[k][b]))>1e-8 for k in ('dense_voltage_mV','dense_muscles','dense_joints')):raise ValueError('Parent source mismatch')
                if mode in ('rial-replay','relay-replay'):
                    source_name='AIYL' if mode=='relay-replay' else 'RIAL'
                    ri=d['model'].neuron_ids.index(source_name);reference_release=expit((trace['dense_voltage_mV'][:,ri]-float(d['fixed'].release_threshold_mV[ri]))/float(d['fixed'].release_slope_mV[ri]))
                    replay=replay_probe(d['model'].neuron_ids,np.asarray(d['fixed'].chemical_mask),trace['dense_time_s'],reference_release,p['events'][0][0],dt,source_name=source_name,post_names=('RIAL',) if mode=='relay-replay' else None)
            if job.get('retain_traces'):
                estimated=2*sum(v.nbytes for v in trace.values());used=sum(p.stat().st_size for p in d['root'].parent.rglob('*') if p.is_file())
                if used+estimated>30*1024**3 or shutil.disk_usage(out).free<50*1024**3+estimated:raise RuntimeError('disk_budget: reserve or30GiB new-run cap')
                path=out/(condition+'.npz');np.savez_compressed(path,**trace);hashes[condition]=file_sha256(path)
            del trace
        parent=job.get('parent_features') or paired['no_cue']
        metrics=score(paired,parent,job['target'],job.get('margin',1.))
        mediation=None
        if mode in ('rial-replay','relay-replay'):
            counter=score(dict(no_cue=paired['no_cue_replay'],cue=paired['cue_replay']),parent,job['target'])
            ratio_key='body_response_amplitude_ratios' if release_objective or relay_objective else 'response_amplitude_ratios'
            original=metrics[ratio_key];replayed=counter[ratio_key]
            mediation=dict(replay_amplitude_ratios=replayed,baseline_preserved=counter['baseline_and_recovery_preserved'],
                original_body_attenuation_nontrivial=bool(all(1-a>1e-6 for a in original)),
                cue_attenuation_reduced=bool(counter['safe'] and counter['useful_response_preserved'] and counter['baseline_and_recovery_preserved'] and all(1-a>1e-6 and max(0.,1-r)<=.25*(1-a) for a,r in zip(original,replayed))),
                scope=('Final diagnostic only: matched no-cue AIYL release replay on existing RIAL chemical edge' if mode=='relay-replay' else 'Final diagnostic only: matched no-cue RIAL release replay on existing SMD chemical edges from cue onset; not used in fitting'))
        verify_inputs(d['root'])
        result=dict(status='completed',metrics=metrics,paired_features=paired,checkpoint_sha256=modelsha,trajectory_sha256=hashes,
                    mediation=mediation,pre_cue_match=True)
    except WallDeadline as e:result=dict(status='time_budget',error=str(e))
    except (ValueError,RuntimeError,FloatingPointError) as e:result=dict(status='disk_budget' if str(e).startswith('disk_budget:') else 'failed',error=str(e))
    result.update(artifacts);result.setdefault('trajectory_sha256',hashes)
    result.update(output=str(out),elapsed_seconds=time.monotonic()-started,variant=job.get('variant','nominal'),grid=job.get('grid','ordinary'),mode=mode)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result
