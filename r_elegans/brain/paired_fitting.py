"""Shared full-network rollout and strictly training-fold paired windows."""
import json
import math
from pathlib import Path

import numpy as np

from r_elegans.data.paired_observations import source_bends, model_source_tangents
from r_elegans.data.integrity import sha256_file as file_sha256


def training_windows(root, uids, frames=8, start=0):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if file_sha256(root/'splits.json') != manifest['split_sha256']:
        raise ValueError('Frozen split checksum mismatch')
    splits = json.loads((root/'splits.json').read_text())['assignments']
    records = {r['uid']:r for r in manifest['records']}
    if len(set(uids)) != len(uids) or not uids:
        raise ValueError('Unique training UIDs required')
    if frames < 8 or start < 0:
        raise ValueError('Need >=8 frames and nonnegative start')
    windows = []
    for uid in uids:
        r = records[uid]
        if r['fold'] != 'train' or splits[uid] != 'train':
            raise ValueError('Fit accepts training-fold recordings only: '+uid)
        if r['status'] != 'prepared_not_fit_admitted':
            raise ValueError('Recording was quarantined')
        path = root/r['targets_file']
        if path.resolve().parent != root.resolve() or file_sha256(path) != r['targets_sha256']:
            raise ValueError('Prepared target checksum/path mismatch')
        with np.load(path,allow_pickle=False) as z:
            sl = slice(start,start+frames)
            times, segments = z['time_s'][sl],z['segment_id'][sl]
            if len(times) != frames or len(np.unique(segments)) != 1 or np.any(np.diff(times)<=0):
                raise ValueError('Window must remain within one contiguous source segment')
            a=z['body_angle_absolute'][sl,:7]
            vector=np.stack((np.cos(a),np.sin(a)),axis=-1)
            u,v=vector[:,:4].sum(axis=1),vector[:,4:7].sum(axis=1)
            reconstructed=np.abs(np.arctan2(u[:,0]*v[:,1]-u[:,1]*v[:,0],np.sum(u*v,axis=1)))
            difference=np.abs(reconstructed-np.abs(z['head_angle'][sl]))
            source_head_check=float(np.nanmax(difference)) if np.isfinite(difference).any() else None
            window = dict(uid=uid,time_s=times-times[0],indices=z['neuron_indices'],
                          fluorescence=z['fluorescence'][sl],valid=z['fluorescence_valid'][sl],
                          bends=source_bends(z['body_angle_absolute'][sl]),source_sha256=r['targets_sha256'],
                          source_head_chord_max_difference_rad=source_head_check)
        if window['time_s'][-1] > 30:
            raise ValueError('Initial pilot windows must be <=30s')
        windows.append(window)
    return windows, manifest


def make_rollout(fixed, config, parent, model, *, medium='wet-agar',dt=.002,
                 brain_step_ms=.1,burn_in=2.,sampling="boxcar"):
    """Endogenous brain -> frozen NMJ -> body -> proprioception, no pacing.

    Media are explicitly mismatched mechanical hypotheses, not source-condition
    calibration or certified error bounds. Fixed chemical gains are zero because
    recording-specific food/stimulus transduction is not yet modeled.
    """
    import jax
    from r_elegans.body import physical_muscle_body_params, build_muscle_projection
    from r_elegans.body.closed_loop import initial_embodied_state, embodied_neural_step
    from r_elegans.body.actuation import muscle_body_step
    from r_elegans.body.neuromuscular import project_muscles_to_joints
    from r_elegans.body.embodied_sensory import (EmbodiedSensoryConfig,default_chemical_world,build_sensory_maps)
    from r_elegans.body.wet_surface import wet_agar_drag,surface_body_step
    if medium not in ('liquid','wet-agar'):
        raise ValueError('Unknown mechanical hypothesis')
    if not all(np.isfinite(v) and v>0 for v in (dt,brain_step_ms)) or not np.isfinite(burn_in) or burn_in<0:
        raise ValueError('Invalid integration settings')
    if dt>.002 or brain_step_ms>.1:
        raise ValueError('Use dt<=.002s and brain substep<=.1ms')
    if np.any(np.asarray(fixed.bias_pA)!=0):
        raise ValueError('Zero-bias network required')
    if sampling not in ('point','boxcar'):raise ValueError('Unknown sample kernel')
    body = physical_muscle_body_params(12,medium='liquid' if medium=='liquid' else 'agar')
    sc = EmbodiedSensoryConfig(chemical_gain_nS=0.)
    world = default_chemical_world()
    maps = build_sensory_maps(model,sc)
    projection = build_muscle_projection(11,head_at_end=True)
    signed = np.asarray(projection.dorsal_weights)-np.asarray(projection.ventral_weights)
    substeps = math.ceil(dt*1000/brain_step_ms)
    def neural(state,params):
        return embodied_neural_step(state,params,fixed,config,parent,model,body,world,maps,dt,substeps,sc)
    fast_neural = jax.jit(neural)
    def liquid_step(state,params):
        brain,sensory,muscles,_,_ = neural(state,params)
        dorsal,ventral = project_muscles_to_joints(muscles,projection)
        newbody = muscle_body_step(state.brain_body.body,body,dorsal,ventral,dt)
        state = state._replace(brain_body=state.brain_body._replace(brain=brain,body=newbody),sensory=sensory)
        return state,(brain.indicator,newbody.joint_angles)
    @jax.jit
    def liquid_chunk(state,params):
        return jax.lax.scan(lambda s,_:liquid_step(s,params),state,None,length=100)
    law = wet_agar_drag()
    def run(params,times):
        times = np.asarray(times,dtype=float)
        if times.ndim!=1 or not len(times) or times[0]!=0 or np.any(np.diff(times)<=0):
            raise ValueError('Increasing relative observation times starting at zero required')
        state = initial_embodied_state(fixed,config,body,world,sc)
        indicators = [np.asarray(state.brain_body.brain.indicator)]
        angles = [np.asarray(state.brain_body.body.joint_angles)]
        requested = burn_in+times
        mid = (requested[1:]+requested[:-1])/2
        edges = np.r_[requested[0]-(requested[1]-requested[0])/2,mid,requested[-1]+(requested[-1]-requested[-2])/2]
        if sampling=='boxcar' and edges[0]<0:raise ValueError('Burn-in must cover first exposure bin')
        steps = math.ceil((edges[-1] if sampling=='boxcar' else requested[-1])/dt)
        if medium=='liquid':
            for _ in range(math.ceil(steps/100)):
                state,trace = liquid_chunk(state,params)
                indicators.extend(np.asarray(trace[0])); angles.extend(np.asarray(trace[1]))
        else:
            state = state._replace(brain_body=state.brain_body._replace(body=jax.tree_util.tree_map(
                lambda x:np.asarray(x,dtype=float),state.brain_body.body)))
            velocity = None
            for _ in range(steps):
                brain,memory,muscles,_,_ = fast_neural(state,params)
                nextbody,velocity,_ = surface_body_step(state.brain_body.body,
                    signed.astype(float)@np.asarray(muscles,dtype=float),body,law,dt,initial_velocity=velocity)
                state = state._replace(brain_body=state.brain_body._replace(brain=brain,body=nextbody),sensory=memory)
                indicators.append(np.asarray(brain.indicator));angles.append(np.asarray(nextbody.joint_angles))
        sample_time = np.arange(len(indicators))*dt
        def sample(values):
            values=np.asarray(values)
            if sampling=='point':
                return np.column_stack([np.interp(requested,sample_time,values[:,j]) for j in range(values.shape[1])])
            return np.stack([values[(sample_time>=a)&(sample_time<b)].mean(axis=0)
                             for a,b in zip(edges[:-1],edges[1:])])
        dense_q=np.asarray(angles)
        indicator=sample(indicators)
        tangents=np.unwrap(model_source_tangents(dense_q,float(body.mechanics.segment_length)),axis=0)
        bends=source_bends(sample(tangents))
        q=np.column_stack([np.interp(requested,sample_time,dense_q[:,j]) for j in range(11)])
        if not np.isfinite(indicator).all() or not np.isfinite(q).all():
            raise ValueError('Nonfinite rollout')
        return dict(indicator=indicator,joints=q,bends=bends)
    return run
