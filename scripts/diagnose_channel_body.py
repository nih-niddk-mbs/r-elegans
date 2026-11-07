"""Fixed-parameter, unpaced channel-brain → 95 muscles → body diagnostic.

Streaming chunks preserve all neuron gates/calcium and body state. No RL,
optimization, gait teacher, periodic forcing or body-to-brain feedback.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import time

import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.brain.conductance_network import load_network,save_network
from r_elegans.brain.rhythm_qualification import rhythm_metrics
from r_elegans.body import default_muscle_body_params,physical_muscle_body_params
from r_elegans.body.channel_bridge import channel_body_chunk,initial_channel_body_state
from r_elegans.model import load_builtin_model
from r_elegans.rl.full_network_actor import actor_from_arrays
if __package__:
    from .experiment_records import capture_run_record
else:
    from experiment_records import capture_run_record


def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def save_state(path,state):
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f,**{'brain_'+k:np.asarray(v) for k,v in state.brain._asdict().items()},
            **{'body_'+k:np.asarray(v) for k,v in state.body._asdict().items()})
    temp.replace(path)


def diagnostic_metrics(voltage,muscles,body,dt,body_length,neuron_ids):
    """Late-half measures; motion is separate from locomotion or animal fidelity."""
    start=len(voltage)//2
    v,m,q=voltage[start:],muscles[start:],body['joint_angles'][start:]
    position=body['position'][start:]
    displacement=float(np.linalg.norm(position[-1]-position[0])/body_length)
    path=float(np.linalg.norm(np.diff(position,axis=0),axis=1).sum()/body_length)
    v_p2p=np.ptp(v,axis=0);m_p2p=np.ptp(m,axis=0);q_p2p=np.ptp(q,axis=0)
    rmd={n:rhythm_metrics(voltage[:,i],dt) for i,n in enumerate(neuron_ids) if n.startswith('RMD')}
    return dict(observation_seconds=len(voltage)*dt,late_seconds=len(v)*dt,
        neural_voltage_p2p_max_mV=float(v_p2p.max()),rhythmic_rmd=sum(x['passed'] for x in rmd.values()),rmd=rmd,
        muscle_p2p_max=float(m_p2p.max()),muscle_p2p_rms=float(np.sqrt(np.mean(m_p2p**2))),
        muscle_activation_min=float(m.min()),muscle_activation_max=float(m.max()),
        muscle_saturation_fraction=float(np.mean((m<.01)|(m>.99))),
        joint_p2p_max_rad=float(q_p2p.max()),joint_p2p_rms_rad=float(np.sqrt(np.mean(q_p2p**2))),
        late_displacement_body_lengths=displacement,late_path_body_lengths=path,
        mean_late_speed_body_lengths_s=path/max((len(position)-1)*dt,dt),
        heading_change_rad=float(body['heading'][-1]-body['heading'][start]),
        muscle_dynamic=bool(m_p2p.max()>1e-4),shape_dynamic=bool(q_p2p.max()>1e-4),
        measurable_displacement=bool(displacement>1e-5),
        locomotion_demonstrated=False,
        interpretation='Dynamic shape/drift flags are not coordinated gait or physiological validation')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network',type=Path,required=True)
    parser.add_argument('--parent-actor',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--medium',choices=['normalized','liquid','agar'],default='liquid')
    parser.add_argument('--condition',choices=['active','cca1_removed','held_voltage'],default='active')
    parser.add_argument('--perturbation-mv',type=float,default=0.)
    parser.add_argument('--duration',type=float,default=300.)
    parser.add_argument('--minutes',type=float,default=12.)
    parser.add_argument('--dt',type=float,default=.002)
    parser.add_argument('--max-brain-step-ms',type=float,default=.1)
    parser.add_argument('--chunk-seconds',type=float,default=2.)
    args=parser.parse_args()
    if not all(math.isfinite(v) for v in (args.perturbation_mv,args.duration,args.minutes,args.dt,args.max_brain_step_ms,args.chunk_seconds)):
        parser.error('Finite diagnostic settings required')
    if min(args.duration,args.minutes,args.dt,args.max_brain_step_ms,args.chunk_seconds)<=0:
        parser.error('Positive times required')
    output=args.output_dir.expanduser().resolve();repo=Path(__file__).resolve().parents[1]
    if output==repo or repo in output.parents:parser.error('Keep scientific artifacts external')
    output.mkdir(parents=True,exist_ok=False);capture_run_record(output,repo,args)
    for path,name in [(args.network,'source_network.npz'),(args.parent_actor,'source_parent_actor.npz')]:shutil.copy2(path,output/name)
    params,fixed,config=load_network(args.network)
    model=load_builtin_model()
    with np.load(args.parent_actor,allow_pickle=False) as z:parent=actor_from_arrays(z)
    if args.condition=='cca1_removed':
        indices=jnp.asarray([i for i,n in enumerate(model.neuron_ids) if n.startswith('RMD')])
        fixed=fixed._replace(cells=fixed.cells._replace(conductances_nS=fixed.cells.conductances_nS.at[indices,8].set(0.)))
    save_network(output/'evaluated_network.npz',params,fixed,config)
    body_params=(default_muscle_body_params(12) if args.medium=='normalized'
                 else physical_muscle_body_params(12,medium=args.medium))
    body_length=float(body_params.mechanics.segment_length)*12.
    write_json(output/'body_parameters.json',dict(medium=args.medium,body_length=body_length,
        units='normalized' if args.medium=='normalized' else 'SI',
        parameters={k:({a:float(b) for a,b in v._asdict().items()} if k=='mechanics' else float(v)) for k,v in body_params._asdict().items()},
        projection='head_at_end=True',nmj_edges=int(np.count_nonzero(model.neuromuscular_params.synapse_weights)),
        readout='Channel-network physical release fields, parent NMJ signs/gains and nonzero-tone muscle sigmoid'))
    state=initial_channel_body_state(fixed,config)
    if args.perturbation_mv:
        # A deterministic, nonperiodic initial-state perturbation, not a drive.
        delta=args.perturbation_mv*jnp.sin(jnp.arange(302)*1.618)
        state=state._replace(brain=state.brain._replace(voltage_mV=state.brain.voltage_mV+delta))
    save_state(output/'state_initial.npz',state)
    steps=max(1,round(args.chunk_seconds/args.dt))
    substeps=math.ceil(args.dt*1000./args.max_brain_step_ms)
    advance=jax.jit(lambda s:channel_body_chunk(s,params,fixed,config,parent,model,body_params,
        dt=args.dt,steps=steps,brain_substeps=substeps,hold_voltage=args.condition=='held_voltage'))
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    start=time.monotonic();chunks=[];simulation_time=0.;reason='duration';finite=True
    # Retain at most the final 30 simulated seconds in the in-memory metrics window;
    # all chunk traces/checkpoints are separately persisted for long-run analysis.
    tail=[]
    while simulation_time+1e-9<args.duration:
        if stopped:reason='signal';break
        if time.monotonic()-start>=args.minutes*60.:reason='time_budget';break
        state,trace=advance(state)
        arrays=jax.device_get(trace)
        finite=all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves((state,arrays)))
        if not finite:reason='nonfinite';break
        index=len(chunks);file=output/f'chunk-{index:05d}.npz'
        body={k:np.asarray(v) for k,v in arrays[3]._asdict().items()}
        np.savez_compressed(file,voltage_mV=arrays[0],calcium_uM=arrays[1],muscles=arrays[2],
            **{'body_'+k:v for k,v in body.items()})
        chunks.append(file.name);tail.append((np.asarray(arrays[0]),np.asarray(arrays[2]),body))
        while len(tail)>max(1,math.ceil(30./(steps*args.dt))):tail.pop(0)
        simulation_time+=steps*args.dt
        save_state(output/'state_latest.npz',state)
        write_json(output/'progress.json',dict(completed_chunks=len(chunks),simulation_seconds=simulation_time,
            elapsed_seconds=time.monotonic()-start,finite=True))
        print(json.dumps(dict(chunk=index,simulation_seconds=simulation_time,elapsed_seconds=time.monotonic()-start)),flush=True)
    metrics=None
    if tail:
        v=np.concatenate([t[0] for t in tail]);m=np.concatenate([t[1] for t in tail])
        body={k:np.concatenate([t[2][k] for t in tail]) for k in tail[0][2]}
        metrics=diagnostic_metrics(v,m,body,args.dt,body_length,model.neuron_ids)
        metrics['joint_limit_fraction']=float(np.mean(np.abs(body['joint_angles'][len(v)//2:])>=.999*float(body_params.max_joint_angle)))
    summary=dict(experiment='E028',condition=args.condition,medium=args.medium,perturbation_mV=args.perturbation_mv,
        status='completed' if finite and tail else 'failed',stop_reason=reason,finite=finite,
        simulation_seconds=simulation_time,requested_duration_seconds=args.duration,elapsed_seconds=time.monotonic()-start,
        dt_s=args.dt,brain_substeps=substeps,brain_step_ms=args.dt*1000./substeps,
        network_sha256=hashlib.sha256(args.network.read_bytes()).hexdigest(),
        parent_sha256=hashlib.sha256(args.parent_actor.read_bytes()).hexdigest(),
        metrics=metrics,chunks=chunks,parameters_trained=False,body_feedback=False,
        gates_are_diagnostics='No successful locomotion claim from transient drift or nonzero displacement alone')
    write_json(output/'summary.json',summary)
    if finite:save_state(output/'state_final.npz',state)
    print(json.dumps(summary),flush=True)
    return 0 if finite and tail else 2


if __name__=='__main__':raise SystemExit(main())
