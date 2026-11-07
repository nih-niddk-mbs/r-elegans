"""Fit all-cell membrane/synapse candidates to identified GCaMP recordings.

Starts from a preserved full-network actor or conductance checkpoint. No target fluorescence is
injected into the neurons. Fits have unknown sensory/neuromodulatory inputs and
unvalidated transferred channel/indicator priors; report generalization honestly.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from r_elegans.brain.conductance_network import (
    ConductanceNetworkConfig,initialize_conductance_network,
    network_recording_intervals,network_recording,save_network,load_network,
)
from r_elegans.data.calcium import load_identified_calcium
from r_elegans.model import load_builtin_model
from r_elegans.rl.full_network_actor import actor_from_arrays
if __package__:
    from .experiment_records import capture_run_record
else:
    from experiment_records import capture_run_record


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--uid',required=True)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--initial-actor',type=Path)
    source.add_argument('--initial-network',type=Path,help='Preserve an existing all-cell channel/kinetics configuration')
    parser.add_argument('--qualify-rhythms',action='store_true',help='Return nonzero if initial/final candidates lack sustained RMD activity in a separate 20 s assay')
    parser.add_argument('--experiment-id',default='E026')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=16)
    parser.add_argument('--iterations',type=int,default=20)
    parser.add_argument('--max-step-ms',type=float,default=1.)
    parser.add_argument('--learning-rate',type=float,default=.01)
    parser.add_argument('--minutes',type=float,default=20.)
    args=parser.parse_args()
    if (args.iterations<1 or not all(math.isfinite(v) and v>0 for v in
            (args.max_step_ms,args.learning_rate,args.minutes))):parser.error('Positive finite fitting settings required')
    output=args.output_dir.expanduser().resolve(); repo=Path(__file__).resolve().parents[1]
    if output==repo or repo in output.parents:parser.error('Keep artifacts external')
    output.mkdir(parents=True,exist_ok=False)
    capture_run_record(output,repo,args)
    parent_path=args.initial_actor if args.initial_actor else args.initial_network
    parent_name='parent_actor.npz' if args.initial_actor else 'parent_network.npz'
    for name,p in [(parent_name,parent_path),('source_recording.h5',args.recording),('source_labels'+''.join(args.labels.suffixes),args.labels)]:shutil.copy2(p,output/name)
    model=load_builtin_model()
    data=load_identified_calcium(args.recording,args.labels,model.neuron_ids,args.uid,frames=args.frames)
    if args.initial_network:
        initial,fixed,config=load_network(args.initial_network)
    else:
        with np.load(args.initial_actor,allow_pickle=False) as z:parent=actor_from_arrays(z)
        config=ConductanceNetworkConfig()
        initial,fixed=initialize_conductance_network(parent,model.neuron_ids,config)
    save_network(output/'network_initial.npz',initial,fixed,config)
    intervals=jnp.asarray(data['intervals_s']); indices=jnp.asarray(data['indices'])
    target=jnp.asarray(data['normalized']); ntrain=data['train_frames']
    substeps=math.ceil(float(intervals.max())*1000./args.max_step_ms)
    predict=jax.jit(lambda p:network_recording_intervals(p,fixed,intervals,substeps,config).indicator[:,indices])
    started=time.monotonic()
    initial_signal=predict(initial)
    initial_signal.block_until_ready()
    # Fixed observation map: only training data/source-model moments establish
    # scale/offset. Never estimate normalization or readout from held-out frames.
    offset=jnp.mean(initial_signal[:ntrain],axis=0)
    gain=1./jnp.maximum(jnp.std(initial_signal[:ntrain],axis=0),.01)
    def normalized_prediction(p):return (predict(p)-offset)*gain
    def loss(p):
        prediction=normalized_prediction(p)
        fit=jnp.mean((prediction[:ntrain]-target[:ntrain])**2)
        prior=jnp.mean(p.log_channel_scale**2)
        for value,start,mask in ((p.raw_chemical,initial.raw_chemical,fixed.chemical_mask),
                                (p.raw_gap,initial.raw_gap,fixed.gap_mask)):
            prior+=jnp.sum(((value-start)*mask)**2)/jnp.maximum(mask.sum(),1.)
        return fit+.001*prior
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(args.learning_rate))
    opt_state=optimizer.init(initial)
    @jax.jit
    def update(p,state):
        value,gradient=jax.value_and_grad(loss)(p)
        updates,state=optimizer.update(gradient,state,p)
        new=optax.apply_updates(p,updates)
        new=new._replace(log_channel_scale=jnp.clip(new.log_channel_scale,-jnp.log(10.),jnp.log(10.)),
            raw_chemical=jnp.where(fixed.chemical_mask,jnp.clip(new.raw_chemical,initial.raw_chemical-2.,initial.raw_chemical+2.),initial.raw_chemical),
            raw_gap=jnp.where(fixed.gap_mask,jnp.clip(new.raw_gap,initial.raw_gap-2.,initial.raw_gap+2.),initial.raw_gap))
        return new,state,value
    initial_pred=np.asarray(normalized_prediction(initial))
    history=[]; p=initial
    for iteration in range(args.iterations):
        if iteration and time.monotonic()-started>args.minutes*60.:break
        p,opt_state,value=update(p,opt_state)
        value=float(value)
        if not math.isfinite(value) or not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(p)):
            raise RuntimeError('Nonfinite fit; initial model and prior checkpoints retained')
        history.append(value)
        print(json.dumps(dict(iteration=iteration+1,pre_update_objective=value,elapsed_seconds=time.monotonic()-started)),flush=True)
    prediction=np.asarray(normalized_prediction(p))
    trajectory=jax.jit(lambda p:network_recording_intervals(p,fixed,intervals,substeps,config))(p)
    refined=jax.jit(lambda p:network_recording_intervals(p,fixed,intervals,substeps*2,config))(p)
    refinement=np.asarray((refined.indicator[:,indices]-trajectory.indicator[:,indices])*gain)
    save_network(output/'network_final.npz',p,fixed,config)
    (output/'optimizer_final.msgpack').write_bytes(serialization.to_bytes(jax.device_get(opt_state)))
    np.savez_compressed(output/'trajectories.npz',time_s=data['time_s'],target=np.asarray(target),
        raw_fluorescence=data['fluorescence'],initial_prediction=initial_pred,prediction=prediction,
        voltage_mV=np.asarray(trajectory.voltage_mV),calcium_uM=np.asarray(trajectory.calcium_uM),
        indicator=np.asarray(trajectory.indicator),readout_gain=np.asarray(gain),readout_offset=np.asarray(offset),
        training_mean=data['training_mean'],training_std=data['training_std'])
    def mse(a,s):return float(np.mean((a[s]-np.asarray(target)[s])**2))
    summary=dict(experiment=args.experiment_id,status='completed',installed=False,uid=args.uid,
        parent_model=str(parent_path.resolve()),parent_sha256=sha(parent_path),
        parent_model_type='conductance-network' if args.initial_network else 'full-network-actor',
        recording_sha256=sha(args.recording),labels_sha256=sha(args.labels),
        initial_sha256=sha(output/'network_initial.npz'),final_sha256=sha(output/'network_final.npz'),
        config=asdict(config),neurons=302,channel_components_per_neuron=17,gates_per_neuron=35,
        observed_neurons=data['names'],observation_indices=data['indices'].tolist(),source_rows=data['rows'],
        confidence=data['confidence'],excluded_labels=data['excluded_labels'],frames=args.frames,
        training_frames=ntrain,heldout_frames=args.frames-ntrain,elapsed_seconds=time.monotonic()-started,
        iterations=len(history),substeps=substeps,maximum_step_ms=float(intervals.max())*1000./substeps,
        initial_train_mse=mse(initial_pred,slice(None,ntrain)),final_train_mse=mse(prediction,slice(None,ntrain)),
        initial_heldout_mse=mse(initial_pred,slice(ntrain,None)),final_heldout_mse=mse(prediction,slice(ntrain,None)),
        training_mean_baseline_heldout_mse=float(np.mean(np.asarray(target)[ntrain:]**2)),
        substep_refinement_prediction_rmse=float(np.sqrt(np.mean(refinement**2))),
        final_objective=float(loss(p)),history=history,
        maximum_log_channel_change=float(np.max(np.abs(p.log_channel_scale))),
        maximum_chemical_parameter_change=float(np.max(np.abs(p.raw_chemical-initial.raw_chemical))),
        maximum_gap_parameter_change=float(np.max(np.abs(p.raw_gap-initial.raw_gap))),
        voltage_range_mV=[float(trajectory.voltage_mV.min()),float(trajectory.voltage_mV.max())],
        limitations=['Short-window candidate, not unique physiological parameters or validated behavior',
            'Transferred channel priors for most neurons; normalized-to-physical migration includes bounds',
            'No known sensory drive, proprioception, neuromodulation or latent input fitted',
            'GCaMP7f observation uses uncalibrated effective Hill/low-pass kinetics, not nuclear transport',
            'No target voltage, fluorescence or behavioral teacher forcing',
            'Neuron identity coverage differs from complete 302-cell simulation'],
        model_evidence='Calcium-constrained candidate; no unique channel/synapse identification')
    summary['train_improved']=summary['final_train_mse']<summary['initial_train_mse']
    summary['heldout_improved']=summary['final_heldout_mse']<summary['initial_heldout_mse']
    summary['finite']=all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(trajectory))
    summary['fit_and_refinement_seconds']=summary['elapsed_seconds']
    if args.qualify_rhythms:
        from r_elegans.brain.rhythm_qualification import rhythm_metrics
        assay_substeps=math.ceil(2./args.max_step_ms)
        def assay(q):return network_recording(q,fixed,.002,10000,assay_substeps,config)
        def assess(q):
            s=jax.jit(assay)(q)
            results={n:rhythm_metrics(np.asarray(s.voltage_mV)[:,i],.002)
                     for i,n in enumerate(model.neuron_ids) if n.startswith('RMD')}
            return dict(passed=any(r['passed'] for r in results.values()),cells=results)
        summary['rhythm_qualification']=dict(initial=assess(initial),final=assess(p),
            duration_s=20.,sample_dt_s=.002,substeps=assay_substeps,
            scope='Exploratory persistence gate, not animal phenotype or coordinated gait')
    summary['elapsed_seconds']=time.monotonic()-started
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('history','observed_neurons','confidence','source_rows','observation_indices')}),flush=True)
    qualified=not args.qualify_rhythms or (summary['rhythm_qualification']['initial']['passed']
        and summary['rhythm_qualification']['final']['passed'])
    return 0 if summary['finite'] and qualified else 2


if __name__=='__main__':raise SystemExit(main())
