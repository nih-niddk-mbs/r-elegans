"""Fit intrinsic densities for autonomous amplitude or repeated forward motion."""
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

from r_elegans.brain.autonomous_wiggle import (
    autonomous_fit_reference,apply_density_offsets,density_proposal,
    amplitude_metrics,frozen_fit_fields_unchanged,movement_metrics,movement_achieved,movement_refinement_agrees,
)
from r_elegans.brain.conductance_network import load_network,save_network
from r_elegans.brain.single_compartment import CHANNEL_NAMES
from r_elegans.body import physical_muscle_body_params
from r_elegans.body.channel_bridge import initial_channel_body_state,channel_body_chunk
from r_elegans.model import load_builtin_model
from r_elegans.rl.full_network_actor import actor_from_arrays
from scripts.experiment_records import capture_run_record
from scripts.run_embodied_diagnostics import select_input_bundle,verify_input_bundle
from scripts.diagnose_channel_body import write_json,save_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--objective',choices=['amplitude','movement'],default='amplitude')
    parser.add_argument('--target-speed',type=float,default=1e-4,help='Movement target in body lengths/s')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--minutes',type=float,default=60.)
    parser.add_argument('--iterations',type=int,default=10000)
    parser.add_argument('--fit-seconds',type=float,help='Default 12 s for amplitude, 30 s for movement')
    parser.add_argument('--validation-seconds',type=float,default=300.)
    parser.add_argument('--target-p2p-rad',type=float,default=.01)
    parser.add_argument('--maximum-factor',type=float,default=4.)
    args=parser.parse_args()
    if args.fit_seconds is None:args.fit_seconds=30. if args.objective=='movement' else 12.
    if not all(math.isfinite(x) and x>0 for x in (args.minutes,args.fit_seconds,args.validation_seconds,args.target_p2p_rad,args.target_speed)):
        parser.error('Positive finite settings required')
    if args.fit_seconds<4 or args.validation_seconds<args.fit_seconds or args.iterations<0:
        parser.error('Fit >=4 s, validation >= fit, iterations >=0 required')
    if not math.isfinite(args.maximum_factor) or args.maximum_factor<=1:parser.error('Factor must exceed one')
    repo=Path(__file__).resolve().parents[1];out=args.output_dir.expanduser().resolve()
    if out==repo or repo in out.parents:parser.error('Keep results external')
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,repo,args)
    inputs=select_input_bundle(args.inputs)
    if args.objective=='movement' and args.inputs is None:inputs=inputs.parent/'autonomous_wiggle_v1'
    manifest=verify_input_bundle(inputs)
    shutil.copytree(inputs,out/'source_inputs')
    initial,fixed,config=load_network(out/'source_inputs/network.npz')
    model=load_builtin_model()
    with np.load(out/'source_inputs/parent_actor.npz',allow_pickle=False) as z:parent=actor_from_arrays(z)
    fixed,mask=autonomous_fit_reference(initial,fixed,model.neuron_ids)
    assert np.all(np.asarray(fixed.bias_pA)==0)
    save_network(out/'network_initial.npz',initial,fixed,config)
    np.savez_compressed(out/'allowed_parameters.npz',channel_mask=mask)
    indices,channels=np.nonzero(mask)
    experiment='E031' if args.objective=='movement' else 'E030'
    policy=dict(experiment=experiment,objective=args.objective,parameters=[dict(neuron=model.neuron_ids[i],channel=CHANNEL_NAMES[c]) for i,c in zip(indices,channels)],
        maximum_density_factor=args.maximum_factor,
        fixed=['all nonselected densities','LEAK','capacitance','kinetics','reversals','calcium handling',
               'chemical and gap strengths','release parameters','NMJs','muscle/body parameters'],
        injected_currents='All compensation/bias currents zero throughout; no sensory or periodic input',
        scope='302-neuron network simulated; only uncertain existing non-LEAK RMD densities fit',
        method='Bounded stochastic hill climb; shared-channel and independent-cell proposals; no gradient, gait teacher or RL',
        physiological_status='Exploratory RMD densities explicitly trainable; not measured stereotypical constants',
        target_joint_p2p_rad=args.target_p2p_rad,target_speed_body_lengths_s=args.target_speed,
        movement_gate='>=0.01-rad wiggle, 3 positive progress blocks, target speed and 5x positive baseline, 2% efficiency, <0.35-rad turn',
        source_manifest=manifest)
    write_json(out/'fit_policy.json',policy)
    body=physical_muscle_body_params(12,medium='liquid');dt=.002
    start=time.monotonic();stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    # Reserve a substantial portion of the budget for independent validation.
    reserve=min(args.minutes*60.*.5,args.validation_seconds*1.6+args.fit_seconds*6+30.)
    search_deadline=start+args.minutes*60.-reserve
    rng=np.random.default_rng(args.seed);best=np.zeros(int(mask.sum()))
    initial_state=initial_channel_body_state(fixed,config)
    kernels={}
    def kernel(seconds,step,body_dt,hold_voltage):
        key=(seconds,step,body_dt,hold_voltage)
        if key not in kernels:
            kernels[key]=jax.jit(lambda s,p:channel_body_chunk(s,p,fixed,config,parent,model,body,
                dt=body_dt,steps=round(seconds/body_dt),brain_substeps=math.ceil(body_dt*1000./step),hold_voltage=hold_voltage))
        return kernels[key]
    def evaluate(offsets,seconds,*,step=.1,perturbation=0.,body_dt=.002,hold_voltage=False):
        params=apply_density_offsets(initial,jnp.asarray(offsets),mask)
        state=initial_state
        if perturbation:
            state=state._replace(brain=state.brain._replace(voltage_mV=state.brain.voltage_mV+
                perturbation*jnp.sin(jnp.arange(302)*1.618)))
        final,trace=jax.device_get(kernel(seconds,step,body_dt,hold_voltage)(state,params))
        if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves((final,trace))):
            metrics=dict(valid=False,score=-1.,reason='nonfinite')
        else:
            if args.objective=='movement':
                metrics=movement_metrics(trace[0],trace[2],trace[3],body_dt,float(body.mechanics.segment_length)*12.,model.neuron_ids,float(body.max_joint_angle))
            else:metrics=amplitude_metrics(trace[0],trace[2],trace[3].joint_angles,body_dt,model.neuron_ids,float(body.max_joint_angle))
        return params,final,trace,metrics
    def save_trace(path,trace):
        np.savez_compressed(path,voltage_mV=trace[0],calcium_uM=trace[1],muscles=trace[2],
            **{'body_'+k:np.asarray(v) for k,v in trace[3]._asdict().items()})
    selected=out/'selected';selected.mkdir()
    def save_selected(index,offsets,params,state,trace,metrics):
        save_network(selected/f'network-{index:05d}.npz',params,fixed,config)
        save_trace(selected/f'trace-{index:05d}.npz',trace)
        save_state(selected/f'state-{index:05d}.npz',state)
        save_network(out/'network_best.npz',params,fixed,config)
        write_json(out/'best.json',dict(iteration=index,offsets=offsets.tolist(),metrics=metrics))
    params,state,trace,baseline=evaluate(best,args.fit_seconds)
    write_json(out/'baseline.json',baseline);save_selected(0,best,params,state,trace,baseline)
    best_metrics=baseline;selected_iteration=0;completed=0
    history=(out/'history.jsonl').open('w')
    try:
        for iteration in range(1,args.iterations+1):
            if stopped or time.monotonic()>=search_deadline:break
            # This first stage seeks a small amplitude milestone, not maximum bend.
            if args.objective=='amplitude' and best_metrics['valid'] and best_metrics['sustained_joint_p2p_rad']>=args.target_p2p_rad:break
            sigma=.12*2**((iteration//20)%3)
            candidate=density_proposal(best,mask,rng,sigma,args.maximum_factor)
            params,state,trace,metrics=evaluate(candidate,args.fit_seconds)
            accepted=metrics['score']>best_metrics['score']
            if accepted:
                best=candidate;best_metrics=metrics;selected_iteration=iteration
                save_selected(iteration,best,params,state,trace,metrics)
            completed=iteration
            history.write(json.dumps(dict(iteration=iteration,offsets=candidate.tolist(),accepted=accepted,
                metrics=metrics,elapsed_seconds=time.monotonic()-start),allow_nan=False)+'\n');history.flush()
            write_json(out/'search_state.json',dict(completed_iterations=iteration,best=best.tolist(),
                best_metrics=best_metrics,rng_state=rng.bit_generator.state,selected_iteration=selected_iteration,
                continuation='Replay metadata retained; CLI starts a fresh search, not exact mid-run resume'))
            print(json.dumps(dict(iteration=iteration,accepted=accepted,best_amplitude=best_metrics.get('sustained_joint_p2p_rad',0.),
                rhythmic_rmd=best_metrics.get('rhythmic_rmd',0),
                best_forward_speed=best_metrics.get('minimum_forward_speed_body_lengths_s'),
                elapsed_minutes=(time.monotonic()-start)/60.)),flush=True)
    finally:history.close()
    validation=[]
    cases=[('long',args.validation_seconds,.1,0.,dt,False),('perturb-plus',args.fit_seconds,.1,1.,dt,False),
           ('perturb-minus',args.fit_seconds,.1,-1.,dt,False),('brain-refined',args.fit_seconds,.05,0.,dt,False)]
    if args.objective=='movement':cases.extend([
        ('body-refined',args.fit_seconds,.1,0.,.001,False),('held-voltage',args.fit_seconds,.1,0.,dt,True)])
    for name,seconds,step,perturbation,body_dt,hold_voltage in cases:
        if stopped or time.monotonic()>=start+args.minutes*60.:break
        params,state,trace,metrics=evaluate(best,seconds,step=step,perturbation=perturbation,body_dt=body_dt,hold_voltage=hold_voltage)
        save_trace(out/f'validation-{name}.npz',trace);save_state(out/f'validation-state-{name}.npz',state)
        validation.append(dict(case=name,seconds=seconds,brain_step_ms=step,dt_s=body_dt,perturbation_mV=perturbation,metrics=metrics))
        print(json.dumps(dict(validation=name,metrics=metrics),allow_nan=False),flush=True)
    best_params=apply_density_offsets(initial,jnp.asarray(best),mask)
    frozen_ok=frozen_fit_fields_unchanged(best_params,initial,mask)
    if args.objective=='movement':
        baseline_speed=baseline.get('minimum_forward_speed_body_lengths_s',0.)
        def qualified(v):
            m=v['metrics']
            if v['case']=='held-voltage':return (m.get('rhythmic_rmd')==0 and m.get('joint_p2p_max_rad',1.)<1e-5
                and abs(m.get('minimum_forward_speed_body_lengths_s',1.))<args.target_speed*.01)
            achieved=movement_achieved(m,args.target_speed,baseline_speed)
            if v['case'] in ('body-refined','brain-refined'):
                return achieved and movement_refinement_agrees(best_metrics,m,args.target_speed)
            return achieved
    else:
        def qualified(v):return v['metrics']['valid'] and v['metrics']['sustained_joint_p2p_rad']>=args.target_p2p_rad
    passed=not stopped and frozen_ok and len(validation)==len(cases) and all(qualified(v) for v in validation)
    # Standardized pair for subsequent assays/continuations, never overwriting the source.
    fitted=out/'fitted_inputs';fitted.mkdir()
    shutil.copyfile(out/'network_best.npz',fitted/'network.npz')
    shutil.copyfile(out/'source_inputs/parent_actor.npz',fitted/'parent_actor.npz')
    write_json(fitted/'inputs.json',dict(experiment=experiment,objective=args.objective,passed=bool(passed),
        sha256={n:hashlib.sha256((fitted/n).read_bytes()).hexdigest() for n in ('network.npz','parent_actor.npz')}))
    summary=dict(experiment=experiment,objective=args.objective,seed=args.seed,status='completed' if not stopped else 'interrupted',passed=bool(passed),
        baseline=baseline,best_training=best_metrics,validation=validation,iterations=completed,selected_iteration=selected_iteration,
        target_p2p_rad=args.target_p2p_rad,target_speed_body_lengths_s=args.target_speed,elapsed_seconds=time.monotonic()-start,
        frozen_fields_unchanged=bool(frozen_ok),injected_currents_zero=bool(np.all(np.asarray(fixed.bias_pA)==0)),
        network_best_sha256=hashlib.sha256((out/'network_best.npz').read_bytes()).hexdigest(),
        interpretation='First autonomous motion criterion, not animal-matched behavior' if args.objective=='movement' else 'Autonomous bend amplitude only; no useful locomotion or biological validity claim')
    write_json(out/'summary.json',summary)
    print(json.dumps(summary,allow_nan=False),flush=True)
    return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
