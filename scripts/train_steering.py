"""Fit one signed shallow-steering skill, then test actual curved locomotion.

Full recurrent brain; fixed anatomical NMJs. No shared-skill retention loss.
"""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import signal
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from r_elegans.body import initialize_muscle_body,default_muscle_body_params
from r_elegans.model import load_builtin_model
from r_elegans.rl.full_network_actor import init_full_network_actor,actor_from_arrays,make_full_network_actor_interface
from r_elegans.rl.motor_curriculum import commanded_muscle_sequence,motor_stimulation_vector,muscle_sequence_body,traveling_metrics,traveling_passes
from r_elegans.rl.steering import steering_target,steering_metrics,turn_checks,assess_steering,selection_key,trajectory_objective
if __package__:
    from .experiment_records import capture_run_record,file_sha256
    from .train_timed import save_actor,save_json,checkpoint,finite_tree
    from .train_motor_alternation import command_sequences
else:
    from experiment_records import capture_run_record,file_sha256
    from train_timed import save_actor,save_json,checkpoint,finite_tree
    from train_motor_alternation import command_sequences


def main(skill='steering'):
    head_mode=skill=='head-sweep'
    if skill not in ('steering','head-sweep'):raise ValueError('Unknown standalone skill')
    if head_mode:
        from r_elegans.rl.head_sweep import (head_target,head_stimulation_vector,head_metrics,
            head_checks,assess_head_sweep,head_imitation_metrics,head_selection_key)
    target_function=head_target if head_mode else steering_target
    metric_function=head_metrics if head_mode else steering_metrics
    imitation_function=head_imitation_metrics if head_mode else traveling_metrics
    selection_function=head_selection_key if head_mode else selection_key
    assessment_function=assess_head_sweep if head_mode else assess_steering
    parser=argparse.ArgumentParser(description='Fit a standalone stationary head-sweep policy through all 302 neurons.' if head_mode else __doc__)
    parser.add_argument('--initial-actor',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    if head_mode:parser.set_defaults(bias=0.)
    else:parser.add_argument('--bias',type=float,default=.006)
    parser.add_argument('--amplitude',type=float,default=.04 if head_mode else .07)
    parser.add_argument('--minutes',type=float,default=480.)
    parser.add_argument('--iterations',type=int,default=1000000)
    parser.add_argument('--learning-rate',type=float,default=.0003)
    parser.add_argument('--body-loss-weight',type=float,default=.5)
    parser.add_argument('--validation-every',type=int,default=250)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--oracle-only',action='store_true')
    args=parser.parse_args()
    if (not all(math.isfinite(v) and v>0 for v in (args.minutes,args.learning_rate,args.amplitude))
            or (not head_mode and (not math.isfinite(args.bias) or not 0<abs(args.bias)<=.012))
            or not math.isfinite(args.body_loss_weight) or args.body_loss_weight<0
            or args.iterations<1 or args.validation_every<1):
        parser.error('Require positive finite budgets/rate/amplitude and signed bias magnitude in (0,.012]')
    source=json.loads((args.initial_actor.parent/'manifest.json').read_text())
    if (source.get('input_mode')!='current' or source.get('body_units')!='legacy_normalized'
            or source.get('muscle_projection')!='anterior_to_head_at_last_segment'):
        parser.error('Requires a head-aligned normalized current-input forward checkpoint')
    if not 0<args.amplitude<=(.08 if head_mode else min(.1,1.3*source['amplitude']+1e-8)):
        parser.error('Target amplitude must be <=.1 and at most 30% above source amplitude')
    gain,delay=source['stimulation_gain'],source['propagation_delay']
    if not all(math.isfinite(v) and v>0 for v in (gain,delay)) or not .2<=delay<=.8:
        parser.error('Invalid source pacing gain or propagation delay')
    repo=Path(__file__).resolve().parents[1];output=args.output_dir.expanduser().resolve()
    if output==repo or repo in output.parents:
        parser.error('Artifacts must stay outside Git')
    model=load_builtin_model();expected=init_full_network_actor(model,jax.random.PRNGKey(0))
    with np.load(args.initial_actor,allow_pickle=False) as data:actor=actor_from_arrays(data)
    for name in ('chemical_mask','gap_mask'):
        if not np.array_equal(getattr(actor.neural,name),getattr(expected.neural,name)):
            parser.error('Checkpoint topology mismatch')
    for name in ('raw_nmj','unknown_polarity','muscle_threshold','raw_muscle_slope'):
        if not np.array_equal(getattr(actor,name),getattr(expected,name)):
            parser.error('Requires original fixed NMJ/muscle mapping')
    output.mkdir(parents=True,exist_ok=False);record=capture_run_record(output,repo,args)
    started=time.perf_counter();deadline=started+args.minutes*60;stop=False
    def request_stop(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    sign=1 if args.bias>0 else -1
    stimulation=head_stimulation_vector(model,gain) if head_mode else motor_stimulation_vector(model,gain)
    interface=make_full_network_actor_interface(model,train_group='neural',input_mode='current')
    def target(c,dt=.02):
        return target_function(c,amplitude=args.amplitude,bias=args.bias,delay=delay,dt=dt)
    @jax.jit
    def predict(a,c):return commanded_muscle_sequence(a,interface,c,stimulation)
    @jax.jit
    def probe_body(a,c):return muscle_sequence_body(predict(a,c))
    def states(body):
        if not finite_tree(body):raise RuntimeError('Nonfinite body trajectory')
        return jax.tree_util.tree_map(lambda a,b:np.concatenate([np.asarray(a)[None],np.asarray(b)]),initialize_muscle_body(12),body)
    def metrics(body,n):
        s=states(body)
        return metric_function(s.position,s.heading,s.joint_angles,period_steps=n)
    # The oracle is checked before neural optimization; it does not enter the brain.
    oracle={}
    for period in (2.4,3.,3.7):
        n=round(period/.01);c=.85*jnp.sin(2*jnp.pi*jnp.arange(n*10)/n)
        body=jax.jit(lambda c:muscle_sequence_body(target(c,.01),dt=.01))(c)
        oracle[str(period)]=metrics(body,n)
        s=states(body)
        np.savez_compressed(output/f'period-{period}-oracle.npz',commands=np.asarray(c),time=s.time,
            position=s.position,heading=s.heading,joint_angles=s.joint_angles,muscle_activation=s.muscle_activation)
    oracle_valid=(all(all(head_checks(m,m).values()) for m in oracle.values()) if head_mode else
        all(m['mean_forward_body_lengths']>=.001 and all(turn_checks(m,m,sign=sign)[k]
            for k in ('axis_turn','path_turn','consistent_turn','axis_path_agreement')) for m in oracle.values()))
    stage_name='stationary_head_sweep' if head_mode else 'standalone_shallow_steering'
    manifest=dict(stage=stage_name+'_oracle_calibration' if args.oracle_only else stage_name,skill_id='MOT-09' if head_mode else 'MOT-08',seed=args.seed,actor_neurons=302,
        initial_actor=str(args.initial_actor.resolve()),initial_actor_sha256=file_sha256(args.initial_actor),
        amplitude=args.amplitude,steering_bias=args.bias,expected_turn_sign=sign,
        input_mode='current',stimulation_gain=gain,propagation_delay=delay,body_units='legacy_normalized',
        muscle_projection='anterior_to_head_at_last_segment',train_group='neural',fixed_nmj=True,
        shared_skill_retention_required=False,learning_rate=args.learning_rate,minutes=args.minutes,
        body_loss_weight=args.body_loss_weight,objective='normalized_muscle_error_plus_weight_times_mean_joint_and_centroid_trajectory_error',
        iterations=args.iterations,training_steps=192,tuning_steps=384,validation_every=args.validation_every,
        checkpoint_every=100,selection='imitation_and_relative_head_localization_probe' if head_mode else 'imitation_and_actual_centroid_turn_probe',
        source_record=str(record.relative_to(output)),timestamp_utc=datetime.now(timezone.utc).isoformat(),
        oracle_valid=oracle_valid,oracle=oracle,minimum_progress=.001,minimum_turn_radians_per_cycle=5e-5,
        minimum_oracle_turn_fraction=.5,backend=jax.default_backend(),jax_version=jax.__version__)
    save_json(output/'manifest.json',manifest)
    if head_mode:
        for key in ('steering_bias','expected_turn_sign','minimum_turn_radians_per_cycle','minimum_oracle_turn_fraction'):
            manifest.pop(key)
        manifest.update(stimulation_scheme='anterior_NMJ_weighted_currents_v1',anterior_profile_width=.18,
            target_propagation_delay=None,inherited_forward_delay_unused=True,minimum_oracle_amplitude_fraction=.5,
            minimum_progress=None,maximum_cycle_displacement=.0001,maximum_trunk_drift=.002,
            minimum_head_angle_p2p=.02,minimum_head_lateral_p2p=.0005,maximum_posterior_anterior_ratio=.25)
        save_json(output/'manifest.json',manifest)
    if args.oracle_only or not oracle_valid:
        save_json(output/'summary.json',dict(stage=manifest['stage'],status='completed',error=None,
            passed=oracle_valid if args.oracle_only else False,oracle_only=args.oracle_only,
            oracle_valid=oracle_valid,iterations=0,oracle=oracle))
        return 0 if args.oracle_only and oracle_valid else 2
    train=command_sequences(192);targets=jax.vmap(target)(train)
    target_bodies=jax.jit(jax.vmap(muscle_sequence_body))(targets)
    tuning=command_sequences(384,held_out=True);tuning_targets=jax.vmap(target)(tuning)
    np.savez_compressed(output/'supervision.npz',training_commands=np.asarray(train),training_targets=np.asarray(targets),
        tuning_commands=np.asarray(tuning),tuning_targets=np.asarray(tuning_targets),stimulation=np.asarray(stimulation))
    @jax.jit
    def evaluate(a):return jax.vmap(imitation_function)(jax.vmap(lambda c:predict(a,c))(tuning),tuning_targets)
    probe_commands=.85*jnp.sin(2*jnp.pi*jnp.arange(1200)/120)
    probe_oracle=metrics(jax.jit(lambda c:muscle_sequence_body(target(c)))(probe_commands),120)
    def validate(a):
        m=evaluate(a)
        if not finite_tree(m):raise RuntimeError('Nonfinite tuning metrics')
        imitation=[{k:float(v[i]) for k,v in m.items()} for i in range(len(tuning))]
        probe=metrics(probe_body(a,probe_commands),120)
        return dict(imitation=imitation,probe=probe,key=selection_function(imitation,probe,probe_oracle,sign=sign))
    def loss(a):return trajectory_objective(jax.vmap(lambda c:predict(a,c))(train),targets,target_bodies,body_weight=args.body_loss_weight)
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(args.learning_rate));opt_state=optimizer.init(actor)
    @jax.jit
    def update(a,state):
        value,g=jax.value_and_grad(loss)(a);delta,state=optimizer.update(g,state,a)
        return optax.apply_updates(a,delta),state,value
    initial=validate(actor);best=actor;best_report=initial;selected=iteration=0;status='completed';error=None
    save_actor(output/'actor_initial.npz',actor);save_actor(output/'actor_best.npz',best)
    def consider(history):
        nonlocal best,best_report,selected
        report=validate(actor);history.write(json.dumps(dict(iteration=iteration,**report))+'\n')
        if tuple(report['key'])<tuple(best_report['key']):
            best,best_report,selected=actor,report,iteration;save_actor(output/'actor_best.npz',best)
        print(f"validation iteration={iteration} key={report['key']} selected={selected}",flush=True)
    try:
        with (output/'loss.jsonl').open('w',buffering=1) as losses,(output/'validation.jsonl').open('w',buffering=1) as history:
            history.write(json.dumps(dict(iteration=0,**initial))+'\n')
            while iteration<args.iterations and time.perf_counter()<deadline-30 and not stop:
                candidate,state,value=jax.block_until_ready(update(actor,opt_state))
                if not finite_tree((candidate,state,value)):raise RuntimeError('Nonfinite update; retaining prior finite state')
                actor,opt_state=candidate,state;iteration+=1
                losses.write(json.dumps(dict(iteration=iteration,objective=float(value),elapsed_seconds=time.perf_counter()-started))+'\n')
                if iteration==1 or iteration%100==0:
                    checkpoint(output/'latest.msgpack',actor,opt_state,jax.random.PRNGKey(args.seed),iteration)
                    save_actor(output/'actor_latest.npz',actor)
                    print(f'iteration={iteration} objective={float(value):.7f} elapsed_min={(time.perf_counter()-started)/60:.1f}',flush=True)
                if iteration%args.validation_every==0:consider(history)
            if iteration%args.validation_every:consider(history)
        if stop:status='interrupted'
        if iteration==0:raise RuntimeError('Budget expired before first update')
    except Exception as exception:status,error='failed',repr(exception)
    finally:
        checkpoint(output/'latest.msgpack',actor,opt_state,jax.random.PRNGKey(args.seed),iteration)
        save_actor(output/'actor_final.npz',actor);save_actor(output/'actor_best.npz',best)
    decisions=[]
    for period in (2.4,3.,3.7):
        reports={}
        for dt,rest,isotropic in ((.02,False,False),(.01,False,False),(.01,True,False),(.01,False,True)):
            n=round(period/dt);c=jnp.zeros(n*10) if rest else .85*jnp.sin(2*jnp.pi*jnp.arange(n*10)/n)
            current=make_full_network_actor_interface(model,dt=dt,input_mode='current')
            params=default_muscle_body_params(12)
            if isotropic:params=params._replace(mechanics=params.mechanics._replace(perpendicular_drag=params.mechanics.parallel_drag))
            body=jax.jit(lambda a:muscle_sequence_body(commanded_muscle_sequence(a,current,c,stimulation),dt=dt,body_params=params))(best)
            label='isotropic' if isotropic else 'rest' if rest else 'fine' if dt==.01 else 'coarse'
            reports[label]=metrics(body,n);s=states(body)
            np.savez_compressed(output/f'period-{period}-{label}.npz',time=s.time,commands=np.asarray(c),
                position=s.position,heading=s.heading,joint_angles=s.joint_angles,muscle_activation=s.muscle_activation)
        result=dict(period_seconds=period,**assessment_function(reports['coarse'],reports['fine'],reports['rest'],
            reports['isotropic'],oracle[str(period)],sign=sign),cycles=reports)
        decisions.append(result);print(json.dumps({k:v for k,v in result.items() if k!='cycles'}),flush=True)
    passed=best_report['key'][0]==0 and all(r['passed'] for r in decisions)
    result=dict(stage=manifest['stage'],skill_id=manifest['skill_id'],status=status,error=error,passed=passed,
        physical_passed=all(r['passed'] for r in decisions),tuning_passed=best_report['key'][0]==0,
        iterations=iteration,selected_iteration=selected,steering_bias=args.bias,initial=initial,final=best_report,
        periods=decisions,elapsed_seconds=time.perf_counter()-started,
        stop_reason='signal' if stop else 'iteration_cap' if iteration>=args.iterations else 'time_budget',
        limitations='Standalone externally paced normalized skill; no biological calibration or runtime skill switching')
    if head_mode:result.pop('steering_bias')
    save_json(output/'summary.json',result)
    return 0 if passed and status=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
