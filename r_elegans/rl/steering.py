"""Standalone shallow steering: synthetic target and centroid/body turn tests."""
import numpy as np

from .locomotion import cycle_metrics, assess_propulsion


def steering_target(commands, *, amplitude=.07, bias=.006, delay=.4, dt=.02):
    from .motor_curriculum import traveling_bend_target, bend_target
    if not np.isfinite(bias) or not 0 < abs(bias) <= .012:
        raise ValueError("Use a nonzero signed bias of magnitude <=0.012")
    spatial = (bend_target(2,amplitude=.04)[0]-.4)/.04
    return traveling_bend_target(commands,amplitude=amplitude,delay=delay,dt=dt)+bias*spatial


def trajectory_objective(predicted,target,target_body,*,body_weight=.5):
    """Differentiate actual centroid/joint trajectories, not only muscle traces."""
    import jax
    import jax.numpy as jnp
    from .motor_curriculum import muscle_sequence_body
    muscle=jnp.mean((predicted-target)**2)/jnp.maximum(jnp.mean((target-.4)**2),1e-6)
    if body_weight==0:
        return muscle
    body=jax.vmap(muscle_sequence_body)(predicted)
    joint=jnp.mean((body.joint_angles-target_body.joint_angles)**2)/jnp.maximum(jnp.mean(target_body.joint_angles**2),1e-8)
    position=jnp.mean((body.position-target_body.position)**2)/jnp.maximum(jnp.mean(target_body.position**2),1e-10)
    return muscle+body_weight*(joint+position)/2


def steering_metrics(position,heading,angles,*,period_steps):
    """Initial state included; four warmup and six measured complete cycles."""
    report=cycle_metrics(position,heading,angles,period_steps=period_steps)
    points=np.arange(4,11)*period_steps
    q=np.asarray(angles,dtype=float)[points]
    orientations=np.c_[np.zeros(len(q)),np.cumsum(q,axis=1)]
    weights=np.ones(orientations.shape[1]);weights[[0,-1]]=.5
    axis=np.unwrap(np.asarray(heading)[points]+np.arctan2(np.sin(orientations)@weights,np.cos(orientations)@weights))
    yaw=np.diff(axis)
    displacement=np.diff(np.asarray(position)[points],axis=0)
    defined=bool(np.all(np.linalg.norm(displacement,axis=1)>1e-7))
    path_turn=np.diff(np.unwrap(np.arctan2(displacement[:,1],displacement[:,0]))) if defined else np.zeros(5)
    report.update(mean_signed_yaw_per_cycle=float(yaw.mean()),yaw_per_cycle=yaw.tolist(),
        mean_path_turn_per_cycle=float(path_turn.mean()),path_turn_per_cycle=path_turn.tolist(),
        path_turn_defined=defined)
    return report


def turn_checks(metrics,oracle,*,sign):
    yaw=metrics['mean_signed_yaw_per_cycle'];path=metrics['mean_path_turn_per_cycle']
    yaw_floor=max(5e-5,.5*abs(oracle['mean_signed_yaw_per_cycle']))
    path_floor=max(5e-5,.5*abs(oracle['mean_path_turn_per_cycle']))
    return dict(axis_turn=bool(sign*yaw>=yaw_floor),
        path_turn=bool(metrics['path_turn_defined'] and sign*path>=path_floor),
        consistent_turn=bool(np.mean(sign*np.asarray(metrics['yaw_per_cycle'])>0)>=.8
            and np.mean(sign*np.asarray(metrics['path_turn_per_cycle'])>0)>=.8
            and np.std(metrics['yaw_per_cycle'])<=.3*abs(yaw)+1e-8),
        axis_path_agreement=bool(abs(yaw-path)<=.5*abs(yaw)+1e-6),
        minimum_axis_turn=yaw_floor,minimum_path_turn=path_floor)


def assess_steering(coarse,fine,rest,isotropic,oracle,*,sign):
    propulsion=assess_propulsion(coarse,fine,rest,isotropic=isotropic)
    turn=turn_checks(fine,oracle,sign=sign)
    yaw_error=abs(coarse['mean_signed_yaw_per_cycle']-fine['mean_signed_yaw_per_cycle'])
    path_error=abs(coarse['mean_path_turn_per_cycle']-fine['mean_path_turn_per_cycle'])
    above_controls=bool(sign*coarse['mean_signed_yaw_per_cycle']>0
        and sign*coarse['mean_path_turn_per_cycle']>0
        and abs(fine['mean_signed_yaw_per_cycle'])>max(3*yaw_error,5*abs(rest['mean_signed_yaw_per_cycle']))
        and abs(fine['mean_path_turn_per_cycle'])>3*path_error)
    # Isotropic drag controls translation. It need not forbid geometric rotation.
    passed=bool(propulsion['passed'] and propulsion['direction']=='forward'
        and all(turn[k] for k in ('axis_turn','path_turn','consistent_turn','axis_path_agreement'))
        and above_controls)
    return dict(passed=passed,propulsion=propulsion,turn=turn,above_numerical_controls=above_controls,
        axis_turn_radians_per_cycle=fine['mean_signed_yaw_per_cycle'],
        path_turn_radians_per_cycle=fine['mean_path_turn_per_cycle'],
        axis_timestep_difference=yaw_error,path_timestep_difference=path_error)


def selection_key(imitation,probe,oracle,*,sign):
    from .motor_curriculum import traveling_passes
    turn=turn_checks(probe,oracle,sign=sign)
    flags=[traveling_passes(m) for m in imitation]
    flags += [turn[k] for k in ('axis_turn','path_turn','consistent_turn','axis_path_agreement')]
    flags += [probe['mean_forward_body_lengths']>=.001,probe['max_shape_closure_relative']<.1]
    errors=[m['body_relative_rmse']/.25 for m in imitation]
    errors += [abs(probe['mean_signed_yaw_per_cycle']-oracle['mean_signed_yaw_per_cycle'])/
               max(abs(oracle['mean_signed_yaw_per_cycle']),5e-5)]
    if not np.all(np.isfinite(errors)):
        raise ValueError("Nonfinite steering validation")
    return sum(not flag for flag in flags),float(max(errors)),float(np.mean(errors))
