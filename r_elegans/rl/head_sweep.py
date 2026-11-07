"""Stationary head scanning with anterior localization and relative-geometry tests."""
import numpy as np

from .locomotion import cycle_metrics


def anterior_profile():
    import jax.numpy as jnp
    from r_elegans.body.neuromuscular import muscle_longitudinal_positions
    return jnp.exp(-(muscle_longitudinal_positions()/.18)**2)


def head_target(commands,*,amplitude=.04,**unused):
    import jax.numpy as jnp
    from r_elegans.body.neuromuscular import BODY_WALL_MUSCLE_NAMES
    if not np.isfinite(amplitude) or not 0<amplitude<=.08:
        raise ValueError('Head amplitude must be in (0,.08]')
    dorsal=jnp.asarray([n.startswith('d') for n in BODY_WALL_MUSCLE_NAMES])
    return .4+amplitude*jnp.asarray(commands)[:,None]*anterior_profile()[None,:]*jnp.where(dorsal,1.,-1.)


def head_stimulation_vector(model,gain=.1):
    import jax.numpy as jnp
    from r_elegans.body.neuromuscular import BODY_WALL_MUSCLE_NAMES
    dorsal=jnp.asarray([n.startswith('d') for n in BODY_WALL_MUSCLE_NAMES])
    weights=model.neuromuscular_params.synapse_weights
    signed=weights*model.neuromuscular_params.synapse_signs
    score=jnp.sum(signed*(anterior_profile()*jnp.where(dorsal,1.,-1.))[:,None],axis=0)
    return gain*score/jnp.maximum(weights.sum(axis=0),1.)


def relative_head_geometry(angles):
    """Head = last segment center; trunk reference uses first nine centers.

    Returns angle and lateral excursion relative to the trunk chord. Both are
    invariant to whole-body rigid translation/rotation.
    """
    q=np.asarray(angles,dtype=float)
    if q.ndim!=2 or q.shape[1]!=11:
        raise ValueError('Head-sweep geometry requires the normalized 12-segment body')
    orientation=np.c_[np.zeros(len(q)),np.cumsum(q,axis=1)]
    tangent=np.stack([np.cos(orientation),np.sin(orientation)],axis=-1)
    starts=np.concatenate([np.zeros((len(q),1,2)),np.cumsum(tangent[:,:-1]/12,axis=1)],axis=1)
    centers=starts+tangent/24
    trunk=centers[:,8]-centers[:,0]
    axis=np.arctan2(trunk[:,1],trunk[:,0])
    head=centers[:,-1]-centers[:,8]
    lateral=-head[:,0]*np.sin(axis)+head[:,1]*np.cos(axis)
    return orientation[:,-1]-axis,lateral,axis


def head_metrics(position,heading,angles,*,period_steps):
    report=cycle_metrics(position,heading,angles,period_steps=period_steps)
    q=np.asarray(angles,dtype=float);start,end=4*period_steps,10*period_steps
    angle,lateral,trunk_axis=relative_head_geometry(q)
    points=np.arange(4,11)*period_steps
    displacement=np.diff(np.asarray(position)[points],axis=0)
    world_trunk=np.unwrap(np.asarray(heading)[points]+trunk_axis[points])
    window=q[start:end+1];dynamic=window-window.mean(axis=0)
    front=float(np.sqrt(np.mean(dynamic[:,-3:]**2)))
    posterior=float(np.sqrt(np.mean(dynamic[:,:-3]**2)))
    angle_cycles=[float(np.ptp(angle[i*period_steps:(i+1)*period_steps+1])) for i in range(4,10)]
    lateral_cycles=[float(np.ptp(lateral[i*period_steps:(i+1)*period_steps+1])) for i in range(4,10)]
    report.update(head_angle_p2p=float(np.mean(angle_cycles)),head_lateral_p2p=float(np.mean(lateral_cycles)),
        head_angle_cycle_p2p=angle_cycles,head_lateral_cycle_p2p=lateral_cycles,
        anterior_joint_rms=front,posterior_joint_rms=posterior,posterior_anterior_ratio=posterior/max(front,1e-8),
        maximum_cycle_displacement=float(np.max(np.linalg.norm(displacement,axis=1))),
        maximum_trunk_drift=float(np.max(np.abs(np.diff(world_trunk)))))
    return report


def head_checks(metrics,oracle):
    return dict(head_angle=bool(metrics['head_angle_p2p']>=max(.02,.5*oracle['head_angle_p2p'])),
        head_excursion=bool(metrics['head_lateral_p2p']>=max(.0005,.5*oracle['head_lateral_p2p'])),
        localized=bool(metrics['posterior_anterior_ratio']<=.25),
        stationary=bool(metrics['maximum_cycle_displacement']<=.0001),
        no_global_turn=bool(metrics['maximum_trunk_drift']<=.002),
        periodic=bool(metrics['max_shape_closure_relative']<.1),
        stable=bool(np.std(metrics['head_angle_cycle_p2p'])<=.2*metrics['head_angle_p2p']+1e-8
            and np.std(metrics['head_lateral_cycle_p2p'])<=.2*metrics['head_lateral_p2p']+1e-8))


def assess_head_sweep(coarse,fine,rest,isotropic,oracle,**unused):
    checks=head_checks(fine,oracle)
    errors={name:abs(coarse[name]-fine[name]) for name in ('head_angle_p2p','head_lateral_p2p')}
    numerical=all(fine[name]>3*errors[name] for name in errors)
    rest_quiet=all(rest[name]<=.1*fine[name] for name in errors)
    coarse_consistent=all(head_checks(coarse,oracle).values())
    return dict(passed=bool(all(checks.values()) and numerical and rest_quiet and coarse_consistent),
        checks=checks,above_numerical_controls=bool(numerical),rest_quiet=bool(rest_quiet),
        head_angle_p2p=fine['head_angle_p2p'],head_lateral_p2p=fine['head_lateral_p2p'],
        posterior_anterior_ratio=fine['posterior_anterior_ratio'],
        maximum_cycle_displacement=fine['maximum_cycle_displacement'],
        timestep_differences=errors,isotropic_head_angle_p2p=isotropic['head_angle_p2p'])


def head_imitation_metrics(predicted,target):
    import jax.numpy as jnp
    from .motor_curriculum import muscle_sequence_body
    body=muscle_sequence_body(predicted).joint_angles
    reference=muscle_sequence_body(target).joint_angles
    head=body[:,-3:].sum(axis=1);desired=reference[:,-3:].sum(axis=1)
    a,b=head-head.mean(),desired-desired.mean()
    return dict(muscle_rmse=jnp.sqrt(jnp.mean((predicted-target)**2)),
        body_relative_rmse=jnp.sqrt(jnp.mean((body-reference)**2))/jnp.maximum(jnp.sqrt(jnp.mean(reference**2)),1e-8),
        head_correlation=jnp.sum(a*b)/jnp.maximum(jnp.linalg.norm(a)*jnp.linalg.norm(b),1e-8))


def head_selection_key(imitation,probe,oracle,**unused):
    flags=[m['muscle_rmse']<.025 and m['body_relative_rmse']<.25 and m['head_correlation']>.9 for m in imitation]
    flags+=list(head_checks(probe,oracle).values())
    errors=[m['body_relative_rmse']/.25 for m in imitation]
    errors+=[abs(probe['head_angle_p2p']-oracle['head_angle_p2p'])/max(oracle['head_angle_p2p'],.02),
        probe['posterior_anterior_ratio']/.25,probe['maximum_cycle_displacement']/.0001]
    if not np.all(np.isfinite(errors)):raise ValueError('Nonfinite head-sweep validation')
    return sum(not flag for flag in flags),float(max(errors)),float(np.mean(errors))
