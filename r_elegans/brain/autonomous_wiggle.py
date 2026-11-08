"""Amplitude-only channel fitting, without injected currents or a gait teacher."""
import jax.numpy as jnp
import numpy as np

from .single_compartment import CHANNEL_INDEX, CHANNEL_NAMES
from .rhythm_qualification import rhythm_metrics


def autonomous_fit_reference(params, fixed, neuron_ids):
    """Only existing, non-leak RMD densities are unknowns in this first stage.

    RMD density priors are exploratory assumptions, so they can fit here. This
    does not change the conservative calcium-recording fitting policy.
    """
    mask=np.zeros(params.log_channel_scale.shape,dtype=bool)
    for i,name in enumerate(neuron_ids):
        if name.startswith('RMD'):
            mask[i]=np.asarray(fixed.cells.conductances_nS[i])>0
    mask[:,CHANNEL_INDEX['LEAK']]=False
    if not mask.any():raise ValueError('No eligible RMD conductances')
    return fixed._replace(bias_pA=jnp.zeros_like(fixed.bias_pA)),mask


def apply_density_offsets(initial, offsets, mask):
    indices=np.nonzero(mask)
    return initial._replace(log_channel_scale=initial.log_channel_scale.at[indices].add(offsets))


def density_proposal(best, mask, rng, sigma, maximum_factor=4.):
    """Mix shared-channel and independent-cell proposals; no symmetry enforced."""
    if not np.isfinite(maximum_factor) or maximum_factor<=1:raise ValueError('Factor must exceed one')
    _,channels=np.nonzero(mask)
    noise=rng.normal(size=len(CHANNEL_NAMES))[channels] if rng.random()<.5 else rng.normal(size=len(best))
    return np.clip(best+sigma*noise,-np.log(maximum_factor),np.log(maximum_factor))


def amplitude_metrics(voltage, muscles, joints, dt, neuron_ids, max_joint_angle):
    """Reward repeated late bends, never a static posture or startup deflection."""
    v,m,q=map(np.asarray,(voltage,muscles,joints))
    if not all(np.isfinite(x).all() for x in (v,m,q)):
        return dict(valid=False,score=-1.,reason='nonfinite')
    late=len(q)//2
    peak_to_peak=np.ptp(q[late:],axis=0)
    joint=int(np.argmax(peak_to_peak))
    # Analyze the whole trace with late-window persistence gates, as in E027.
    rhythm=rhythm_metrics(q[:,joint],dt,minimum_amplitude_mV=1e-4)
    rmd={name:rhythm_metrics(v[:,i],dt) for i,name in enumerate(neuron_ids) if name.startswith('RMD')}
    count=sum(x['passed'] for x in rmd.values())
    amplitude=float(min(rhythm.get('window_amplitudes_mV',[0.])))
    saturation=float(np.mean((m[late:]<.01)|(m[late:]>.99)))
    joint_limits=bool(np.any(np.abs(q)>=.99*max_joint_angle))
    bounded=bool(v.min()>-120 and v.max()<80)
    valid=bounded and not joint_limits and saturation<.05 and count>=1 and rhythm['passed']
    return dict(valid=bool(valid),score=amplitude if valid else -1.,
        sustained_joint_p2p_rad=amplitude,joint_p2p_max_rad=float(peak_to_peak[joint]),
        selected_joint=joint,joint_rhythm_passed=rhythm['passed'],
        rhythmic_rmd=count,rmd=rmd,muscle_p2p_max=float(np.ptp(m[late:],axis=0).max()),
        voltage_min_mV=float(v.min()),voltage_max_mV=float(v.max()),
        muscle_saturation_fraction=saturation,joint_limits_reached=joint_limits,
        interpretation='Amplitude only; no propulsion, spatial phase or biological fidelity criterion')


def frozen_fit_fields_unchanged(candidate, initial, mask):
    """Exact restrictions, including all recurrent strengths and leak scales."""
    return (np.array_equal(np.asarray(candidate.log_channel_scale)[~mask],np.asarray(initial.log_channel_scale)[~mask])
        and np.array_equal(candidate.raw_chemical,initial.raw_chemical)
        and np.array_equal(candidate.raw_gap,initial.raw_gap))


def movement_metrics(voltage, muscles, body, dt, body_length, neuron_ids, max_joint_angle):
    """Repeated headward net progress after warmup; path length is not reward.

    Forward is the mean tail-to-head axis at the start of the late window,
    fixed over that window. Three separate blocks reject one startup impulse.
    """
    result=amplitude_metrics(voltage,muscles,body.joint_angles,dt,neuron_ids,max_joint_angle)
    if result.get('reason')=='nonfinite':return result
    position=np.asarray(body.position);q=np.asarray(body.joint_angles);heading=np.asarray(body.heading)
    if not all(np.isfinite(x).all() for x in (position,q,heading)):
        return dict(valid=False,score=-1.,reason='nonfinite')
    late=len(q)//2
    angles=heading[late:,None]+np.concatenate((np.zeros((len(q)-late,1)),np.cumsum(q[late:],axis=1)),axis=1)
    axes=np.stack((np.cos(angles).mean(axis=1),np.sin(angles).mean(axis=1)),axis=1)
    norm=np.linalg.norm(axes[0]);axis=axes[0]/max(norm,1e-12)
    orientation=np.unwrap(np.arctan2(axes[:,1],axes[:,0]))
    turning=float(np.max(np.abs(orientation-orientation[0])))
    blocks=np.array_split(np.arange(late,len(q)),3)
    speeds=[float(np.dot(position[b[-1]]-position[b[0]],axis)/body_length/((len(b)-1)*dt)) for b in blocks]
    progress=float(np.dot(position[-1]-position[late],axis)/body_length)
    path=float(np.linalg.norm(np.diff(position[late:],axis=0),axis=1).sum()/body_length)
    valid=bool(result['valid'] and result['sustained_joint_p2p_rad']>=.01 and turning<.35 and norm>.5)
    result.update(valid=valid,score=min(speeds) if valid else -1.,
        forward_speed_blocks_body_lengths_s=speeds,minimum_forward_speed_body_lengths_s=min(speeds),
        forward_progress_body_lengths=progress,path_body_lengths=path,
        forward_efficiency=progress/max(path,1e-12),maximum_axis_turn_rad=turning,
        interpretation='Repeated forward progress with autonomous wiggle; not animal-matched locomotion')
    return result


def movement_achieved(metrics, target_speed, baseline_speed):
    required=max(target_speed,5.*max(baseline_speed,0.))
    return bool(metrics['valid'] and metrics['minimum_forward_speed_body_lengths_s']>=required
        and metrics['forward_efficiency']>=.02)


def movement_refinement_agrees(reference, refined, target_speed):
    """Reject improvements dominated by a timestep-dependent translation error."""
    if not reference['valid'] or not refined['valid']:return False
    a=reference['minimum_forward_speed_body_lengths_s'];b=refined['minimum_forward_speed_body_lengths_s']
    return abs(a-b)<=.25*max(abs(a),target_speed)
