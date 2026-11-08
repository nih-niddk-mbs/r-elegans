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
