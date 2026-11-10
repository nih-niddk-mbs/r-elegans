"""Sustained body-wide amplitude, without phase, direction or motion control."""
import numpy as np
from r_elegans.amplitude_analysis import sustained_excursions


def body_amplitude_metrics(voltage,muscles,body,dt,body_length,neuron_ids,max_joint_angle):
    """Reward median per-joint amplitudes in each contiguous body third.

    No coherent-frequency, phase progression, PCA, heading, speed or efficiency
    criterion. Per-joint scoring avoids cancellation of opposing phases.
    """
    v,m,q,a=map(np.asarray,(voltage,muscles,body.joint_angles,body.muscle_activation))
    if not all(np.isfinite(x).all() for x in (v,m,q,a)):
        return dict(valid=False,score=-1.,reason='nonfinite',body_amplitude_qualified=False)
    qamp,qrepeat=sustained_excursions(q);aamp,arepeat=sustained_excursions(a)
    vamp,_=sustained_excursions(v)
    late=len(q)//2;groups=np.array_split(np.arange(q.shape[1]),3)
    regional_q=np.asarray([np.median(qamp[g]) for g in groups])
    regional_a=np.asarray([np.median(aamp[g]) for g in groups])
    saturation=float(np.mean((m[late:]<.01)|(m[late:]>.99)))
    joint_limits=bool(np.any(np.abs(q)>=.99*max_joint_angle))
    neural_count=int((vamp>=.05).sum())
    valid=bool(v.min()>-120 and v.max()<80 and saturation<.05 and not joint_limits
        and neural_count>0 and qamp.max()>1e-5)
    coverage=float(np.min(np.minimum(np.log1p(regional_q/1e-5)/np.log1p(.003/1e-5),1.)))
    qualified=bool(valid and np.all(regional_q>=.003))
    return dict(valid=valid,score=coverage if valid else -1.,body_amplitude_qualified=qualified,
        curriculum_stage='amplitude-qualified' if qualified else 'body-amplitude',body_coverage_score=coverage,
        region_order=['posterior','middle','anterior'],regional_joint_median_p2p_rad=regional_q.tolist(),
        regional_drive_median_p2p=regional_a.tolist(),joint_sustained_p2p_rad=qamp.tolist(),drive_sustained_p2p=aamp.tolist(),
        repetitive_joint_count=int(qrepeat.sum()),repetitive_drive_count=int(arepeat.sum()),
        sustained_joint_p2p_rad=float(qamp.max()),joint_p2p_max_rad=float(np.ptp(q[late:],axis=0).max()),
        sustained_neural_rhythms=neural_count,neural_rhythm_threshold_mV=.05,
        voltage_min_mV=float(v.min()),voltage_max_mV=float(v.max()),muscle_saturation_fraction=saturation,
        joint_limits_reached=joint_limits,wave_qualified=False,locomotion_demonstrated=False,
        interpretation='Body-wide sustained amplitude only; relative phase, direction and propulsion unassessed')


def amplitude_refinement_agrees(reference,refined):
    if not reference.get('body_amplitude_qualified') or not refined.get('body_amplitude_qualified'):return False
    for field in ('regional_joint_median_p2p_rad',):
        a=np.asarray(reference[field]);b=np.asarray(refined[field])
        if np.any(np.abs(a-b)>.25*np.maximum(a,.003)):return False
    return True
