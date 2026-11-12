"""Phase-free post-cue alternation prerequisites, not biological gait fitting."""
import numpy as np
from scipy.signal import find_peaks


def alternation_statistics(trace,target_rad=1e-5,begin_s=4.):
    t=np.asarray(trace['dense_time_s']);q=np.asarray(trace['dense_joints'])
    if target_rad<=0 or not np.isfinite(target_rad) or q.shape!=(len(t),11) or not np.isfinite(q).all():
        raise ValueError('Finite11-joint trace and positive target required')
    late=q[t>=begin_s];times=t[t>=begin_s]
    if len(late)<40 or np.any(np.diff(times)<=0):raise ValueError('Ordered post-cue samples required')
    prominence=max(1e-8,.1*target_rad)
    ranges=[];turns=[]
    for indices in np.array_split(np.arange(len(late)),2):
        w=late[indices];clock=times[indices]-times[indices].mean()
        slope=clock@w/float(clock@clock)
        detrended=w-w.mean(axis=0)-clock[:,None]*slope
        ranges.append(np.ptp(detrended,axis=0))
        turns.append([len(find_peaks(w[:,j],prominence=prominence)[0])+len(find_peaks(-w[:,j],prominence=prominence)[0]) for j in range(11)])
    amplitudes=np.asarray(ranges);minimum=amplitudes.min(axis=0);ratio=minimum/np.maximum(amplitudes.max(axis=0),1e-12)
    peaks=np.array([len(find_peaks(late[:,j],prominence=prominence)[0]) for j in range(11)])
    troughs=np.array([len(find_peaks(-late[:,j],prominence=prominence)[0]) for j in range(11)])
    coverage=np.asarray(turns).min(axis=0)>0
    losses=np.maximum(0,1-minimum/target_rad)+.5*(np.maximum(0,1-peaks/2)+np.maximum(0,1-troughs/2))
    losses+=.5*(~coverage)+np.maximum(0,.5-ratio)
    qualified=(minimum>=target_rad)&(peaks>=2)&(troughs>=2)&coverage&(ratio>=.5)
    joint=int(np.argmin(np.where(qualified,0.,losses)))
    return dict(total=float(losses[joint]),pattern_pass=bool(qualified[joint]),alternating_joint=joint,
        alternating_p2p_rad=float(minimum[joint]),peaks=int(peaks[joint]),troughs=int(troughs[joint]),
        window_amplitude_ratio=float(ratio[joint]),postcue_begin_s=float(begin_s),
        minimum_window_p2p_by_joint_rad=minimum.tolist(),peaks_by_joint=peaks.tolist(),troughs_by_joint=troughs.tolist(),
        qualified_by_joint=qualified.tolist(),
        scope='Short post-cue absolute-joint alternation; finite-window engineering criteria, not long autonomy or a biological gait')


def alternation_metrics(active,sham,neuron_ids,protocol,*,target_rad=1e-5,onset_shift=0.):
    from .sensorimotor_curriculum import response_metrics
    result=response_metrics(active,sham,neuron_ids,protocol,onset_shift=onset_shift)
    pattern=alternation_statistics(active,target_rad)
    sensory_deficit=max(0.,1-result['awa_depolarization_mV']/protocol.sensory_target_mV)
    safety=max(0.,result['maximum_absolute_joint_rad']/protocol.maximum_joint_rad-1.)
    result.update(pattern)
    result['total']=pattern['total']+.25*sensory_deficit+10*safety+result['pre_cue_difference_rad']/protocol.bend_target_rad
    result['passed']=bool(pattern['pattern_pass'] and result['awa_depolarization_mV']>=protocol.sensory_target_mV
        and result['maximum_absolute_joint_rad']<protocol.maximum_joint_rad and result['pre_cue_difference_rad']<1e-7)
    return result


def pattern_resolution_check(coarse,fine,target_rad):
    """Check same-joint amplitude and turning-count stability, no phase target."""
    j=coarse['alternating_joint'];a=coarse['minimum_window_p2p_by_joint_rad'][j];b=fine['minimum_window_p2p_by_joint_rad'][j]
    error=abs(a-b);allowed=.25*max(target_rad,a)
    counts=all(abs(coarse[key][j]-fine[key][j])<=1 for key in ('peaks_by_joint','troughs_by_joint'))
    return dict(joint=j,amplitude_difference_rad=error,allowed_difference_rad=allowed,
        turning_counts_consistent=counts,passed=bool(fine['qualified_by_joint'][j] and error<=allowed and counts))
