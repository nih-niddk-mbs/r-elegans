"""Bounded density/connection proposals and autonomous timing objective."""
import numpy as np
from scipy.signal import find_peaks
from .connection_amplitude import amplitude_score, propose_connections
from .hybrid_assessment import cycle_excursions, peak_frequency

SWIM_TARGET_HZ=1.8
TIMING_SOURCE='https://pmc.ncbi.nlm.nih.gov/articles/PMC2996704/'


def body_frequency_profiles(trace, window_seconds):
    t=np.asarray(trace['dense_time_s']);q=np.asarray(trace['dense_joints']);profiles=[]
    for start,end in ((t[-1]-2*window_seconds,t[-1]-window_seconds),(t[-1]-window_seconds,t[-1])):
        use=(t>=start-1e-12)&(t<=end+1e-12);clock=t[use];shape=q[use]
        cycles=cycle_excursions(shape,late_only=False)['joint_cycle_excursion_rad'];regions=[]
        for indices in np.array_split(np.arange(11),3):
            frequencies=[];joints=[]
            for j in indices:
                prominence=max(1e-6,.1*cycles[j]);signal=shape[:,j]
                if (cycles[j]<1e-6 or len(find_peaks(signal,prominence=prominence)[0])<3
                        or len(find_peaks(-signal,prominence=prominence)[0])<3):continue
                f=peak_frequency(clock,signal,prominence)
                if f is not None:frequencies.append(f);joints.append(int(j))
            regions.append(dict(joints=joints,frequencies_hz=frequencies,
                                median_hz=float(np.median(frequencies)) if frequencies else None))
        profiles.append(regions)
    return profiles


def timing_score(trace, reference_index, *, target_hz, window_seconds, baseline):
    result=amplitude_score(trace,reference_index,window_seconds=window_seconds,targets=np.full(3,.003))
    profiles=result['reference_motor_profiles']
    rhythmic=bool(result['neural_guard'] and all(
        p['amplitude_mV']>=max(.1,.5*b['amplitude_mV']) for p,b in zip(profiles,baseline['reference_motor_profiles'])))
    coverage=bool(np.all(np.asarray(result['window_cycle_excursions_rad'])>=
                          .5*np.asarray(baseline['window_cycle_excursions_rad'])))
    body_profiles=body_frequency_profiles(trace,window_seconds)
    frequencies=[region['median_hz'] for window in body_profiles for region in window]
    valid=all(f is not None and np.isfinite(f) and f>0 for f in frequencies)
    error=float(np.mean(np.abs(np.log(np.asarray(frequencies)/target_hz)))) if valid else 20.
    qualified=bool(valid and all(.9*target_hz<=f<=1.1*target_hz for f in frequencies))
    result.update(amplitude_data_loss=result['data_loss'],frequency_loss=error,
                  data_loss=result['data_loss']+error,target_hz=target_hz,
                  neural_guard=rhythmic,baseline_body_coverage_preserved=coverage,
                  timing_target_passed=qualified,body_frequency_profiles=body_profiles,
                  frequency_observable='Body-joint curvature rhythm; neuronal oscillation rates are not equated to gait frequency')
    result['safety_passed']=bool(result['safety_passed'] and coverage)
    result['target_passed']=bool(result['target_passed'] and rhythmic and coverage and qualified)
    return result


def propose_timing(best,spec,rng,sigma,lower,upper,focus_cells):
    count=int(spec.channels.sum());mode=int(rng.integers(4))
    if mode==0:
        proposed,description=propose_connections(best[count:],spec,rng,sigma,lower[count:],upper[count:],focus_cells)
        result=best.copy();result[count:]=proposed
        return result,dict(kind='connections',details=description)
    noise=np.zeros(len(best));rows,columns=np.nonzero(spec.channels)
    if mode==1:
        channel=int(rng.choice(np.unique(columns)))
        mask=columns==channel
        if rng.random()<.75:mask &= np.isin(rows,focus_cells)
        selected=np.flatnonzero(mask);noise[selected]=rng.normal()
    elif mode==2:
        pool=np.asarray(focus_cells) if rng.random()<.75 else np.arange(spec.channels.shape[0])
        cells=rng.choice(pool,size=min(8,len(pool)),replace=False)
        selected=np.flatnonzero(np.isin(rows,cells));noise[selected]=rng.normal(size=len(selected))
    else:
        selected=rng.choice(len(best),size=min(128,len(best)),replace=False)
        noise[selected]=rng.normal(size=len(selected))
    return np.clip(best+sigma*noise,lower,upper),dict(kind='densities_or_joint',mode=mode,coordinates=len(selected))
