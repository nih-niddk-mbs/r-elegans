"""Intrinsic bend-spread/phase curriculum and complete-cycle translation.

Observed oscillations determine frequency/phase. Nothing in these metrics is
fed into neurons or muscles. They score saved forward simulations only.
"""
import numpy as np

from .autonomous_wiggle import amplitude_metrics
from .rhythm_qualification import rhythm_metrics
from r_elegans.trajectory_analysis import analyze_trajectory


def _body_axes(q,h):
    angles=h[:,None]+np.column_stack((np.zeros(len(q)),np.cumsum(q,axis=1)))
    return np.column_stack((np.cos(angles).mean(axis=1),np.sin(angles).mean(axis=1)))


def cycle_averaged_progress(q,position,heading,dt,body_length):
    """Integrate positions over complete cycles, for both crossing directions.

    Regression uses three blocks of cycle-mean positions, not instantaneous
    endpoints. Same-joint phase is not exact whole-shape closure, so report it.
    """
    q,p,h=map(lambda x:np.asarray(x,dtype=float),(q,position,heading))
    if min(dt,body_length)<=0 or not np.isfinite([dt,body_length]).all():raise ValueError('Positive finite units required')
    n=len(q)
    if q.ndim!=2 or n<40 or p.shape!=(n,2) or h.shape!=(n,):raise ValueError('Invalid trajectory shapes')
    if not all(np.isfinite(x).all() for x in (q,p,h)):raise ValueError('Nonfinite trajectory')
    q,p,h=q[n//2:],p[n//2:],h[n//2:];n=len(q)
    groups=np.array_split(np.arange(q.shape[1]),3)
    ref=q[:,groups[-1]].mean(axis=1);ref-=ref.mean()
    axes=_body_axes(q,h);norm=np.linalg.norm(axes,axis=1)
    axis=axes[0]/max(norm[0],1e-12)
    orientation=np.unwrap(np.arctan2(axes[:,1],axes[:,0]))
    turn=float(np.max(np.abs(orientation-orientation[0])))
    result=dict(cycle_progress_valid=False,phase_origins=[],minimum_forward_speed_body_lengths_s=0.,
        mean_cycle_speed_body_lengths_s=0.,forward_efficiency=0.,maximum_axis_turn_rad=turn,
        geometry_valid=bool(norm.min()>.5 and turn<.35),phase_origin_speed_spread_body_lengths_s=0.)
    if np.ptp(ref)<1e-4:return result
    power=np.abs(np.fft.rfft(ref))**2;power[0]=0.;f0=np.fft.rfftfreq(n,dt)[np.argmax(power)]
    if f0<=0:return result
    # Piecewise-linear position integral, including fractional crossing endpoints.
    integral=np.vstack((np.zeros(2),np.cumsum(.5*(p[:-1]+p[1:])*dt,axis=0)))
    def at(samples,values):
        return np.column_stack([np.interp(samples,np.arange(n),values[:,i]) for i in range(values.shape[1])])
    def integrate_to(samples):
        i=np.minimum(np.floor(samples).astype(int),n-2);f=(samples-i)[:,None]
        return integral[i]+dt*(p[i]*f+.5*(p[i+1]-p[i])*f*f)
    all_slopes=[];means=[];closures=[]
    for label,sign in (('rising',1.),('falling',-1.)):
        signal=ref*sign;idx=np.flatnonzero((signal[:-1]<=0)&(signal[1:]>0))
        samples=idx-signal[idx]/(signal[idx+1]-signal[idx]);kept=[]
        for x in samples:
            if not kept or (x-kept[-1])*dt>=.5/f0:kept.append(float(x))
        samples=np.asarray(kept)
        if len(samples)<10:return result
        periods=np.diff(samples)*dt;good=(periods>=.5*np.median(periods))&(periods<=1.5*np.median(periods))
        cycle_mean=np.diff(integrate_to(samples),axis=0)[good]/periods[good,None]
        times=((samples[:-1]+samples[1:])*.5*dt)[good]
        if len(times)<9:return result
        forward=(cycle_mean-p[0])@axis/body_length
        slopes=[]
        for block in np.array_split(np.arange(len(times)),3):
            t=times[block]-times[block].mean();y=forward[block]-forward[block].mean()
            slopes.append(float(np.dot(t,y)/np.dot(t,t)))
        t=times-times.mean();speed=float(np.dot(t,forward-forward.mean())/np.dot(t,t))
        phase_q=at(samples,q);closure=np.sqrt(np.mean(np.diff(phase_q,axis=0)**2,axis=1))[good]
        result['phase_origins'].append(dict(direction=label,cycles=len(times),block_speeds_body_lengths_s=slopes,
            regression_speed_body_lengths_s=speed,median_shape_closure_rms_rad=float(np.median(closure))))
        all_slopes.extend(slopes);means.append(speed);closures.extend(closure.tolist())
    path=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()/body_length)
    result.update(cycle_progress_valid=True,minimum_forward_speed_body_lengths_s=min(all_slopes),
        mean_cycle_speed_body_lengths_s=float(np.mean(means)),
        phase_origin_speed_spread_body_lengths_s=float(abs(means[0]-means[1])),
        median_shape_closure_rms_rad=float(np.median(closures)),
        forward_efficiency=float(np.mean(means)*(n-1)*dt/max(path,1e-12)))
    return result


def _phase_progression(coefficients):
    phases=np.unwrap(np.angle(coefficients[::-1]))  # head -> middle -> posterior
    steps=np.diff(phases);span=float(abs(steps.sum()))
    monotonic=bool(np.all(steps>.1) or np.all(steps<-.1))
    smooth=bool(np.max(np.abs(steps))<=.8*np.pi)
    quality=(min(1.,span/(np.pi/3))*abs(steps.sum())/max(float(np.abs(steps).sum()),1e-12)
        *min(1.,float(np.abs(steps).min())/.1)
        *float(np.clip((.8*np.pi-np.abs(steps).max())/(.2*np.pi),0.,1.)))
    return dict(phases_head_to_tail_rad=phases.tolist(),increments_rad=steps.tolist(),span_rad=span,
        monotonic=monotonic,smooth=smooth,quality=float(quality),qualified=bool(monotonic and smooth and span>=np.pi/3))


def body_wave_metrics(voltage,muscles,body,dt,body_length,neuron_ids,max_joint_angle):
    base=amplitude_metrics(voltage,muscles,body.joint_angles,dt,neuron_ids,max_joint_angle)
    if base.get('reason')=='nonfinite':return base
    v,q,a,p,h=[np.asarray(x) for x in (voltage,body.joint_angles,body.muscle_activation,body.position,body.heading)]
    if not all(np.isfinite(x).all() for x in (a,p,h)):return dict(valid=False,score=-1.,reason='nonfinite')
    progress=cycle_averaged_progress(q,p,h,dt,body_length)
    # Graded rhythms count too; do not force the oscillator to retain RMD identity.
    amps=np.ptp(v[len(v)//2:],axis=0)
    rhythmic=sum(rhythm_metrics(v[:,i],dt,minimum_amplitude_mV=.05)['passed'] for i in np.flatnonzero(amps>=.05))
    valid=bool(base['voltage_min_mV']>-120 and base['voltage_max_mV']<80 and not base['joint_limits_reached']
        and base['muscle_saturation_fraction']<.05 and base['joint_rhythm_passed']
        and base['sustained_joint_p2p_rad']>=.01 and rhythmic>0 and progress['geometry_valid'])
    base.update(valid=valid,score=-1.,wave_qualified=False,curriculum_stage='spread',sustained_neural_rhythms=rhythmic,
        neural_rhythm_threshold_mV=.05,cycle_progress=progress,locomotion_demonstrated=False)
    if not valid:return base
    analysis,_=analyze_trajectory(v,q,a,p,h,dt=dt,body_length=body_length,neuron_ids=neuron_ids)
    frequency=1./analysis['mean_cycle_period_s'] if analysis.get('mean_cycle_period_s') else analysis['dominant_frequency_hz']
    if frequency is None:return base
    start=len(q)//2;q=q[start:];a=a[start:]
    groups=np.array_split(np.arange(q.shape[1]),3)
    signals=np.column_stack([q[:,g].mean(axis=1) for g in groups]+[a[:,g].mean(axis=1) for g in groups])
    signals=signals-signals.mean(axis=0)
    time=np.arange(len(q))*dt;design=np.column_stack((np.cos(2*np.pi*frequency*time),np.sin(2*np.pi*frequency*time)))
    coefficients=np.linalg.lstsq(design,signals,rcond=None)[0]
    complex_coeff=coefficients[0]-1j*coefficients[1]
    harmonic_p2p=2*np.abs(complex_coeff)
    power=np.clip(np.var(design@coefficients,axis=0)/np.maximum(np.var(signals,axis=0),1e-30),0.,1.)
    relative_phases=[]
    for indices in np.array_split(np.arange(len(q)),5):
        b=np.linalg.lstsq(design[indices],signals[indices],rcond=None)[0];c=b[0]-1j*b[1]
        relative_phases.append(np.angle(c)-np.angle(c[[2,2,2,5,5,5]]))
    locking=np.abs(np.exp(1j*np.asarray(relative_phases)).mean(axis=0))
    quality=np.minimum(power/.5,1.)*np.minimum(locking/.8,1.)
    coverage=float(np.min(np.minimum(np.log1p(harmonic_p2p/1e-5)/np.log1p(.003/1e-5),1.)*quality))
    joint_phase=_phase_progression(complex_coeff[:3]);activation_phase=_phase_progression(complex_coeff[3:])
    second_mode=float(analysis['shape_pca_variance_ratios'][1])
    phase_quality=min(joint_phase['quality'],activation_phase['quality'])*min(1.,second_mode/.02)
    stage='coordinate' if coverage>=1.-1e-10 else 'spread'
    qualified=bool(np.all(harmonic_p2p>=.003) and np.all(power>=.5) and np.all(locking>=.8)
        and joint_phase['qualified'] and activation_phase['qualified'] and second_mode>=.02)
    base.update(score=coverage if stage=='spread' else 1.+phase_quality,
        wave_qualified=qualified,curriculum_stage=stage,body_coverage_score=coverage,frequency_hz=float(frequency),
        region_order=['posterior','middle','anterior'],joint_harmonic_p2p_rad=harmonic_p2p[:3].tolist(),
        drive_harmonic_p2p=harmonic_p2p[3:].tolist(),common_frequency_power_fraction=power.tolist(),
        phase_locking=locking.tolist(),joint_phase=joint_phase,drive_phase=activation_phase,
        second_shape_mode_fraction=second_mode,
        interpretation='Intrinsic body-wave primitive; translation is diagnostic, not a wave qualification claim')
    return base


def wave_refinement_agrees(reference,refined):
    if not reference.get('wave_qualified') or not refined.get('wave_qualified'):return False
    for field in ('joint_harmonic_p2p_rad','drive_harmonic_p2p'):
        x=np.asarray(reference[field]);y=np.asarray(refined[field])
        if np.any(np.abs(x-y)>.25*np.maximum(x,.003)):return False
    for field in ('joint_phase','drive_phase'):
        delta=np.asarray(reference[field]['increments_rad'])-np.asarray(refined[field]['increments_rad'])
        if np.max(np.abs(np.angle(np.exp(1j*delta))))>.2:return False
    return abs(reference['frequency_hz']-refined['frequency_hz'])<=.05*reference['frequency_hz']


def cycle_movement_metrics(*args):
    metrics=body_wave_metrics(*args)
    progress=metrics.get('cycle_progress',{})
    metrics.update(valid=bool(metrics.get('wave_qualified') and progress.get('cycle_progress_valid')),
        minimum_forward_speed_body_lengths_s=progress.get('minimum_forward_speed_body_lengths_s',0.),
        forward_efficiency=progress.get('forward_efficiency',0.))
    metrics['score']=metrics['minimum_forward_speed_body_lengths_s'] if metrics['valid'] else -1.
    return metrics


def cycle_movement_achieved(metrics,target_speed,baseline_speed):
    p=metrics.get('cycle_progress',{})
    return bool(metrics['valid'] and metrics['minimum_forward_speed_body_lengths_s']>=max(target_speed,5.*max(baseline_speed,0.))
        and metrics['forward_efficiency']>=.02
        and p['phase_origin_speed_spread_body_lengths_s']<=.25*max(abs(p['mean_cycle_speed_body_lengths_s']),target_speed)
        and p['median_shape_closure_rms_rad']<=.05*metrics['sustained_joint_p2p_rad'])
