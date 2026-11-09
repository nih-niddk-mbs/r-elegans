"""Read-only spatial and equal-phase analysis of saved unpaced trajectories."""
import numpy as np
from scipy.signal import coherence


def _axes(q, heading):
    angles=heading[:,None]+np.column_stack((np.zeros(len(q)),np.cumsum(q,axis=1)))
    axes=np.column_stack((np.cos(angles).mean(axis=1),np.sin(angles).mean(axis=1)))
    return axes/np.maximum(np.linalg.norm(axes,axis=1,keepdims=True),1e-12)


def analyze_trajectory(voltage, q, activation, position, heading, *, dt, body_length, neuron_ids):
    """Use late half; index 0 is tail and last joint is anterior in this model.

    Equal-phase samples are interpolated upward mean crossings at the strongest
    joint. Whole-shape mismatch is reported: one joint is not a full gait phase.
    Spectral phase is descriptive, not evidence of neural causality.
    """
    if not np.isfinite(dt) or not np.isfinite(body_length) or min(dt,body_length)<=0:
        raise ValueError('Positive finite sample interval/body length required')
    v,q,a,p,h=[np.asarray(x,dtype=float) for x in (voltage,q,activation,position,heading)]
    n=len(q)
    if n<40 or q.ndim!=2 or a.shape!=q.shape or p.shape!=(n,2) or h.shape!=(n,) or v.shape!=(n,len(neuron_ids)):
        raise ValueError('Inconsistent or too-short trajectory arrays')
    if not all(np.isfinite(x).all() for x in (v,q,a,p,h)):raise ValueError('Nonfinite trajectory')
    start=n//2;t=np.arange(n)*dt;late=t[start:];y=q[start:];centered=y-y.mean(axis=0)
    amp=np.ptp(y,axis=0);ref=int(np.argmax(amp));variance=np.var(y,axis=0);total=float(variance.sum())
    eigenvalues,eigenvectors=np.linalg.eigh(centered.T@centered)
    eigenvalues=np.maximum(eigenvalues[::-1],0.);eigenvectors=eigenvectors[:,::-1]
    ratios=eigenvalues/max(float(eigenvalues.sum()),1e-30) if total>1e-25 else np.zeros_like(eigenvalues)
    pc=centered@eigenvectors[:,:2]
    axis=_axes(y,h[start:])[0];forward=(p[start:]-p[start])@axis/body_length
    blocks=np.array_split(np.arange(len(late)),3)
    raw_speeds=[float((forward[b[-1]]-forward[b[0]])/((len(b)-1)*dt)) for b in blocks]
    neural_amp=np.ptp(v[start:],axis=0);order=np.argsort(neural_amp)[::-1][:15]
    result=dict(samples=n,dt_s=dt,recording_seconds=n*dt,late_seconds=len(late)*dt,body_length=body_length,
        strongest_joint=ref,joint_indexing='0 tail -> last anterior; matches head_at_end=True projection',
        joint_p2p_rad=amp.tolist(),activation_p2p=np.ptp(a[start:],axis=0).tolist(),
        anterior_three_joint_variance_fraction=float(variance[-3:].sum()/total) if total>1e-25 else None,
        shape_pca_variance_ratios=ratios.tolist(),
        active_joint_threshold_rad=max(1e-4,float(amp.max())*.1),
        raw_forward_block_speeds_body_lengths_s=raw_speeds,
        raw_net_forward_body_lengths=float(forward[-1]),
        raw_net_speed_body_lengths_s=float(forward[-1]/(late[-1]-late[0])),
        neural_p2p_over_2mV=int((neural_amp>2.).sum()),
        top_neurons=[dict(neuron=str(neuron_ids[i]),p2p_mV=float(neural_amp[i])) for i in order],
        cycles=0,dominant_frequency_hz=None,equal_phase_regression_speed_body_lengths_s=None,
        median_cycle_speed_body_lengths_s=None,positive_cycle_fraction=None,median_shape_closure_rms_rad=None,
        spectral_phase=[],phase_interpretation='Single-frequency phase modulo a period; tiny-amplitude joints excluded')
    derived=dict(time_s=late,forward_body_lengths=forward,joint_demeaned_rad=centered,shape_pc=pc,
        phase_times_s=np.empty(0),phase_forward_body_lengths=np.empty(0),cycle_speed_body_lengths_s=np.empty(0))
    if amp.max()<1e-4:return result,derived
    power=np.abs(np.fft.rfft(centered[:,ref]))**2;freq=np.fft.rfftfreq(len(late),dt)
    power[0]=0.;f0=float(freq[np.argmax(power)])
    if f0<=0:return result,derived
    result['dominant_frequency_hz']=f0
    carrier=np.exp(-2j*np.pi*f0*(late-late[0]));coeff=centered.T@carrier
    active=amp>=result['active_joint_threshold_rad']
    nperseg=min(len(late)//2,max(64,round(8./(f0*dt))))
    for i in range(q.shape[1]-1,-1,-1):
        if not active[i]:continue
        cf,cv=coherence(centered[:,ref],centered[:,i],fs=1./dt,nperseg=nperseg)
        coh=float(cv[np.argmin(np.abs(cf-f0))])
        phase=float(np.angle(coeff[i]/coeff[ref]));lag=-phase/(2*np.pi*f0)
        result['spectral_phase'].append(dict(joint=i,p2p_rad=float(amp[i]),coherence=coh,
            phase_relative_to_reference_rad=phase,lag_modulo_period_s=float(lag)))
    signal=centered[:,ref];cross=np.flatnonzero((signal[:-1]<=0)&(signal[1:]>0))
    fraction=-signal[cross]/(signal[cross+1]-signal[cross])
    sample=cross+fraction
    # Suppress extra harmonic/noise crossings within half the dominant period.
    kept=[]
    for x in sample:
        if not kept or (x-kept[-1])*dt>=.5/f0:kept.append(float(x))
    if len(kept)<3:return result,derived
    sample=np.asarray(kept);phase_t=late[0]+sample*dt
    def interpolate(values):
        return np.column_stack([np.interp(sample,np.arange(len(late)),values[:,i]) for i in range(values.shape[1])])
    phase_p=interpolate(p[start:]);phase_q=interpolate(y)
    phase_h=np.interp(sample,np.arange(len(late)),np.unwrap(h[start:]))
    phase_axes=_axes(phase_q,phase_h)
    delta=np.diff(phase_p,axis=0)/body_length
    speeds=(delta*phase_axes[:-1]).sum(axis=1)/np.diff(phase_t)
    phase_forward=(phase_p-p[start])@axis/body_length
    slope=float(np.polyfit(phase_t-phase_t.mean(),phase_forward,1)[0])
    closure=np.sqrt(np.mean(np.diff(phase_q,axis=0)**2,axis=1))
    result.update(cycles=len(speeds),equal_phase_regression_speed_body_lengths_s=slope,
        median_cycle_speed_body_lengths_s=float(np.median(speeds)),
        cycle_speed_percentiles_body_lengths_s=np.percentile(speeds,[5,50,95]).tolist(),
        positive_cycle_fraction=float(np.mean(speeds>0)),median_shape_closure_rms_rad=float(np.median(closure)),
        maximum_shape_closure_rms_rad=float(closure.max()),
        median_cycle_heading_change_rad=float(np.median(np.diff(phase_h))),
        mean_cycle_period_s=float(np.mean(np.diff(phase_t))))
    derived.update(phase_times_s=phase_t,phase_forward_body_lengths=phase_forward,cycle_speed_body_lengths_s=speeds)
    return result,derived


def plot_analysis(result, derived, output, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=derived['time_s'];last=t>=max(t[0],t[-1]-5.)
    stride=max(1,int(last.sum())//1500);indices=np.flatnonzero(last)[::stride]
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    q=derived['joint_demeaned_rad'][indices];lim=max(float(np.max(np.abs(q))),1e-6)
    im=axes[0,0].imshow(q.T,aspect='auto',origin='lower',extent=[t[indices[0]],t[indices[-1]],-.5,q.shape[1]-.5],
        cmap='RdBu_r',vmin=-lim,vmax=lim)
    fig.colorbar(im,ax=axes[0,0],label='demeaned joint angle (rad)')
    axes[0,0].set(xlabel='simulation time (s)',ylabel='joint: tail 0 → anterior 10',title='Final five seconds of shape')
    axes[0,1].plot(result['joint_p2p_rad'],'o-');axes[0,1].set(xlabel='joint (tail → anterior)',ylabel='peak-to-peak (rad)',title='Spatial bend amplitude')
    phase=[x for x in result['spectral_phase'] if x['coherence']>=.8]
    if phase:axes[0,2].plot([x['joint'] for x in phase],[x['lag_modulo_period_s']*1000 for x in phase],'o-')
    axes[0,2].set(xlabel='joint (tail → anterior)',ylabel='relative lag modulo period (ms)',title='Active joints, coherence ≥0.8')
    axes[1,0].plot(t[indices],derived['forward_body_lengths'][indices],label='raw centroid')
    pt=derived['phase_times_s'];pf=derived['phase_forward_body_lengths'];sel=pt>=t[indices[0]]
    axes[1,0].plot(pt[sel],pf[sel],'o',ms=3,label='same reference-joint phase')
    axes[1,0].set(xlabel='time (s)',ylabel='forward position (body lengths)',title='Oscillation versus equal-phase position');axes[1,0].legend()
    if len(pt):axes[1,1].plot(pt,pf,'o-',ms=2)
    axes[1,1].set(xlabel='time (s)',ylabel='forward position (body lengths)',title='Equal-phase trajectory, full late window')
    pc=derived['shape_pc'][indices]
    axes[1,2].plot(pc[:,0],pc[:,1],lw=.5)
    axes[1,2].set_aspect('equal',adjustable='datalim')
    axes[1,2].set(xlabel='shape PC1 (rad)',ylabel='shape PC2 (rad)',title=f"Shape loop; PC1 {result['shape_pca_variance_ratios'][0]*100:.3f}% variance")
    fig.suptitle(title);fig.savefig(output,dpi=150);plt.close(fig)
