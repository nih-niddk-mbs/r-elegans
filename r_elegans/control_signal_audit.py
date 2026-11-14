"""Read-only, sampled cue/sham diagnostics; no causal or biological certification."""
import numpy as np
from scipy.special import expit

WINDOWS={'pre':(30.,40.),'stop_cue':(40.,42.),'after_stop':(42.,48.),
         'rest':(48.,58.),'start_cue':(58.,60.),'after_start':(60.,68.),'resumed':(68.,78.)}


def paired_summary(t,cue,sham,names):
    t=np.asarray(t);cue=np.asarray(cue);sham=np.asarray(sham)
    if (t.ndim!=1 or len(t)<2 or np.any(np.diff(t)<=0) or cue.shape!=sham.shape
            or cue.shape!=(len(t),len(names)) or not np.isfinite(cue).all() or not np.isfinite(sham).all()):
        raise ValueError('Finite aligned paired arrays and named columns required')
    result={}
    for label,(a,b) in WINDOWS.items():
        use=(t>=a)&(t<b)
        if not np.any(use):raise ValueError('Trace must cover all audit windows')
        x=cue[use];y=sham[use];d=x-y
        result[label]=[dict(name=name,mean_cue=float(x[:,j].mean()),mean_sham=float(y[:,j].mean()),
            cue_p2p=float(np.ptp(x[:,j])),sham_p2p=float(np.ptp(y[:,j])),
            maximum_absolute_difference=float(np.max(abs(d[:,j]))),rms_difference=float(np.sqrt(np.mean(d[:,j]**2)))) for j,name in enumerate(names)]
    return result


def source_currents(voltage,chemical,gap,reversal,threshold,slope,sources,targets):
    """Instantaneous inward pA at recorded samples, not integrated channel currents.

    Matrix convention [postsynaptic,presynaptic]. Includes target voltage changes;
    current differences are descriptive and do not isolate causal mediation.
    """
    v=np.asarray(voltage);src=np.asarray(sources,dtype=int);dst=np.asarray(targets,dtype=int)
    release=expit((v[:,src]-np.asarray(threshold)[src])/np.asarray(slope)[src])
    w=np.asarray(chemical)[np.ix_(dst,src)];g=np.asarray(gap)[np.ix_(dst,src)]
    chemical_current=release@(w*np.asarray(reversal)[src]).T-v[:,dst]*(release@w.T)
    gap_current=v[:,src]@g.T-v[:,dst]*g.sum(axis=1)
    return chemical_current,gap_current


def replay_chemical(t,events,levels,initial_receptor,initial_adapted,reference,tau_r,tau_a,gain):
    """Offline chemical-only replay at the nominal interval; not recorded injection.

    Initial entry is pre-update. Input uses accumulated body time, as liquid_step
    does. No neuron/body simulation, sensory signs or kinetic parameters change.
    """
    t=np.asarray(t)
    if t.ndim!=1 or len(t)<2:raise ValueError('At least two timestamps required')
    events=np.asarray(events);levels=np.asarray(levels)
    dt=float(t[1]-t[0])
    if dt<=0 or not np.allclose(np.diff(t),dt,rtol=0,atol=1e-10):raise ValueError('Uniform time grid required')
    receptor=np.array(initial_receptor,dtype=float);adapted=np.array(initial_adapted,dtype=float)
    response=np.zeros((len(t),6));clock=0.
    for i in range(1,len(t)):
        concentration=levels[np.clip(np.searchsorted(events,clock,side='right')-1,0,len(events)-1)]
        occupancy=concentration/(concentration+reference)
        receptor+=(-np.expm1(-dt/tau_r))*(occupancy-receptor)
        r=receptor-adapted
        adapted+=(-np.expm1(-dt/tau_a))*(receptor-adapted)
        response[i]=[r[0],r[0],-r[1],-r[1],r[2],-r[2]]
        clock+=dt
    return gain*np.maximum(response,0.)
