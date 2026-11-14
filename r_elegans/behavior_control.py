"""Sensory-controlled state transitions and checkpoint-specific behavior gates.

Cue associations and numerical thresholds are engineering tasks, not innate
diacetyl/salt reflexes or evidence of ancestral behavior.
"""
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks

from .hybrid_assessment import cycle_excursions,peak_frequency

CONTROL_GROUPS={
    'sensory_integrators':('AIYL','AIYR','AIBL','AIBR','AIZL','AIZR','RIAL','RIAR'),
    'locomotor_integrators':('RIBL','RIBR','RIML','RIMR'),
    'forward_premotor':('AVBL','AVBR','PVCL','PVCR'),
    'stop_candidate':('RIS',),
}
SENSORY_CELLS=('AWAL','AWAR','ASEL','ASER')


@dataclass(frozen=True)
class ControlProtocol:
    shift: float=0.
    rest_extension: float=0.
    repeated: bool=False
    redundant: bool=False
    dose: float=1.
    sham: bool=False
    omit_start: bool=False
    omit_stop: bool=False

    def __post_init__(self):
        if (not np.isfinite([self.shift,self.rest_extension,self.dose]).all() or self.shift<0
                or self.rest_extension<0 or self.dose<=0 or (self.repeated and self.rest_extension)):
            raise ValueError('Finite nonnegative timings, positive dose; repeated and extended-rest fixtures are separate')

    @property
    def duration(self):return (140. if self.repeated else 90.)+self.shift+self.rest_extension

    def events(self):
        events=[(40.+self.shift,42.+self.shift,'stop'),
                (58.+self.shift+self.rest_extension,60.+self.shift+self.rest_extension,'start')]
        if self.repeated:
            events.extend([(90.+self.shift,92.+self.shift,'stop'),(108.+self.shift,110.+self.shift,'start')])
        if self.redundant:
            events.extend([(32.+self.shift,34.+self.shift,'start'),(52.+self.shift,54.+self.shift,'stop'),
                           (74.+self.shift,76.+self.shift,'start')])
        return events

    def windows(self):
        active=[(30.+self.shift,40.+self.shift),
                (68.+self.shift+self.rest_extension,78.+self.shift+self.rest_extension),
                (78.+self.shift+self.rest_extension,88.+self.shift+self.rest_extension)]
        rest=[(48.+self.shift,58.+self.shift+self.rest_extension)]
        if self.repeated:
            active.extend([(118.+self.shift,128.+self.shift),(128.+self.shift,138.+self.shift)])
            rest.append((98.+self.shift,108.+self.shift))
        return active,rest


def bath_schedule(protocol):
    """Only actual ligand levels are visible to the sensory/brain model."""
    import jax.numpy as jnp
    from .body.embodied_sensory import ChemicalBathSchedule,validate_chemical_world
    events=protocol.events();times=sorted({0.,protocol.duration,*[x for a,b,_ in events for x in (a,b)]})
    concentration=[]
    for t in times:
        c=np.array([0.,0.,50000.])
        if not protocol.sham:
            for a,b,kind in events:
                if a<=t<b:
                    if kind=='start' and not protocol.omit_start:c[0]+=protocol.dose
                    if kind=='stop' and not protocol.omit_stop:c[2]+=10000.*protocol.dose
        concentration.append(c)
    world=ChemicalBathSchedule(jnp.asarray(times),jnp.asarray(np.asarray(concentration)))
    validate_chemical_world(world)
    return world


def forward_cycles(t,q,position,heading,body_length=.001):
    """Equal-phase displacement with whole-shape closure; descriptive physics."""
    q=np.asarray(q);t=np.asarray(t);p=np.asarray(position);h=np.asarray(heading)
    cyclic=cycle_excursions(q,late_only=False)['joint_cycle_excursion_rad']
    ref=int(np.argmax(cyclic));signal=q[:,ref]-q[:,ref].mean()
    indices=np.flatnonzero((signal[:-1]<=0)&(signal[1:]>0))
    if len(indices)<4:return dict(passed=False,speed_BL_s=None,cycles=0,maximum_relative_shape_closure=None)
    alpha=-signal[indices]/(signal[indices+1]-signal[indices]);clock=t[indices]+alpha*(t[indices+1]-t[indices])
    def interp(x):
        if x.ndim==1:return x[indices]+alpha*(x[indices+1]-x[indices])
        return x[indices]+alpha[:,None]*(x[indices+1]-x[indices])
    shape=interp(q);poses=interp(p);angles=interp(h)
    orientation=angles[:,None]+np.column_stack((np.zeros(len(shape)),np.cumsum(shape,axis=1)))
    axes=np.column_stack((np.cos(orientation).mean(axis=1),np.sin(orientation).mean(axis=1)))
    axes/=np.maximum(np.linalg.norm(axes,axis=1,keepdims=True),1e-12)
    direction=axes[:-1]+axes[1:];direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-12)
    speeds=np.sum(np.diff(poses,axis=0)*direction,axis=1)/np.diff(clock)/body_length
    scale=max(float(np.sqrt(np.mean(np.ptp(q,axis=0)**2))),1e-6)
    closure=np.sqrt(np.mean(np.diff(shape,axis=0)**2,axis=1))/scale
    good=closure<=.02
    speed=float(np.median(speeds[good])) if np.sum(good)>=3 else None
    centered=q-q.mean(axis=0)
    _,singular,basis=np.linalg.svd(centered,full_matrices=False);pc=centered@basis[:2].T
    loop=[]
    for a,b in zip(indices[:-1],indices[1:]):
        polygon=pc[a:b+1]
        area=.5*abs(float(np.sum(polygon[:,0]*np.roll(polygon[:,1],-1)-polygon[:,1]*np.roll(polygon[:,0],-1))))
        loop.append(area/max(scale**2,1e-12))
    area=float(np.median(np.asarray(loop)[good])) if np.any(good) else 0.
    return dict(passed=bool(speed is not None and speed>=.01 and np.mean(good)>=.8 and area>=1e-3),
                speed_BL_s=speed,cycles=int(len(speeds)),closed_cycles=int(np.sum(good)),
                maximum_relative_shape_closure=float(np.max(closure)),normalized_shape_loop_area=area,
                pca_second_variance_fraction=float(singular[1]**2/max(float(np.sum(singular**2)),1e-30)))


def metrics(trace,protocol,baseline=None):
    t=np.asarray(trace['dense_time_s']);q=np.asarray(trace['dense_joints'])
    if q.shape!=(len(t),11) or not np.isfinite(q).all() or t[-1]<protocol.duration-1e-9:
        raise ValueError('Complete finite state-control trajectory required')
    active,rest=protocol.windows();on=[];off=[]
    for a,b in active:
        use=(t>=a-1e-10)&(t<b-1e-10)
        clock=t[use];shape=q[use];cycles=cycle_excursions(shape,late_only=False)
        per_joint=np.asarray(cycles['joint_cycle_excursion_rad']);frequencies=[];gaps=[]
        for region in np.array_split(np.arange(11),3):
            fs=[]
            for j in region:
                prominence=max(1e-6,.1*per_joint[j])
                if (per_joint[j]>=1e-6 and len(find_peaks(shape[:,j],prominence=prominence)[0])>=3
                        and len(find_peaks(-shape[:,j],prominence=prominence)[0])>=3):
                    f=peak_frequency(clock,shape[:,j],prominence)
                    if f is not None:fs.append(f)
            frequencies.append(float(np.median(fs)) if fs else None)
            if fs:
                j=int(region[np.argmax(per_joint[region])])
                peaks=find_peaks(shape[:,j],prominence=max(1e-6,.1*per_joint[j]))[0]
                gaps.append(float(np.max(np.diff(np.r_[clock[0],clock[peaks],clock[-1]]))) if len(peaks)>1 else float(b-a))
            else:gaps.append(float(b-a))
        amplitude=np.asarray(cycles['regional_cycle_excursion_rad'])
        floor=np.full(3,.003) if baseline is None else np.maximum(.003,.8*np.asarray(baseline['on'][0]['amplitudes']))
        rhythmic=all(f is not None and 1.62<=f<=1.98 for f in frequencies) and max(gaps)<=1.
        propulsion=forward_cycles(clock,shape,trace['dense_position'][use],trace['dense_heading'][use])
        on.append(dict(window=[a,b],amplitudes=amplitude.tolist(),frequency_hz=frequencies,
                       amplitude_floor=floor.tolist(),passed=bool(np.all(amplitude>=floor) and rhythmic),
                       maximum_interpeak_gap_s=gaps,maximum_raw_joint_range_rad=float(np.max(np.ptp(shape,axis=0))),propulsion=propulsion))
    transitions=[]
    for a,b in rest:
        use=(t>=a-1e-10)&(t<b-1e-10);span=np.ptp(q[use],axis=0)
        off.append(dict(window=[a,b],maximum_raw_joint_range_rad=float(np.max(span)),passed=bool(np.max(span)<=1e-4)))
        stop_onset=a-8.;segment=(t>=stop_onset)&(t<b)
        values=q[segment];clock=t[segment]
        suffix_max=np.maximum.accumulate(values[::-1],axis=0)[::-1]
        suffix_min=np.minimum.accumulate(values[::-1],axis=0)[::-1]
        quiet=(np.max(suffix_max-suffix_min,axis=1)<=1e-4)&(clock<=b-1.)
        stop_latency=float(clock[np.flatnonzero(quiet)[0]]-stop_onset) if np.any(quiet) else None
        after=(t>=b)&(t<=min(protocol.duration,b+30.))
        movement=np.max(np.abs(q[after]-q[np.flatnonzero(t>=b)[0]]),axis=1)>1e-4
        start_latency=float(t[after][np.flatnonzero(movement)[0]]-b) if np.any(movement) else None
        transitions.append(dict(stop_onset_s=stop_onset,settled_stop_latency_s=stop_latency,
                                start_onset_s=b,first_detectable_movement_latency_s=start_latency))
    amplitude_loss=max(float(np.mean(np.maximum(0,np.log(np.asarray(r['amplitude_floor'])/np.maximum(r['amplitudes'],1e-7))))) for r in on)
    frequency_loss=max(float(np.mean([abs(np.log(f/1.8)) if f is not None else 10. for f in r['frequency_hz']])) for r in on)
    rest_loss=max(float(np.log1p(r['maximum_raw_joint_range_rad']/1e-4)) for r in off)
    safety=bool(np.max(np.abs(q))<.5)
    control=bool(safety and all(r['passed'] for r in on) and all(r['passed'] for r in off))
    velocities=[r['propulsion']['speed_BL_s'] for r in on]
    swim_loss=float(np.mean([max(0.,(.01-v)/.01) if v is not None else 10. for v in velocities]))
    swim_loss+=float(np.mean([max(0.,1-r['propulsion'].get('normalized_shape_loop_area',0.)/1e-3) for r in on]))
    return dict(control_passed=control,swimming_passed=bool(control and all(r['propulsion']['passed'] for r in on)),
                total=amplitude_loss+frequency_loss+rest_loss+10*(not safety),swim_loss=swim_loss,
                stop_loss=rest_loss,restart_loss=amplitude_loss+frequency_loss+rest_loss,
                on=on,off=off,transition_latencies=transitions,safety_passed=safety,initial_movement_passed=on[0]['passed'])


def qualify_behavior(checkpoint_sha256,motion_passed,control_report):
    """A parent's control evidence never certifies changed behavior weights."""
    required=('passed','complete','positive_cases_passed','negative_controls_passed','sensory_specificity_passed','refinement_passed')
    return bool(motion_passed and all(control_report.get(key) is True for key in required)
                and control_report.get('causal_control_groups')
                and control_report.get('checkpoint_sha256')==checkpoint_sha256)


def case_protocol(name):
    if name=='repeat':return ControlProtocol(repeated=True)
    if name=='long-rest':return ControlProtocol(rest_extension=28.)
    if name=='onset-shift':return ControlProtocol(shift=3.)
    if name=='dose-low':return ControlProtocol(dose=.8)
    if name=='dose-high':return ControlProtocol(dose=1.2)
    if name=='redundant':return ControlProtocol(redundant=True)
    if name=='no-cue':return ControlProtocol(sham=True)
    if name=='start-blocked':return ControlProtocol(omit_start=True)
    if name=='stop-blocked':return ControlProtocol(omit_stop=True)
    return ControlProtocol()


POSITIVE_CASES=('nominal','repeat','long-rest','onset-shift','dose-low','dose-high','redundant','neural-half','body-half')
VALIDATION_CASES=POSITIVE_CASES+('chemical-blocked','no-cue','start-blocked','stop-blocked','held-voltage')+tuple('controller-'+name for name in CONTROL_GROUPS)


def control_report(checkpoint_sha256,cases,refinement_passed):
    """Enforce cue specificity, idempotence and a non-paralyzing neural intervention."""
    rows={r['case']:r for r in cases}
    complete=(set(rows)==set(VALIDATION_CASES) and len(cases)==len(VALIDATION_CASES)
              and all(r.get('checkpoint_sha256')==checkpoint_sha256 for r in cases))
    positives=complete and all(rows[n]['metrics']['control_passed'] for n in POSITIVE_CASES)
    negatives=complete and all(not rows[n]['metrics']['control_passed'] for n in ('chemical-blocked','no-cue','start-blocked','stop-blocked','held-voltage'))
    cue_specific=complete and all(rows[n]['metrics']['initial_movement_passed'] and
        rows[n].get('pre_cue_maximum_difference_rad',1.)<=1e-7 and
        all(r['maximum_raw_joint_range_rad']>=.003 for r in rows[n]['metrics']['off']) for n in ('chemical-blocked','no-cue','stop-blocked'))
    if complete:
        start=rows['start-blocked']
        cue_specific=cue_specific and start['metrics']['initial_movement_passed'] and start.get('pre_start_maximum_difference_rad',1.)<=1e-7 and all(r['passed'] for r in start['metrics']['off']) and all(r['maximum_raw_joint_range_rad']<=1e-4 for r in start['metrics']['on'][1:])
    causal=[]
    # An unsuccessful baseline cannot establish loss of learned control.
    if complete and positives and cue_specific:
        for group in CONTROL_GROUPS:
            r=rows['controller-'+group]['metrics']
            if r['initial_movement_passed'] and not r['control_passed']:causal.append(group)
    return dict(checkpoint_sha256=checkpoint_sha256,complete=complete,positive_cases_passed=bool(positives),
                negative_controls_passed=bool(negatives),sensory_specificity_passed=bool(cue_specific),
                causal_control_groups=causal,refinement_passed=bool(refinement_passed),
                passed=bool(complete and positives and negatives and cue_specific and causal and refinement_passed))
