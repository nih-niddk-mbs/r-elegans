"""E085 small sensory-triggered amplitude attenuation, not full stopping."""
import numpy as np
from .hybrid_assessment import cycle_excursions,peak_frequency

RUNGS=(.02,.05,.10)
CELLS=('AWAL','AWAR','AIYL','AIYR','AIZL','AIZR','AIBL','AIBR','RIAL','RIAR')


def protocol(variant):
    shift=5. if variant=='shift' else 0.
    dose=.8 if variant=='low-dose' else 1.
    events=[(40.+shift,50.+shift)]
    response=[(41.+shift,46.+shift)];active=[(30.+shift,40.+shift),(60.+shift,70.+shift)]
    duration=70.+shift;late=[(46.+shift,50.+shift)]
    if variant=='repeat':
        events.append((85.,95.));response.append((86.,91.));active.extend([(75.,85.),(105.,115.)]);duration=115.;late.append((91.,95.))
    elif variant=='redundant':
        events.append((53.,63.));response.append((54.,59.));active=[(30.,40.),(73.,83.)];duration=83.
    elif variant=='long-cue':
        events=[(40.,60.)];active=[(30.,40.),(75.,85.)];late=[(51.,56.)];duration=85.
    elif variant not in ('nominal','shift','low-dose'):raise ValueError('Unknown sensory protocol')
    return dict(duration=duration,events=events,response=response,active=active,late=late,dose=dose)


def world(variant,cued):
    import jax.numpy as jnp
    from .body.embodied_sensory import ChemicalBathSchedule,validate_chemical_world
    p=protocol(variant);times=sorted({0.,p['duration'],*[x for event in p['events'] for x in event]})
    concentrations=[[p['dose'] if cued and any(a<=t<b for a,b in p['events']) else 0.,0.,50000.] for t in times]
    result=ChemicalBathSchedule(jnp.asarray(times),jnp.asarray(concentrations));validate_chemical_world(result)
    return result


def parameter_groups(spec,ids,origin,lower,upper):
    ci=np.nonzero(spec.channels);ei=np.nonzero(spec.chemical);nc=len(ci[0]);groups=[];used=set()
    def add(name,indices):
        ix=np.asarray(indices,dtype=int)
        if not len(ix):return
        if used.intersection(ix.tolist()):raise ValueError('Overlapping parameters')
        lo=max(-.4,float(np.max(lower[ix]-origin[ix])));hi=min(.4,float(np.min(upper[ix]-origin[ix])))
        if lo>1e-10 or hi<-1e-10:raise ValueError('Parent outside anchor bounds')
        if hi-lo<1e-8:return
        groups.append(dict(name=name,indices=ix.tolist(),lower=min(0.,lo),upper=max(0.,hi)));used.update(ix.tolist())
    for n in CELLS:
        cell=ids.index(n)
        for label,columns in [('inward',[6,7,8,14]),('outward',[0,1,2,3,4,5,9,10,11,12,13,15])]:
            add('density:'+n+':'+label,np.flatnonzero((ci[0]==cell)&np.isin(ci[1],columns)))
    relays=[ids.index(n) for n in CELLS if n not in ('AWAL','AWAR','RIAL')]
    for n in CELLS:
        pre=ids.index(n)
        add('chemical:'+n+'->relays',nc+np.flatnonzero((ei[1]==pre)&np.isin(ei[0],relays)))
        if n!='RIAL':add('chemical:'+n+'->RIAL',nc+np.flatnonzero((ei[1]==pre)&(ei[0]==ids.index('RIAL'))))
    return groups


def window_features(trace,ids,a,b,frequency_estimator=peak_frequency):
    t=trace['dense_time_s'];use=(t>=a)&(t<b);q=trace['dense_joints'][use]
    if use.sum()<40:raise ValueError('Missing amplitude window')
    cy=cycle_excursions(q,late_only=False);joint=np.asarray(cy['joint_cycle_excursion_rad']);freq=[]
    for region in np.array_split(np.arange(11),3):
        j=int(region[np.argmax(joint[region])]);freq.append(frequency_estimator(t[use],q[:,j],max(1e-6,.1*joint[j])))
    v=trace['dense_voltage_mV'][use];ri=ids.index('RIAL')
    return dict(window=[a,b],amplitude=float(np.sqrt(np.mean(joint**2))),regional_cycles_rad=cy['regional_cycle_excursion_rad'],
        frequency_hz=freq,RIAL_mean_mV=float(v[:,ri].mean()),RIAL_std_mV=float(v[:,ri].std()))


def features(trace,ids,variant,frequency_estimator=peak_frequency):
    p=protocol(variant)
    if not all(np.isfinite(x).all() for x in trace.values()):raise ValueError('Nonfinite rollout')
    q=trace['dense_joints'];v=trace['dense_voltage_mV']
    return {**{key:[window_features(trace,ids,a,b,frequency_estimator) for a,b in p[key]] for key in ('active','response','late')},
        'safe':bool(abs(q).max()<.5 and v.min()>-100 and v.max()<60)}


def score(paired,parent,target,margin=1.):
    if not 0<target<=.1 or not 1<=margin<=1.5:raise ValueError('Small partial-attenuation task only')
    a=paired['no_cue'];b=paired['cue'];base=np.asarray(parent['active'][0]['regional_cycles_rad'])
    penalty=0.;baseline=True;useful=True
    def preserve(row,reference,frequencies):
        nonlocal penalty
        amp=np.asarray(row['regional_cycles_rad']);ref=np.asarray(reference)
        low=np.maximum(.003,.9*ref);high=1.1*ref
        penalty+=float(np.sum(np.maximum(0.,np.log(low/np.maximum(amp,1e-9)))**2)+np.sum(np.maximum(0.,np.log(np.maximum(amp,1e-9)/high))**2))
        good=bool(np.all(amp>=low)&np.all(amp<=high))
        for f,f0 in zip(row['frequency_hz'],frequencies):
            penalty+=10. if f is None else (max(0.,abs(f-1.8)-.18)/.18)**2
            good=good and f is not None and 1.62<=f<=1.98
            if f is None or f0 is None or min(f,f0)<=0:penalty+=10.;good=False
            else:
                penalty+=(max(0.,abs(np.log(f/f0))-np.log(1.05))/.05)**2
                good=good and abs(np.log(f/f0))<=np.log(1.05)
        return good
    for row in a['active']+a['response']+a['late']:baseline=preserve(row,base,parent['active'][0]['frequency_hz']) and baseline
    for row,reference in zip(b['active'],a['active']):baseline=preserve(row,reference['regional_cycles_rad'],reference['frequency_hz']) and baseline
    ratios=[];error=[]
    for row,reference in zip(b['response'],a['response']):
        ratio=row['amplitude']/max(reference['amplitude'],1e-9);ratios.append(float(ratio))
        error.append(((ratio-(1-target*margin))/target)**2)
        amps=np.asarray(row['regional_cycles_rad']);ref=np.asarray(reference['regional_cycles_rad'])
        penalty+=float(np.sum(np.maximum(0.,np.log(.5*ref/np.maximum(amps,1e-9)))**2)+np.sum(np.maximum(0.,np.log(np.maximum(amps,1e-9)/(1.1*ref)))**2))
        useful=useful and bool(np.all(amps>=.5*ref)&np.all(amps<=1.1*ref)) and all(f is not None and 1.62<=f<=1.98 for f in row['frequency_hz'])
        for f,f0 in zip(row['frequency_hz'],reference['frequency_hz']):
            if f is None or f0 is None or min(f,f0)<=0:penalty+=10.;useful=False
            else:
                penalty+=(max(0.,abs(np.log(f/f0))-np.log(1.05))/.05)**2
                useful=useful and abs(np.log(f/f0))<=np.log(1.05)
    safe=a['safe'] and b['safe']
    loss=float(np.mean(error)+100*penalty+1000*(not safe)+10*(not useful))
    return dict(loss=loss,response_amplitude_ratios=ratios,late_amplitude_ratios=[x['amplitude']/max(y['amplitude'],1e-9) for x,y in zip(b['late'],a['late'])],
        safe=bool(safe),baseline_and_recovery_preserved=bool(baseline),useful_response_preserved=bool(useful),
        partial_attenuation_passed=bool(safe and baseline and useful and all(r<=1-target for r in ratios)),
        training_margin_met=bool(safe and baseline and useful and all(r<=1-target*margin for r in ratios)),
        target=target,margin=margin,full_start_stop_qualified=False,swimming_qualified=False)


def replay_probe(ids,chemical_mask,times,release,onset,dt):
    """Final diagnostic only: replay matched no-cue RIAL release on its SMD edges."""
    import jax.numpy as jnp
    from .brain.conductance_network import ChemicalReleaseOverride
    source=ids.index('RIAL');posts=[i for i,n in enumerate(ids) if n.startswith('SMD') and chemical_mask[i,source]>0]
    times=np.asarray(times);release=np.asarray(release)
    if not posts or times.shape!=release.shape or not np.isfinite(times).all() or np.any(np.diff(times)<=0) or not np.isfinite(release).all() or np.any((release<0)|(release>1)):
        raise ValueError('Valid matched no-cue release and existing branch required')
    post=jnp.asarray(posts,dtype=jnp.int32);pre=jnp.full(len(posts),source,dtype=jnp.int32)
    clock=jnp.asarray(times);values=jnp.asarray(release)
    def probe(t):
        probability=jnp.interp(t+.5*dt,clock,values)
        return ChemicalReleaseOverride(post,pre,jnp.full(len(posts),probability),jnp.asarray(t>=onset))
    return probe
