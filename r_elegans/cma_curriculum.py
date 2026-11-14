"""Grouped bounded parameters and paired sensory cadence curriculum, not RL."""
import numpy as np
from .behavior_control import CONTROL_GROUPS,ControlProtocol
from .hybrid_assessment import cycle_excursions,peak_frequency
from scipy.signal import find_peaks

RUNGS=(.05,.10,.20,.35,.50)
CONDITIONS=('no_cue','stop_only','start_only','stop_start')


def condition_protocol(condition,variant='nominal'):
    if condition not in CONDITIONS:raise ValueError('Unknown paired condition')
    kwargs={'shift':3.,'dose':.8} if variant=='low_shift' else {'repeated':True} if variant=='repeat' else {}
    if variant not in ('nominal','low_shift','repeat'):raise ValueError('Unknown validation variant')
    return ControlProtocol(**kwargs,sham=condition=='no_cue',omit_start=condition=='stop_only',omit_stop=condition=='start_only')


def parameter_groups(spec,ids,origin,lower,upper):
    """Disjoint log multipliers; every member remains inside original bounds."""
    ci=np.nonzero(spec.channels);ei=np.nonzero(spec.chemical);gi=np.nonzero(np.triu(spec.gap,1))
    nc=len(ci[0]);ne=len(ei[0]);size=len(origin);groups=[];occupied=np.zeros(size,dtype=bool)
    names={'ASE':('ASEL','ASER'),'AWA':('AWAL','AWAR'),
           'integrators':CONTROL_GROUPS['sensory_integrators'],'RIB':('RIBL','RIBR'),
           'RIM':('RIML','RIMR'),'RIS':('RIS',),'forward':CONTROL_GROUPS['forward_premotor'],
           'motor':tuple(n for n in ids if n.startswith(('DA','DB','DD','VA','VB','VD','AS','RMD','SMD','SMB')) and n not in ('ASEL','ASER'))}
    sets={k:[ids.index(n) for n in v] for k,v in names.items()}
    def add(name,indices):
        indices=np.asarray(indices,dtype=int)
        if not len(indices):return
        if occupied[indices].any():raise ValueError('Overlapping grouped coordinates')
        lo=max(-.4,float(np.max(lower[indices]-origin[indices])))
        hi=min(.4,float(np.min(upper[indices]-origin[indices])))
        if lo>1e-10 or hi<-1e-10:raise ValueError('Parent outside anchor bounds')
        lo=min(lo,0.);hi=max(hi,0.)
        if hi-lo<1e-8:return
        occupied[indices]=True
        groups.append(dict(name=name,indices=indices.tolist(),lower=lo,upper=hi))
    for src,dst in [('ASE','integrators'),('AWA','integrators'),('integrators','RIB'),('integrators','RIM'),('integrators','RIS'),
                    ('RIB','RIS'),('RIS','RIB'),('RIM','RIS'),('RIS','RIM'),('RIB','RIM'),('RIM','RIB'),
                    ('RIB','motor'),('RIM','motor'),('RIS','motor'),('integrators','motor')]:
        add('chemical:'+src+'->'+dst,nc+np.flatnonzero(np.isin(ei[1],sets[src])&np.isin(ei[0],sets[dst])))
    for a,b in [('ASE','integrators'),('RIS','RIM'),('RIB','forward')]:
        mask=(np.isin(gi[0],sets[a])&np.isin(gi[1],sets[b]))|(np.isin(gi[1],sets[a])&np.isin(gi[0],sets[b]))
        add('gap:'+a+'--'+b,nc+ne+np.flatnonzero(mask))
    for group in ('integrators','RIB','RIM','RIS'):
        for channel_type,columns in [('inward',[6,7,8,14]),('outward',[0,1,2,3,4,5,9,10,11,12,13,15])]:
            add('density:'+group+':'+channel_type,np.flatnonzero(np.isin(ci[0],sets[group])&np.isin(ci[1],columns)))
    return groups


def decode_groups(z,groups,origin,lower,upper):
    z=np.asarray(z,dtype=float)
    if z.shape!=(len(groups),) or not np.isfinite(z).all() or np.any(abs(z)>1+1e-12):raise ValueError('Invalid grouped latent point')
    x=np.array(origin,dtype=float,copy=True)
    for value,group in zip(z,groups):
        delta=value*(group['upper'] if value>=0 else -group['lower'])
        x[group['indices']]+=delta
    if np.any(x<lower-1e-10) or np.any(x>upper+1e-10):raise ValueError('Grouped candidate exceeds original bounds')
    return x


def features(trace,protocol):
    t=np.asarray(trace['dense_time_s']);q=np.asarray(trace['dense_joints'])
    on,off=protocol.windows();result={}
    for label,windows in [('on',on),('off',off)]:
        rows=[]
        for a,b in windows:
            use=(t>=a)&(t<b);clock=t[use];shape=q[use]
            if len(clock)<4 or not np.isfinite(shape).all():raise ValueError('Invalid cadence window')
            cycles=cycle_excursions(shape,late_only=False);per=np.asarray(cycles['joint_cycle_excursion_rad']);fs=[];gaps=[]
            for region in np.array_split(np.arange(shape.shape[1]),3):
                frequencies=[];largest=[]
                for j in region:
                    prominence=max(1e-6,.1*per[j]);peaks=find_peaks(shape[:,j],prominence=prominence)[0]
                    troughs=find_peaks(-shape[:,j],prominence=prominence)[0]
                    if per[j]<1e-5 or len(peaks)<3 or len(troughs)<3:continue
                    f=peak_frequency(clock,shape[:,j],prominence)
                    if f is not None:
                        frequencies.append(f);largest.append(float(np.max(np.diff(np.r_[clock[0],clock[peaks],clock[-1]]))))
                fs.append(float(np.median(frequencies)) if frequencies else None);gaps.append(max(largest) if largest else float(b-a))
            rows.append(dict(window=[a,b],frequency_hz=fs,regional_cycles_rad=list(map(float,cycles['regional_cycle_excursion_rad'])),
                             maximum_peak_gap_s=gaps,maximum_raw_range_rad=float(np.max(np.ptp(shape,axis=0)))))
        result[label]=rows
    result['safe']=bool(np.isfinite(q).all() and np.max(abs(q))<.5 and np.isfinite(trace['dense_voltage_mV']).all() and np.min(trace['dense_voltage_mV'])>-100 and np.max(trace['dense_voltage_mV'])<60)
    return result


def score(paired,parent,target):
    """Slower cadence with retained bends; compare STOP/START to matched controls."""
    if set(paired)!=set(CONDITIONS):raise ValueError('Four paired conditions required')
    if not 0<target<=.5:raise ValueError('Partial-cadence curriculum only')
    nc=paired['no_cue'];ss=paired['stop_start'];so=paired['stop_only'];sa=paired['start_only']
    no_cue_ok=True;violations=[];errors=[];ratios=[];amplitude_ok=True;cadence_ok=True
    for label in ('on','off'):
        for k,row in enumerate(nc[label]):
            base_a=np.asarray(parent['on'][0]['regional_cycles_rad']);a=np.asarray(row['regional_cycles_rad'])
            good=all(f is not None and f0 is not None and 1.62<=f<=1.98 and .95<=f/f0<=1.05 for f,f0 in zip(row['frequency_hz'],parent['on'][0]['frequency_hz'])) and np.all(a>=np.maximum(.003,.9*base_a)) and np.all(a<=1.1*base_a) and max(row['maximum_peak_gap_s'])<=1.
            no_cue_ok=no_cue_ok and good
            # Continuous violations guide search even when outside the feasible set.
            violations.append(float(np.mean(np.maximum(0.,np.log(np.maximum(.003,.9*base_a)/np.maximum(a,1e-7))))))
            violations.append(float(np.mean(np.maximum(0.,np.log(np.maximum(a,1e-7)/(1.1*base_a))))))
            violations.append(float(np.mean([max(0.,abs(np.log(f/f0))-.048) if f is not None and f>0 and f0 is not None and f0>0 else 10. for f,f0 in zip(row['frequency_hz'],parent['on'][0]['frequency_hz'])])))
    # On windows after START must stay slow without START and recover with it.
    for condition,label,indices,wanted in [('stop_start','off',range(len(ss['off'])),1-target),
        ('stop_only','off',range(len(so['off'])),1-target),('stop_only','on',range(1,len(so['on'])),1-target),
        ('stop_start','on',range(1,len(ss['on'])),1.),('start_only','on',range(len(sa['on'])),1.),
        ('start_only','off',range(len(sa['off'])),1.)]:
        for k in indices:
            row=paired[condition][label][k];reference=nc[label][k]
            row_ratios=[]
            for f,f0 in zip(row['frequency_hz'],reference['frequency_hz']):
                ratio=f/f0 if f is not None and f0 is not None and f0>0 else None
                row_ratios.append(ratio);errors.append(((ratio-wanted)/.03)**2 if ratio is not None else 1e4)
                tolerance=min(.025,.3*target) if wanted<1 else .03
                cadence_ok=cadence_ok and ratio is not None and abs(ratio-wanted)<=tolerance
            floor=(.8 if wanted<1 else .9)*np.asarray(reference['regional_cycles_rad'])
            a=np.asarray(row['regional_cycles_rad']);amplitude_ok=amplitude_ok and np.all(a>=np.maximum(1e-4,floor))
            amplitude_error=float(np.mean(np.maximum(0.,np.log(np.maximum(1e-4,floor)/np.maximum(a,1e-7)))**2))
            errors.append(amplitude_error);violations.append(amplitude_error)
            cadence_ok=cadence_ok and max(row['maximum_peak_gap_s'])<=1.5
            ratios.append(dict(condition=condition,window=label,index=k,target_ratio=wanted,frequency_ratios=row_ratios))
    safe=all(v['safe'] for v in paired.values())
    # Pre-cue movement may not be sacrificed to score a conditional change.
    pre_ok=all(np.all(np.asarray(v['on'][0]['regional_cycles_rad'])>=np.maximum(.003,.9*np.asarray(parent['on'][0]['regional_cycles_rad']))) for v in paired.values())
    data=float(np.mean(errors));constraint=float(sum(violations)+10*(not safe)+10*(not pre_ok))
    passed=bool(safe and pre_ok and no_cue_ok and amplitude_ok and cadence_ok)
    return dict(loss=data+100*constraint,data_loss=data,constraint_loss=constraint,target_fraction=target,
        partial_cadence_passed=passed,safe=safe,uncued_movement_preserved=bool(no_cue_ok),pre_movement_preserved=bool(pre_ok),
        bend_amplitude_preserved=bool(amplitude_ok),cadence_ratios=ratios,full_stopping_qualified=False,swimming_qualified=False)
