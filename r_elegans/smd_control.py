"""E082 imposed SMD membrane inhibition and release, never learned stopping."""
import numpy as np
from .hybrid_assessment import cycle_excursions,peak_frequency

SMD=('SMDDL','SMDDR','SMDVL','SMDVR')


def cases():
    rows=[dict(name='sham',targets=[],conductance_nS=0.)]
    for g in (.003,.01,.03,.1):rows.append(dict(name=f'SMD-{g:g}',targets=list(SMD),conductance_nS=g))
    for n in SMD:rows.append(dict(name=n,targets=[n],conductance_nS=.1))
    for label,names in [('dorsal',SMD[:2]),('ventral',SMD[2:])]:
        rows.append(dict(name=label,targets=list(names),conductance_nS=.1))
    return rows


def protocol(variant):
    if variant=='nominal':return 60.,[(30.,40.)],[(20.,30.),(50.,60.)]
    if variant=='shift':return 67.,[(37.,47.)],[(27.,37.),(57.,67.)]
    if variant=='repeat':return 120.,[(30.,40.),(80.,90.)],[(20.,30.),(55.,65.),(105.,115.)]
    if variant=='long-hold':return 150.,[(30.,90.)],[(20.,30.),(130.,150.)]
    raise ValueError('Unknown SMD protocol')


def membrane_probe(ids,case,variant):
    import jax.numpy as jnp
    from .brain.conductance_network import MembraneConductanceInput
    targets=case['targets'];g=float(case['conductance_nS'])
    if not set(targets)<=set(SMD) or not np.isfinite(g) or not 0<=g<=.1:raise ValueError('Outside SMD diagnostic envelope')
    mask=np.zeros(len(ids))
    for n in targets:mask[ids.index(n)]=g
    mask=jnp.asarray(mask);_,intervals,_=protocol(variant)
    def probe(t):
        active=jnp.asarray(False)
        for a,b in intervals:active=active|((t>=a)&(t<b))
        conductance=mask*active
        return MembraneConductanceInput(conductance,-80.*conductance)
    return probe


def compare(trace,sham,ids,case,variant='nominal'):
    t=np.asarray(trace['dense_time_s']);q=np.asarray(trace['dense_joints']);v=np.asarray(trace['dense_voltage_mV'])
    if not np.array_equal(t,sham['dense_time_s']):raise ValueError('Paired clocks differ')
    if not all(np.isfinite(x).all() for data in (trace,sham) for x in data.values()):raise ValueError('Nonfinite diagnostic state')
    _,intervals,active=protocol(variant);sm=[ids.index(n) for n in SMD]
    targets=[ids.index(n) for n in case['targets']]
    if not targets:raise ValueError('Compare inhibited case to sham, not sham to itself')
    before=t<intervals[0][0]
    prefix=float(max(np.max(abs(q[before]-sham['dense_joints'][before])),np.max(abs(v[before]-sham['dense_voltage_mV'][before]))))
    windows=[];preserved=True
    safe=all(abs(data['dense_joints']).max()<.5 and data['dense_voltage_mV'].min()>-100 and data['dense_voltage_mV'].max()<60 for data in (trace,sham))
    for a,b in active:
        use=(t>=a)&(t<b)
        shape=q[use];reference=sham['dense_joints'][use]
        if use.sum()<4:raise ValueError('Missing active window')
        amps=np.asarray(cycle_excursions(shape,late_only=False)['regional_cycle_excursion_rad'])
        base=np.asarray(cycle_excursions(reference,late_only=False)['regional_cycle_excursion_rad'])
        fs=[]
        for region in np.array_split(np.arange(q.shape[1]),3):
            j=int(region[np.argmax(np.ptp(shape[:,region],axis=0))])
            fs.append(peak_frequency(t[use],shape[:,j],max(1e-6,.1*np.ptp(shape[:,j]))))
        ok=bool(np.all(amps>=np.maximum(.003,.8*base)) and all(f is not None and 1.62<=f<=1.98 for f in fs))
        preserved=preserved and ok
        windows.append(dict(window=[a,b],regional_cycles_rad=amps.tolist(),frequencies_hz=fs,preserved=ok))
    off=[]
    for a,b in intervals:
        use=(t>=b-5)&(t<b)
        if use.sum()<4:raise ValueError('Missing off window')
        va=v[use];vb=sham['dense_voltage_mV'][use]
        std=np.std(va[:,targets],axis=0);std0=np.std(vb[:,targets],axis=0)
        rms=float(np.sqrt(np.mean(std**2)));rms0=float(np.sqrt(np.mean(std0**2)))
        ratio=rms/rms0 if rms0>1e-8 else None
        body=float(np.max(np.ptp(q[use],axis=0)));body0=float(np.max(np.ptp(sham['dense_joints'][use],axis=0)))
        shifts=np.mean(va[:,targets]-vb[:,targets],axis=0)
        access=bool(np.all(shifts<=-5.))
        off.append(dict(window=[b-5,b],body_range_ratio=body/max(body0,1e-12),raw_body_range_rad=body,
            target_voltage_rms_ratio=ratio,target_mean_shifts_mV=shifts.tolist(),target_std_mV=std.tolist(),
            sham_target_std_mV=std0.tolist(),smd_p2p_mV=np.ptp(va[:,sm],axis=0).tolist(),
            sham_smd_p2p_mV=np.ptp(vb[:,sm],axis=0).tolist(),inhibition_confirmed=access,
            muscle_range_ratio=float(np.max(np.ptp(trace['dense_muscles'][use],axis=0))/max(1e-12,np.max(np.ptp(sham['dense_muscles'][use],axis=0))))))
    suppressed=all(r['body_range_ratio']<=.2 and r['target_voltage_rms_ratio'] is not None and r['target_voltage_rms_ratio']<=.2 and r['inhibition_confirmed'] for r in off)
    return dict(pre_match=prefix<=1e-8,pre_difference=prefix,safe=safe,active_and_recovery_preserved=bool(preserved),
        active_windows=windows,off_windows=off,reversible_neural_target=bool(prefix<=1e-8 and safe and preserved and suppressed),
        strict_body_rest=all(r['raw_body_range_rad']<=1e-4 for r in off),learned_stopping_qualified=False,
        scope='Imposed neural inhibition; partial reversible control access only, not learned stopping or intrinsic oscillator proof')


def choose(comparisons):
    eligible=[r for r in comparisons if r['metrics']['pre_match'] and r['metrics']['safe'] and all(w['inhibition_confirmed'] for w in r['metrics']['off_windows'])]
    if not eligible:return None
    def rank(r):
        m=r['metrics'];ratio=max(w['body_range_ratio'] for w in m['off_windows'])
        return (not m['active_and_recovery_preserved'],ratio,r['case']['conductance_nS'])
    return min(eligible,key=rank)['case']
