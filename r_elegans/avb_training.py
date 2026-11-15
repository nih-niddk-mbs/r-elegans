"""E078: conditional forward rhythm under selective AVB inhibition.

This is intervention-based circuit fitting, not learned sensory stopping.
"""
import numpy as np
from scipy.signal import find_peaks
from .cma_curriculum import parameter_groups as control_groups
from .hybrid_assessment import cycle_excursions


def protocol(variant='ordinary'):
    if variant == 'repeat':
        return 120., [(30., 45.), (75., 90.)], [(20., 30.), (60., 70.), (105., 115.)]
    shift = 3. if variant == 'shift-weak' else 0.
    if variant not in ('ordinary', 'shift-weak'):
        raise ValueError('Unknown AVB protocol')
    return 60.+shift, [(30.+shift, 40.+shift)], [(20.+shift, 30.+shift), (50.+shift, 60.+shift)]


def membrane_probe(ids, variant='ordinary'):
    import jax.numpy as jnp
    from .brain.conductance_network import MembraneConductanceInput
    _, intervals, _ = protocol(variant)
    mask = np.zeros(len(ids))
    for name in ('AVBL', 'AVBR'):
        mask[ids.index(name)] = .08 if variant == 'shift-weak' else .1
    mask = jnp.asarray(mask)
    def probe(t):
        active = jnp.asarray(False)
        for a, b in intervals:
            active = active | ((t >= a) & (t < b))
        g = mask * active
        return MembraneConductanceInput(g, -80. * g)
    return probe


def parameter_groups(spec, ids, origin, lower, upper):
    groups = control_groups(spec, ids, origin, lower, upper)
    used = {i for g in groups for i in g['indices']}
    ci = np.nonzero(spec.channels)
    ei = np.nonzero(spec.chemical)
    gi = np.nonzero(np.triu(spec.gap, 1))
    nc, ne = len(ci[0]), len(ei[0])
    avb = [ids.index(n) for n in ('AVBL', 'AVBR')]
    motor = [i for i,n in enumerate(ids) if n.startswith(('DA','DB','DD','VA','VB','VD','AS','RMD','SMD','SMB')) and n not in ('ASEL','ASER')]
    def add(name, indices):
        indices = np.asarray([int(i) for i in indices if int(i) not in used], dtype=int)
        if not len(indices):
            return
        lo = max(-.4, float(np.max(lower[indices]-origin[indices])))
        hi = min(.4, float(np.min(upper[indices]-origin[indices])))
        if lo > 1e-10 or hi < -1e-10:
            raise ValueError('Parent outside original bounds')
        if hi-lo < 1e-8:
            return
        groups.append(dict(name=name, indices=indices.tolist(), lower=min(0.,lo), upper=max(0.,hi)))
        used.update(indices)
    for name, cells in [('AVB',avb)] + [(prefix,[i for i in motor if ids[i].startswith(prefix)])
                                         for prefix in ('DA','DB','DD','VA','VB','VD','AS','RMD','SMD','SMB')]:
        for kind, columns in [('inward',[6,7,8,14]), ('outward',[0,1,2,3,4,5,9,10,11,12,13,15])]:
            add('density:'+name+':'+kind, np.flatnonzero(np.isin(ci[0],cells)&np.isin(ci[1],columns)))
    for src,dst,name in [(avb,motor,'AVB->motor'),(motor,avb,'motor->AVB'),(motor,motor,'motor->motor')]:
        add('chemical:'+name,nc+np.flatnonzero(np.isin(ei[1],src)&np.isin(ei[0],dst)))
    for cells,name in [(avb,'AVB'),(motor,'motor')]:
        mask = ((np.isin(gi[0],cells)&np.isin(gi[1],motor)) |
                (np.isin(gi[1],cells)&np.isin(gi[0],motor)))
        add('gap:'+name+'--motor',nc+ne+np.flatnonzero(mask))
    return groups


def window_features(trace, a, b, motor, avb):
    t=np.asarray(trace['dense_time_s']);use=(t>=a)&(t<b)
    if use.sum()<4:
        raise ValueError('Missing AVB assay window')
    q=np.asarray(trace['dense_joints'])[use]
    v=np.asarray(trace['dense_voltage_mV'])[use]
    clock=t[use]
    cycles=cycle_excursions(q,late_only=False)['regional_cycle_excursion_rad']
    frequencies=[]
    for region in np.array_split(np.arange(q.shape[1]),3):
        fs=[]
        for j in region:
            peaks=find_peaks(q[:,j],prominence=max(1e-6,.1*np.ptp(q[:,j])))[0]
            if len(peaks)>=3:
                fs.append(float(1./np.mean(np.diff(clock[peaks]))))
        frequencies.append(float(np.median(fs)) if fs else None)
    return dict(regional_cycles_rad=list(map(float,cycles)),frequency_hz=frequencies,
                raw_range_rad=float(np.max(np.ptp(q,axis=0))),
                motor_rms_mV=float(np.sqrt(np.mean((v[:,motor]-v[:,motor].mean(axis=0))**2))),
                avb_mean_mV=float(v[:,avb].mean()))


def features(trace, ids, variant='ordinary'):
    _, intervals, active = protocol(variant)
    # The last 5 seconds of each inhibition interval allow the onset transient.
    avb=[ids.index(n) for n in ('AVBL','AVBR')]
    motor=[i for i,n in enumerate(ids) if n.startswith(('DB','VB','RMD','SMD','SMB'))]
    windows=lambda values:[window_features(trace,a,b,motor,avb) for a,b in values]
    q=np.asarray(trace['dense_joints']);v=np.asarray(trace['dense_voltage_mV'])
    return dict(active=windows(active),off=windows([(b-5.,b) for a,b in intervals]),
        safe=bool(np.isfinite(q).all() and np.isfinite(v).all() and np.max(abs(q))<.5 and v.min()>-100 and v.max()<60))


def score(sham, inhibited, parent, target=.9):
    """Continuous amplitude/neural suppression loss; no peak-count plateau."""
    violations=[];baseline_ok=True;recovery_ok=True
    base=np.asarray(parent['active'][0]['regional_cycles_rad'])
    for rows in (sham['active']+sham['off'], inhibited['active']):
        for row in rows:
            a=np.asarray(row['regional_cycles_rad']);floor=np.maximum(.003,.8*base)
            violations.extend(np.maximum(0.,np.log(floor/np.maximum(a,1e-9)))**2)
            violations.extend(np.maximum(0.,np.log(np.maximum(a,1e-9)/(1.2*base)))**2)
            good=bool(np.all(a>=floor) and np.all(a<=1.2*base) and all(f is not None and 1.62<=f<=1.98 for f in row['frequency_hz']))
            baseline_ok=baseline_ok and good
            for f in row['frequency_hz']:
                violations.append(10. if f is None else (max(0.,abs(f-1.8)-.18)/.18)**2)
    ratios=[];neural=[];quiet=True;access=True
    for a,b in zip(inhibited['off'],sham['off']):
        ratio=a['raw_range_rad']/max(b['raw_range_rad'],1e-9)
        nr=a['motor_rms_mV']/max(b['motor_rms_mV'],1e-9)
        ratios.append(float(ratio));neural.append(float(nr))
        quiet=quiet and a['raw_range_rad']<=1e-4 and ratio<=1-target and nr<=.2
        access=access and a['avb_mean_mV']<=b['avb_mean_mV']-5.
    for a,b in zip(inhibited['active'][1:],sham['active'][1:]):
        recovery_ok=recovery_ok and bool(np.all(np.asarray(a['regional_cycles_rad'])>=.8*np.asarray(b['regional_cycles_rad'])))
    # Linear ratios keep a gradient toward stronger suppression from the outset.
    data=float(np.mean(np.asarray(ratios)**2)+.25*np.mean(np.asarray(neural)**2))
    safe=sham['safe'] and inhibited['safe']
    loss=data+20*float(np.sum(violations))+100*(not safe)+5*(not access)
    return dict(loss=loss,body_range_ratios=ratios,motor_rms_ratios=neural,
        safe=bool(safe),baseline_preserved=bool(baseline_ok),recovery_preserved=bool(recovery_ok),
        avb_inhibition_confirmed=bool(access),
        avb_component_passed=bool(safe and baseline_ok and recovery_ok and quiet and access),
        sensory_start_stop_qualified=False,swimming_qualified=False)
