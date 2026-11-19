"""E093/E094 sensory-dependent motion suppression, recovery and repeated restart.

Cue-off recovery is restart evidence; initial sensory initiation from rest is
not tested here and never receives full start/stop qualification.
E094 adds the right sensory/relay path and a worst-region loss term.
"""
import numpy as np
from . import relay_range
from .rial_sensory import protocol

RUNGS=(.01,.02,.05,.10,.25,.50,.75,.95)


def parameter_groups(spec,ids,origin,lower,upper,bilateral=False):
    groups=relay_range.parameter_groups(spec,ids,origin,lower,upper)
    ci=np.nonzero(spec.channels);ei=np.nonzero(spec.chemical);nc=len(ci[0])
    def add(name,indices):
        ix=np.asarray(indices,dtype=int)
        if not len(ix):return
        lo=max(-.4,float(np.max(lower[ix]-origin[ix])));hi=min(.4,float(np.min(upper[ix]-origin[ix])))
        if lo>1e-10 or hi<-1e-10:raise ValueError('Motion parent outside declared prior')
        if hi-lo>1e-8:groups.append(dict(name=name,indices=ix.tolist(),lower=min(lo,0.),upper=max(hi,0.)))
    inward,outward=[6,7,8,14],[0,1,2,3,4,5,9,10,11,12,13,15]
    for label,columns in [('inward',inward),('outward',outward)]:
        add('density:RIAL:'+label,np.flatnonzero((ci[0]==ids.index('RIAL'))&np.isin(ci[1],columns)))
    for post in sorted({int(i) for i,j in zip(*ei) if j==ids.index('RIAL') and ids[i].startswith('SMD')}):
        add('chemical:RIAL->'+ids[post],nc+np.flatnonzero((ei[0]==post)&(ei[1]==ids.index('RIAL'))))
    if bilateral:
        for name in ('AWAR','AIYR','RIAR'):
            for label,columns in [('inward',inward),('outward',outward)]:
                add('density:'+name+':'+label,np.flatnonzero((ci[0]==ids.index(name))&np.isin(ci[1],columns)))
        for pre in np.unique(ei[1][ei[0]==ids.index('AIYR')]):
            add('chemical:'+ids[pre]+'->AIYR',nc+np.flatnonzero((ei[0]==ids.index('AIYR'))&(ei[1]==pre)))
        add('chemical:AIYR->RIAR',nc+np.flatnonzero((ei[0]==ids.index('RIAR'))&(ei[1]==ids.index('AIYR'))))
        for post in sorted({int(i) for i,j in zip(*ei) if j==ids.index('RIAR') and ids[i].startswith('SMD')}):
            add('chemical:RIAR->'+ids[post],nc+np.flatnonzero((ei[0]==post)&(ei[1]==ids.index('RIAR'))))
    ix=[i for g in groups for i in g['indices']]
    if len(ix)!=len(set(ix)):raise ValueError('Overlapping motion groups')
    return groups


def features(trace,ids,variant,threshold,slope,nodes=None):
    result=relay_range.features(trace,ids,variant,threshold,slope)
    for category in ('active','response','late'):
        for row in result[category]:
            a,b=row['window'];use=(trace['dense_time_s']>=a)&(trace['dense_time_s']<b)
            t=trace['dense_time_s'][use];q=trace['dense_joints'][use]
            rate=np.diff(q,axis=0)/np.diff(t)[:,None]
            row['regional_angular_speed_rms_rad_s']=[float(np.sqrt(np.mean(rate[:,region]**2))) for region in np.array_split(np.arange(q.shape[1]),3)]
            if nodes:
                from scipy.special import expit
                release={n:expit((trace['dense_voltage_mV'][use][:,ids.index(n)]-th)/sl) for n,(th,sl) in nodes.items()}
                row['node_release_mean']={n:float(r.mean()) for n,r in release.items()}
                row['node_release_q01_q99']={n:list(map(float,np.quantile(r,[.01,.99]))) for n,r in release.items()}
    return result


def combine_replays(probes):
    """Concatenate diagnostic release overrides from several matched source neurons."""
    import jax.numpy as jnp
    from .brain.conductance_network import ChemicalReleaseOverride
    def probe(t):
        parts=[p(t) for p in probes]
        return ChemicalReleaseOverride(*(jnp.concatenate([getattr(x,k) for x in parts]) for k in ('post_indices','pre_indices','release_probability')),parts[0].active)
    return probe


def score(paired,parent,target,margin=1.,worst_region=False,node_target=None):
    if target not in RUNGS or not 1<=margin<=1.1:raise ValueError('Invalid motion rung or margin')
    a,b=paired['no_cue'],paired['cue'];base=parent['active'][0];penalty=0.;preserved=True
    def preserve(row,ref):
        nonlocal penalty,preserved
        for key,floor in [('regional_cycles_rad',.003),('regional_angular_speed_rms_rad_s',1e-8)]:
            x=np.asarray(row[key]);v=np.asarray(ref[key]);lo=np.maximum(floor,.9*v);hi=1.1*v
            preserved=preserved and bool(np.all(x>=lo)&np.all(x<=hi))
            penalty+=float(np.sum((np.maximum(0.,lo-x)/np.maximum(v,floor))**2)+np.sum((np.maximum(0.,x-hi)/np.maximum(v,floor))**2))
        for f,f0 in zip(row['frequency_hz'],ref['frequency_hz']):
            okay=f is not None and f0 is not None and 1.62<=f<=1.98 and min(f,f0)>0
            delta=abs(np.log(f/f0)) if okay else 1.
            preserved=preserved and okay and delta<=np.log(1.05)
            penalty+=(max(0.,delta-np.log(1.05))/.05)**2+(0. if okay else 10.)
    for category in ('active','response','late'):
        for row in a[category]:preserve(row,base)
    recovery_ratios=[];recovery_tolerance=max(1e-5,.25*target)
    for row,ref in zip(b['active'],a['active']):
        preserve(row,ref)
        ratio=np.asarray(row['regional_angular_speed_rms_rad_s'])/np.maximum(ref['regional_angular_speed_rms_rad_s'],1e-9)
        recovery_ratios.append(ratio.tolist());delta=float(np.max(abs(ratio-1)))
        preserved=preserved and delta<=recovery_tolerance
        penalty+=(max(0.,delta-recovery_tolerance)/max(target,.01))**2
    ratios=[];cycles=[]
    for row,ref in zip(b['response'],a['response']):
        ratios.append((np.asarray(row['regional_angular_speed_rms_rad_s'])/np.maximum(ref['regional_angular_speed_rms_rad_s'],1e-9)).tolist())
        cycles.append((np.asarray(row['regional_cycles_rad'])/np.maximum(ref['regional_cycles_rad'],1e-9)).tolist())
    # Relay headroom, cue response and recovery remain hard preservation checks.
    relay=relay_range.score(paired,parent,.02,1.,response_aggregation='weakest')
    relay_ok=bool(relay['relay_operating_range_passed'] and relay['relay_recovery_preserved'] and
        relay['relay_response_direction_consistent'] and relay['relay_response_not_railed'] and
        all(abs(x)>=2e-5 for x in relay['relay_response_mean_changes']))
    relay_penalty=0.
    for row in a['active']+a['response']+a['late']+b['active']:
        relay_penalty+=(max(0.,row['AIYL_release_q99']-.98)/.02)**2+(max(0.,.05-row['AIYL_release_q01'])/.05)**2
    relay_penalty+=sum((max(0.,2e-5-abs(x))/2e-5)**2 for x in relay['relay_response_mean_changes'])
    if not relay['relay_recovery_preserved'] or not relay['relay_response_direction_consistent'] or not relay['relay_response_not_railed']:relay_penalty+=1.
    values=np.asarray(ratios);aim=max(0.,1-target*margin)
    deficit=(np.maximum(0.,values-aim)/target)**2
    objective=float(np.mean(deficit)+(np.max(deficit) if worst_region else 0.))
    safe=bool(a['safe'] and b['safe']);qualified=bool(safe and preserved and relay_ok and np.all(values<=1-target))
    node={}
    if node_target:
        names=sorted(a['response'][0]['node_release_mean'])
        change=float(np.mean([np.mean([rb['node_release_mean'][n]-ra['node_release_mean'][n] for n in names]) for ra,rb in zip(a['response'],b['response'])]))
        term=(max(0.,node_target-change)/node_target)**2;headroom=0.
        for row in a['active']+a['response']+a['late']+b['active']+b['response']:
            for n in names:
                lo,hi=row['node_release_q01_q99'][n];headroom+=(max(0.,hi-.98)/.02)**2+(max(0.,.02-lo)/.02)**2
        node=dict(node_release_change=change,node_target=node_target,node_term=float(term),node_headroom_penalty=float(headroom))
        objective+=float(term)+100*float(headroom)
    return dict(loss=float(objective+100*penalty+100*relay_penalty+1000*(not safe)),target=target,margin=margin,**({'worst_region_term':True} if worst_region else {}),**node,
        metric='maximum_regional_joint_angular_speed_rms_ratio',regional_motion_ratios=ratios,
        cue_off_recovery_ratios=recovery_ratios,cue_recovery_tolerance=recovery_tolerance,
        response_amplitude_ratios=[max(r) for r in ratios],regional_cycle_ratios=cycles,
        relay_response_mean_changes=relay['relay_response_mean_changes'],relay_component_preserved=relay_ok,
        relay_operating_range_passed=relay['relay_operating_range_passed'],relay_recovery_preserved=relay['relay_recovery_preserved'],
        safe=safe,baseline_and_recovery_preserved=bool(preserved),useful_response_preserved=True,
        partial_attenuation_passed=qualified,training_margin_met=bool(qualified and np.all(values<=aim)),
        rest_component_passed=bool(qualified and target>=.95 and np.max(cycles)<=.1),
        full_start_stop_qualified=False,swimming_qualified=False,
        scope='Sensory cue-on movement suppression and cue-off recovery; initial sensory initiation from rest untested')


def panel_assessment(rows,target):
    complete=len(rows)==11 and all(r.get('status')=='completed' for r in rows)
    if not complete:return dict(complete=False,engineering_criteria_met=False,full_start_stop_qualified=False)
    normal=rows[:8];same=len({r['checkpoint_sha256'] for r in rows})==1
    checks=[]
    for row in rows[6:8]:
        delta=float(np.max(abs(np.asarray(normal[0]['metrics']['regional_motion_ratios'])-row['metrics']['regional_motion_ratios'])))
        checks.append(dict(grid=row['grid'],difference=delta,passed=delta<=.25*target))
    blocked=rows[8]['metrics'];block=bool(blocked['safe'] and blocked['baseline_and_recovery_preserved'] and
        np.max(abs(np.asarray(blocked['regional_motion_ratios'])-1))<=.25*target and
        max(abs(x) for x in blocked['relay_response_mean_changes'])<=5e-6)
    dependence=bool(all((r.get('mediation') or {}).get('cue_attenuation_reduced') and
        (r.get('mediation') or {}).get('baseline_preserved') for r in rows[9:11]))
    passed=bool(same and all(r['metrics']['partial_attenuation_passed'] for r in normal) and
        all(c['passed'] for c in checks) and block and dependence)
    return dict(complete=True,same_checkpoint=same,numerical_checks=checks,sensory_block_passed=block,
        control_interneuron_dependence_passed=dependence,engineering_criteria_met=passed,target=target,
        rest_restart_component_qualified=bool(passed and all(r['metrics']['rest_component_passed'] for r in normal)),
        full_start_stop_qualified=False,scope='Partial suppression or rest/restart component only; no initial sensory initiation or swimming qualification')
