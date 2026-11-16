"""E084 causal RIAL membrane versus existing chemical branch interventions."""
import numpy as np
from .smd_control import SMD,protocol,compare as motor_compare


def cases():
    rows=[dict(name='sham',kind='sham',targets=[],conductance_nS=0.)]
    for g in (.01,.03,.1):rows.append(dict(name=f'RIAL-{g:g}',kind='membrane',targets=['RIAL'],conductance_nS=g))
    rows.append(dict(name='RIAR-0.1',kind='membrane',targets=['RIAR'],conductance_nS=.1))
    for name,targets,value in [('RIAL-SMD-mean',SMD,'mean'),('RIAL-ventral-mean',SMD[2:],'mean'),('RIAL-ventral-zero',SMD[2:],'zero')]:
        rows.append(dict(name=name,kind='branch',source='RIAL',targets=list(targets),hold=value,conductance_nS=0.))
    return rows


def membrane_probe(ids,case,variant):
    import jax.numpy as jnp
    from .brain.conductance_network import MembraneConductanceInput
    if case['kind']!='membrane' or not set(case['targets'])<={'RIAL','RIAR'}:raise ValueError('Unknown RIA probe')
    g=float(case['conductance_nS'])
    if not np.isfinite(g) or not 0<g<=.1:raise ValueError('Probe outside diagnostic envelope')
    mask=np.asarray([g if n in case['targets'] else 0. for n in ids]);mask=jnp.asarray(mask)
    _,intervals,_=protocol(variant)
    def probe(t):
        active=jnp.asarray(False)
        for a,b in intervals:active=active|((t>=a)&(t<b))
        conductance=mask*active
        return MembraneConductanceInput(conductance,-80.*conductance)
    return probe


def branch_probe(ids,chemical_mask,baseline_release,case,variant):
    import jax.numpy as jnp
    from .brain.conductance_network import ChemicalReleaseOverride
    if case['kind']!='branch' or case['source']!='RIAL' or not set(case['targets'])<=set(SMD) or case['hold'] not in ('mean','zero'):
        raise ValueError('Only existing RIAL-to-SMD chemical branches supported')
    source=ids.index('RIAL');mean=np.asarray(baseline_release)
    if mean.shape!=(len(ids),) or not np.isfinite(mean).all() or np.any((mean<0)|(mean>1)):raise ValueError('Valid release probabilities required')
    posts=[ids.index(n) for n in case['targets'] if chemical_mask[ids.index(n),source]>0]
    if not posts:raise ValueError('No existing chemical edges; refusing to invent a branch')
    if len(posts)!=len(set(posts)):raise ValueError('Duplicate override edge')
    pre=jnp.full(len(posts),source,dtype=jnp.int32);post=jnp.asarray(posts,dtype=jnp.int32)
    value=jnp.full(len(posts),float(mean[source]) if case['hold']=='mean' else 0.)
    _,intervals,_=protocol(variant)
    def probe(t):
        active=jnp.asarray(False)
        for a,b in intervals:active=active|((t>=a)&(t<b))
        return ChemicalReleaseOverride(post,pre,value,active)
    return probe


def compare(trace,sham,ids,case,variant='nominal'):
    # Common motor windows and descriptive SMD metrics; do not demand a RIAL mean
    # voltage shift for a branch hold, which deliberately leaves that cell live.
    m=motor_compare(trace,sham,ids,dict(targets=list(SMD[2:])),variant)
    t=trace['dense_time_s'];_,intervals,_=protocol(variant);rows=[]
    ri=[ids.index(n) for n in ('RIAL','RIAR')]
    for (_,b),off in zip(intervals,m['off_windows']):
        use=(t>=b-5)&(t<b);v=trace['dense_voltage_mV'][use];v0=sham['dense_voltage_mV'][use]
        rows.append(dict(window=[b-5,b],RIA_names=['RIAL','RIAR'],mean_voltage_mV=v[:,ri].mean(axis=0).tolist(),
            sham_mean_voltage_mV=v0[:,ri].mean(axis=0).tolist(),voltage_std_mV=v[:,ri].std(axis=0).tolist(),
            sham_voltage_std_mV=v0[:,ri].std(axis=0).tolist()))
    m['RIA_windows']=rows
    m['intervention_type']=case['kind']
    m['ventral_SMD_metrics']=m.pop('off_windows')
    # The inherited gate describes direct SMD inhibition and is inapplicable here.
    m.pop('reversible_neural_target');m.pop('scope')
    for row in m['ventral_SMD_metrics']:
        row['ventral_SMD_rms_ratio']=row.pop('target_voltage_rms_ratio')
        row['ventral_SMD_mean_shifts_mV']=row.pop('target_mean_shifts_mV')
        row['ventral_SMD_std_mV']=row.pop('target_std_mV')
        row['sham_ventral_SMD_std_mV']=row.pop('sham_target_std_mV')
        row.pop('inhibition_confirmed')
    if case['kind']=='membrane':
        targets=[ri.index(ids.index(n)) for n in case['targets']]
        m['membrane_inhibition_confirmed']=all(all(r['mean_voltage_mV'][i]-r['sham_mean_voltage_mV'][i]<=-5 for i in targets) for r in rows)
    else:m['membrane_inhibition_confirmed']=None
    m['scope']='Imposed RIA membrane or existing-edge release intervention; no learned stopping or isolated pacemaker proof'
    return m


def choose_membrane(comparisons):
    rows=[r for r in comparisons if r['case']['kind']=='membrane' and r['case']['targets']==['RIAL'] and r['metrics']['pre_match'] and r['metrics']['safe']]
    if not rows:return None
    return min(rows,key=lambda r:(not r['metrics']['membrane_inhibition_confirmed'],not r['metrics']['active_and_recovery_preserved'],
        max(w['body_range_ratio'] for w in r['metrics']['ventral_SMD_metrics'])))['case']
