"""Interneuron conductance probes: diagnostics, never a learned behavior policy."""
import numpy as np
from .behavior_control import CONTROL_GROUPS
from .hybrid_assessment import cycle_excursions,peak_frequency


def cases():
    result=[dict(name='sham',group=None,conductance_nS=0.,reversal_mV=0.)]
    for group in CONTROL_GROUPS:
        for label,reversal in [('excite',0.),('inhibit',-80.)]:
            for strength in (.01,.1):
                result.append(dict(name=f'{group}-{label}-{strength:g}',group=group,conductance_nS=strength,reversal_mV=reversal))
    return result


def membrane_probe(ids,case):
    import jax.numpy as jnp
    from .brain.conductance_network import MembraneConductanceInput
    if case['group'] is not None and case['group'] not in CONTROL_GROUPS:raise ValueError('Unknown control group')
    strength=float(case['conductance_nS']);reversal=float(case['reversal_mV'])
    if not np.isfinite([strength,reversal]).all() or strength<0 or strength>.1 or not -80<=reversal<=0:
        raise ValueError('Probe outside declared diagnostic envelope')
    mask=np.zeros(len(ids))
    for name in CONTROL_GROUPS.get(case['group'],()):mask[ids.index(name)]=1.
    mask=jnp.asarray(mask)
    def probe(t):
        g=mask*strength*jnp.asarray((t>=30.)&(t<40.))
        return MembraneConductanceInput(g,g*reversal)
    return probe


def compare(trace,sham):
    t=np.asarray(trace['dense_time_s']);q=np.asarray(trace['dense_joints']);v=np.asarray(trace['dense_voltage_mV'])
    if not np.array_equal(t,sham['dense_time_s']) or q.shape!=np.asarray(sham['dense_joints']).shape:
        raise ValueError('Matched diagnostic time grids required')
    if not np.isfinite(q).all() or not np.isfinite(v).all():raise ValueError('Nonfinite diagnostic trajectory')
    result={}
    for name,(a,b) in {'pre':(20.,30.),'during':(35.,40.),'recovery':(50.,60.)}.items():
        use=(t>=a)&(t<b)
        if use.sum()<3:raise ValueError('Missing diagnostic time window')
        shape=q[use];reference=np.asarray(sham['dense_joints'])[use]
        amplitudes=cycle_excursions(shape,late_only=False)['regional_cycle_excursion_rad']
        base=cycle_excursions(reference,late_only=False)['regional_cycle_excursion_rad']
        frequencies=[]
        for region in np.array_split(np.arange(shape.shape[1]),3):
            j=int(region[np.argmax(np.ptp(shape[:,region],axis=0))])
            frequencies.append(peak_frequency(t[use],shape[:,j],max(1e-6,.1*np.ptp(shape[:,j]))))
        result[name]=dict(frequencies_hz=frequencies,maximum_raw_joint_range_rad=float(np.max(np.ptp(shape,axis=0))),
            sham_maximum_raw_joint_range_rad=float(np.max(np.ptp(reference,axis=0))),
            regional_cycles_rad=list(map(float,amplitudes)),sham_regional_cycles_rad=list(map(float,base)),
            voltage_maximum_difference_mV=float(np.max(abs(v[use]-np.asarray(sham['dense_voltage_mV'])[use]))))
    pre=q[t<30]-np.asarray(sham['dense_joints'])[t<30]
    pv=v[t<30]-np.asarray(sham['dense_voltage_mV'])[t<30]
    baseline=bool(np.max(abs(pre))<=1e-8 and np.max(abs(pv))<=1e-8 and np.all(np.asarray(result['pre']['regional_cycles_rad'])>=.003))
    ratio=result['during']['maximum_raw_joint_range_rad']/max(result['during']['sham_maximum_raw_joint_range_rad'],1e-12)
    recovery=bool(np.all(np.asarray(result['recovery']['regional_cycles_rad'])>=np.maximum(.003,.8*np.asarray(result['recovery']['sham_regional_cycles_rad']))))
    recovery=recovery and all(f is not None and 1.62<=f<=1.98 for f in result['recovery']['frequencies_hz'])
    quiet=bool(result['during']['maximum_raw_joint_range_rad']<=1e-4)
    safe=bool(np.max(abs(q))<.5 and np.min(v)>-100 and np.max(v)<60)
    return dict(windows=result,pre_cue_match=baseline,during_to_sham_raw_range_ratio=float(ratio),
        quiet_during_probe=quiet,recovery_amplitude_preserved=recovery,safety_passed=safe,
        reversible_diagnostic_suppression=bool(baseline and quiet and recovery and safe),
        learned_stopping_qualified=False,scope='Imposed interneuron conductance; diagnostic body-motion suppression only, not proof of neural oscillator silencing')
