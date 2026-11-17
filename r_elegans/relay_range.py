"""E087 restore AIYL operating range and sensory responsiveness, not stopping."""
import numpy as np
from scipy.special import expit
from . import rial_sensory
from .rial_release import interpolated_frequency

RUNGS = (.02, .05, .10)  # Required upper release headroom, not motor attenuation.
CUE_SCALE = .001  # Minimum absolute mean release change = headroom * this.


def direction_consistent(rows):
    values=[x for row in rows for x in row['metrics']['relay_response_mean_changes']]
    return bool(values and (all(x>0 for x in values) or all(x<0 for x in values)))


def parameter_groups(spec, ids, origin, lower, upper):
    ci, ei = np.nonzero(spec.channels), np.nonzero(spec.chemical)
    nc = len(ci[0]); groups = []
    def add(name, ix):
        ix = np.asarray(ix, dtype=int)
        if not len(ix): return
        lo = max(-.4, float(np.max(lower[ix]-origin[ix])))
        hi = min(.4, float(np.min(upper[ix]-origin[ix])))
        if lo > 1e-10 or hi < -1e-10: raise ValueError('Parent outside original bounds')
        if hi-lo > 1e-8:
            groups.append(dict(name=name, indices=ix.tolist(), lower=min(0.,lo), upper=max(0.,hi)))
    for name in ('AWAL', 'AIYL'):
        for label, columns in [('inward',[6,7,8,14]),('outward',[0,1,2,3,4,5,9,10,11,12,13,15])]:
            add('density:'+name+':'+label,np.flatnonzero((ci[0]==ids.index(name)) & np.isin(ci[1],columns)))
    for pre in np.unique(ei[1][ei[0]==ids.index('AIYL')]):
        add('chemical:'+ids[pre]+'->AIYL',nc+np.flatnonzero((ei[0]==ids.index('AIYL')) & (ei[1]==pre)))
    add('chemical:AIYL->RIAL',nc+np.flatnonzero((ei[0]==ids.index('RIAL')) & (ei[1]==ids.index('AIYL'))))
    return groups


def features(trace, ids, variant, threshold, slope):
    result = rial_sensory.features(trace,ids,variant,frequency_estimator=interpolated_frequency)
    cell = ids.index('AIYL')
    for category in ('active','response','late'):
        for row in result[category]:
            a,b = row['window']; t = trace['dense_time_s']; use = (t>=a)&(t<b)
            release = expit((trace['dense_voltage_mV'][use,cell]-threshold)/slope)
            row['AIYL_release_mean'] = float(release.mean())
            row['AIYL_release_q01'],row['AIYL_release_q99'] = map(float,np.quantile(release,[.01,.99]))
            row['AIYL_release_half_means'] = [float(x.mean()) for x in np.array_split(release,2)]
    return result


def score(paired, parent, target, margin=1., *, response_aggregation='mean'):
    if response_aggregation not in ('mean','weakest'):raise ValueError('Unknown relay response aggregation')
    body = rial_sensory.score(paired,parent,target,margin)
    body_ratios = body['response_amplitude_ratios']
    old_error = np.mean([((r-(1-target*margin))/target)**2 for r in body_ratios])
    a,b = paired['no_cue'],paired['cue']; cue_min = target*CUE_SCALE
    penalty = 0.; operating = True; training_range = True
    for row in a['active']+a['response']+a['late']+b['active']:
        lo,hi = row['AIYL_release_q01'],row['AIYL_release_q99']
        operating = operating and lo>=.05 and hi<=1-target
        training_range = training_range and lo>=.05 and hi<=1-target*margin
        penalty += (max(0.,.05-lo)/.05)**2+(max(0.,hi-(1-target*margin))/target)**2
    changes = [x['AIYL_release_mean']-y['AIYL_release_mean'] for x,y in zip(b['response'],a['response'])]
    directions = [x-y for row,ref in zip(b['response'],a['response'])
                  for x,y in zip(row['AIYL_release_half_means'],ref['AIYL_release_half_means'])]
    consistent = bool(all(x>0 for x in directions) or all(x<0 for x in directions))
    recovery = True
    for row,ref in zip(b['active'],a['active']):
        delta = abs(row['AIYL_release_mean']-ref['AIYL_release_mean'])
        recovery = recovery and delta<=max(1e-6,.25*cue_min)
        penalty += (max(0.,delta-max(1e-6,.25*cue_min))/cue_min)**2
    not_railed = all(x['AIYL_release_q01']>=.005 and x['AIYL_release_q99']<=.995 for x in b['response'])
    penalty += sum((max(0.,.005-x['AIYL_release_q01'])/.005)**2 +
                   (max(0.,x['AIYL_release_q99']-.995)/.005)**2 for x in b['response'])
    error = (((min(abs(x) for x in changes)-cue_min*margin)/cue_min)**2 if response_aggregation=='weakest'
             else np.mean([((abs(x)-cue_min*margin)/cue_min)**2 for x in changes]))
    body.update(loss=float(max(0.,body['loss']-old_error)+penalty+error+float(not consistent)),
        body_response_amplitude_ratios=body_ratios,
        response_amplitude_ratios=[1-abs(x) for x in changes],
        late_amplitude_ratios=[1-abs(x['AIYL_release_mean']-y['AIYL_release_mean']) for x,y in zip(b['late'],a['late'])],
        metric='one_minus_absolute_AIYL_mean_release_change',
        relay_response_aggregation=response_aggregation,
        relay_response_mean_changes=changes,relay_minimum_cue_change=cue_min,
        relay_operating_range_passed=bool(operating),relay_recovery_preserved=bool(recovery),
        relay_response_direction_consistent=consistent,relay_response_not_railed=bool(not_railed))
    body['baseline_and_recovery_preserved'] = bool(body['baseline_and_recovery_preserved'] and recovery)
    eligible = body['safe'] and body['baseline_and_recovery_preserved'] and body['useful_response_preserved']
    body['partial_attenuation_passed'] = bool(eligible and operating and consistent and not_railed and all(abs(x)>=cue_min for x in changes))
    body['training_margin_met'] = bool(eligible and training_range and consistent and not_railed and all(abs(x)>=cue_min*margin for x in changes))
    return body
