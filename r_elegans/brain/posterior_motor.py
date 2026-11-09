"""Posterior motor recruitment from the network's own transmitter rhythms."""
from dataclasses import dataclass
import re
import numpy as np
from scipy.special import expit
from .body_amplitude import body_amplitude_metrics,sustained_excursions
from .joint_autonomous_fit import joint_proposal


@dataclass(frozen=True)
class MotorTargets:
    indices: np.ndarray
    weights: np.ndarray
    posterior_fraction: np.ndarray


def posterior_motor_targets(neuron_ids, effective_nmj_weights, projection):
    """Select strong posterior motor projections; signs do not set anatomy."""
    d,v=map(np.asarray,projection);w=np.asarray(effective_nmj_weights,dtype=float)
    if w.shape!=(d.shape[1],len(neuron_ids)) or np.any(w<0) or not np.isfinite(w).all():
        raise ValueError('Finite nonnegative canonical NMJ weights required')
    w=w/np.maximum(w.sum(axis=1,keepdims=True),1.)
    posterior=np.array_split(np.arange(len(d)),3)[0]
    tail=(d+v)[posterior].sum(axis=0)@w
    total=(d+v).sum(axis=0)@w
    fraction=np.divide(tail,total,out=np.zeros_like(tail),where=total>0)
    motor=np.asarray([re.fullmatch(r'(DA|DB|DD|VA|VB|VD|AS)\d+',n) is not None for n in neuron_ids])
    eligible=np.flatnonzero(motor&(fraction>=.5)&(tail>0))
    if not len(eligible):raise ValueError('No strong posterior motor projections')
    # Stable sort makes tied contributions reproducible in canonical order.
    order=eligible[np.argsort(-tail[eligible],kind='stable')]
    count=np.searchsorted(np.cumsum(tail[order]),.95*tail[order].sum())+1
    selected=order[:count]
    return MotorTargets(selected,tail[selected]/tail[selected].sum(),fraction[selected])


def rhythmic_release_sources(voltage,threshold,slope):
    release=expit((np.asarray(voltage)-np.asarray(threshold))/np.asarray(slope))
    amplitude,_=sustained_excursions(release)
    selected=np.flatnonzero(amplitude>=.01)
    return selected[np.argsort(-amplitude[selected],kind='stable')[:16]]


def posterior_motor_metrics(voltage,muscles,body,dt,body_length,neuron_ids,max_joint_angle,*,targets,projection,release_threshold,release_slope):
    report=body_amplitude_metrics(voltage,muscles,body,dt,body_length,neuron_ids,max_joint_angle)
    d,v=map(np.asarray,projection);posterior=np.array_split(np.arange(len(d)),3)[0]
    net=np.asarray(muscles)@(d[posterior]-v[posterior]).T
    drive,_=sustained_excursions(net)
    drive_median=float(np.median(drive))
    q=report['regional_joint_median_p2p_rad'][0]
    release=expit((np.asarray(voltage)[:,targets.indices]-np.asarray(release_threshold)[targets.indices])/
        np.asarray(release_slope)[targets.indices])
    release_amplitude,_=sustained_excursions(release)
    recruitment=float(targets.weights@np.minimum(np.log1p(release_amplitude/1e-5)/np.log1p(.05/1e-5),1.))
    coverage=lambda a:float(min(np.log1p(a/1e-5)/np.log1p(.003/1e-5),1.))
    qualified=bool(report['valid'] and q>=.003 and drive_median>=.003)
    report.update(score=.6*coverage(q)+.35*coverage(drive_median)+.05*recruitment if report['valid'] else -1.,
        posterior_motor_qualified=qualified,tail_amplitude_qualified=qualified,
        tail_coverage_score=coverage(q),posterior_recruitment_score=recruitment,
        posterior_net_muscle_sustained_p2p=drive.tolist(),posterior_net_muscle_median_p2p=drive_median,
        posterior_motor_release_sustained_p2p=release_amplitude.tolist(),
        curriculum_stage='posterior-motor-qualified' if qualified else 'posterior-motor-recruitment',
        interpretation='Short tail motor/muscle/bend primitive;no phase,control,body-wide or locomotion claim')
    return report


def posterior_motor_refinement_agrees(reference,refined):
    if not reference.get('posterior_motor_qualified') or not refined.get('posterior_motor_qualified'):return False
    for a,b in ((reference['regional_joint_median_p2p_rad'][0],refined['regional_joint_median_p2p_rad'][0]),
                (reference['posterior_net_muscle_median_p2p'],refined['posterior_net_muscle_median_p2p'])):
        if abs(a-b)>.25*max(a,.003):return False
    return True


def chemical_driver_paths(chemical,target,drivers):
    """Existing source→target and source→intermediate→target paths only."""
    drivers=np.asarray(drivers,dtype=int);drivers=drivers[drivers!=target]
    paths=[((target,int(s)),) for s in drivers if chemical[target,s]]
    middle=np.flatnonzero(chemical[target]);middle=middle[middle!=target]
    if len(middle) and len(drivers):
        mi,si=np.nonzero(chemical[np.ix_(middle,drivers)])
        paths.extend(((target,int(middle[m])),(int(middle[m]),int(drivers[s])))
                     for m,s in zip(mi,si) if middle[m]!=drivers[s])
    return paths


def posterior_motor_proposal(best,spec,rng,sigma,maximum_factor,*,targets,driver_cells):
    if sigma<=0 or not np.isfinite(sigma) or maximum_factor<=1 or not np.isfinite(maximum_factor):
        raise ValueError('Positive sigma and factor>1 required')
    chance=rng.random()
    if chance<.1:return joint_proposal(best,spec,rng,sigma,maximum_factor)
    a,b,c=spec.counts.values();noise=np.zeros(spec.size)
    ci,ei,gi=np.nonzero(spec.channels),np.nonzero(spec.chemical),np.nonzero(np.triu(spec.gap,1))
    cell=int(rng.choice(targets.indices,p=targets.weights))
    density=np.flatnonzero(ci[0]==cell)
    incoming=a+np.flatnonzero(ei[0]==cell)
    gap=a+b+np.flatnonzero((gi[0]==cell)|(gi[1]==cell))
    if chance<.55:
        paths=chemical_driver_paths(spec.chemical,cell,driver_cells)
        if paths:
            path=paths[int(rng.integers(len(paths)))];lookup={edge:a+i for i,edge in enumerate(zip(*ei))}
            selected=np.asarray([lookup[edge] for edge in path])
            noise[selected]=rng.choice([-1.,1.])*rng.uniform(1.,2.)
            # Joint density perturbation at the recruited cell, not a forced input.
            selected=rng.choice(density,size=min(4,len(density)),replace=False)
            noise[selected]=rng.normal(size=len(selected))
        elif len(incoming):
            selected=rng.choice(incoming,size=min(16,len(incoming)),replace=False)
            noise[selected]=rng.normal()
        else:noise[density]=rng.normal(size=len(density))
    elif chance<.8 and len(incoming):
        selected=rng.choice(incoming,size=min(32,len(incoming)),replace=False)
        noise[selected]=rng.normal()
        selected=rng.choice(density,size=min(4,len(density)),replace=False)
        noise[selected]=rng.normal(size=len(selected))
    else:
        noise[density]=rng.normal(size=len(density))
        if len(gap):
            selected=rng.choice(gap,size=min(16,len(gap)),replace=False)
            noise[selected]=rng.normal()
    return np.clip(best+2*sigma*noise,-np.log(maximum_factor),np.log(maximum_factor))
