"""Posterior amplitude primitive with a small autonomous recruitment signal."""
import numpy as np
from .body_amplitude import body_amplitude_metrics, sustained_excursions
from .joint_autonomous_fit import joint_proposal


def tail_focus_cells(nmj_weights, projection):
    dorsal, ventral = map(np.asarray, projection)
    posterior = np.array_split(np.arange(len(dorsal)), 3)[0]
    muscles = (dorsal[posterior]+ventral[posterior]).sum(axis=0)>0
    return np.flatnonzero(np.asarray(nmj_weights)[muscles].sum(axis=0)>0)


def tail_amplitude_metrics(voltage, muscles, body, dt, body_length, neuron_ids, max_joint_angle, *, focus_cells):
    report = body_amplitude_metrics(voltage, muscles, body, dt, body_length, neuron_ids, max_joint_angle)
    excursions, _ = sustained_excursions(voltage)
    focus = np.asarray(focus_cells, dtype=int)
    if not len(focus):raise ValueError('Tail motor-neuron inputs required')
    # This bounded auxiliary signal is heuristic; voltage never qualifies motion.
    recruitment = float(np.mean(np.minimum(np.log1p(excursions[focus]/.05)/np.log1p(5./.05), 1.)))
    amplitude = report['regional_joint_median_p2p_rad'][0]
    coverage = float(min(np.log1p(amplitude/1e-5)/np.log1p(.003/1e-5), 1.))
    qualified = bool(report['valid'] and amplitude>=.003)
    report.update(score=.95*coverage+.05*recruitment if report['valid'] else -1.,
        posterior_recruitment_score=recruitment, tail_coverage_score=coverage,
        tail_amplitude_qualified=qualified, curriculum_stage='tail-qualified' if qualified else 'tail-amplitude',
        interpretation='Short-assay tail amplitude only; body-wide qualification reported separately, no locomotion claim')
    return report


def tail_refinement_agrees(reference, refined):
    if not reference.get('tail_amplitude_qualified') or not refined.get('tail_amplitude_qualified'):return False
    a, b = (m['regional_joint_median_p2p_rad'][0] for m in (reference,refined))
    return abs(a-b)<=.25*max(a,.003)


def tail_proposal(best, spec, rng, sigma, maximum_factor, *, focus_cells):
    if sigma<=0 or not np.isfinite(sigma) or maximum_factor<=1 or not np.isfinite(maximum_factor):
        raise ValueError('Positive sigma and finite factor>1 required')
    focus = np.asarray(focus_cells, dtype=int)
    if not len(focus):raise ValueError('Tail focus required')
    if rng.random()<.1:
        return joint_proposal(best,spec,rng,sigma,maximum_factor)
    a,b,c = spec.counts.values()
    ci,ei,gi = np.nonzero(spec.channels),np.nonzero(spec.chemical),np.nonzero(np.triu(spec.gap,1))
    cell = int(rng.choice(focus)); mode = int(rng.integers(4))
    if mode==0:
        pool = np.flatnonzero(ci[0]==cell)
    elif mode==1:
        pool = a+np.flatnonzero(ei[0]==cell)
    elif mode==2:
        pool = a+b+np.flatnonzero((gi[0]==cell)|(gi[1]==cell))
    else:
        channel = int(rng.choice(np.unique(ci[1])))
        pool = np.flatnonzero(np.isin(ci[0],focus)&(ci[1]==channel))
    if not len(pool):pool=np.flatnonzero(ci[0]==cell)
    selected = rng.choice(pool,size=min(8,len(pool)),replace=False)
    noise = np.zeros(spec.size);noise[selected]=rng.normal(size=len(selected))
    return np.clip(best+sigma*noise,-np.log(maximum_factor),np.log(maximum_factor))
